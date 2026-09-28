# Deployment

Two documented paths. Path 1 is what this project actually deploys with (chosen for cost -- see [docs/decisions](decisions/)); Path 2 is PLAN.md's original Cloud Run + managed Postgres suggestion, kept here as a documented alternative, not run by default.

## Path 1 (recommended): free tier, three providers

Streamlit Community Cloud (UI) + Cloud Run (API, scales to zero) + Neon or Supabase (Postgres with pgvector, auto-suspends when idle). At demo-level traffic this is **$0/month**. The tradeoff: the database and the API's Cloud Run instance both cold-start after idle time, so the first request after a while takes a few seconds longer.

### 1. Database (Neon or Supabase)
1. Create a free project at [neon.tech](https://neon.tech) or [supabase.com](https://supabase.com).
2. Enable the `vector` extension if the provider doesn't do it automatically (Neon/Supabase both support pgvector; the extension itself is created by this project's own migration, not a manual step).
3. Copy the connection string -- this is `DATABASE_URL`.
4. Run migrations and load data against it once, from your machine:
   ```
   DATABASE_URL=<connection string> uv run python -m filings_rag.migrate
   DATABASE_URL=<connection string> make ingest load
   ```

### 2. API (Cloud Run)
```
gcloud run deploy filings-rag-api \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --min-instances 0 \
  --set-env-vars DATABASE_URL=<connection string> \
  --set-secrets VOYAGE_API_KEY=voyage-api-key:latest,ANTHROPIC_API_KEY=anthropic-api-key:latest,OPENROUTER_API_KEY=openrouter-api-key:latest
```
`--source .` builds from the repo's `Dockerfile` directly (Cloud Run's buildpack path). `--min-instances 0` is what makes idle time free. Put real API keys in Secret Manager (`gcloud secrets create ...`), not `--set-env-vars` -- that flag's values are visible in the Cloud Run console and `gcloud run services describe` output.

### 3. UI (Streamlit Community Cloud)
1. Push this repo to GitHub (already required for Cloud Run's `--source` deploy above).
2. At [share.streamlit.io](https://share.streamlit.io), create a new app pointing at `ui/app.py` on your `main` branch.
3. In the app's settings, add `FILINGS_RAG_API_URL` = the Cloud Run URL from step 2.

## Path 2: Cloud Run + managed Postgres (PLAN.md's original suggestion)

Same Cloud Run deploy as above, but the database is Cloud SQL for Postgres (or another managed provider) instead of Neon/Supabase. This is the "production-shaped" option -- a real always-on instance with predictable performance and no cold-start -- but it is **not actually cheap**: even the smallest Cloud SQL tier runs roughly $7-10+/month, billed whether or not anyone uses it, since Cloud SQL doesn't scale to zero the way Cloud Run and serverless Postgres do.

```
gcloud sql instances create filings-rag-db \
  --database-version=POSTGRES_16 \
  --tier=db-f1-micro \
  --region=us-central1
gcloud sql databases create filings --instance=filings-rag-db
```
Cloud SQL's Postgres supports the `vector` extension; enable it the same way (`CREATE EXTENSION IF NOT EXISTS vector`, already in this project's own migration). The UI can either stay on Streamlit Community Cloud (pointed at this Cloud Run URL instead) or move to a second Cloud Run service running `streamlit run ui/app.py` in its own container, if keeping everything inside one GCP project matters more than free hosting for the UI.

## A real Docker gotcha, found by actually running the image

`.env`'s values are quoted (`VOYAGE_API_KEY="pa-..."`), which `pydantic-settings`' own `.env` parser strips correctly. `docker run --env-file .env` does **not** strip those quotes -- it passes the literal value, quote characters included, silently turning a valid API key into an invalid one (401s) and a valid URL into a broken hostname (DNS failures). Found by actually running the built image against `.env`, not by reading the Dockerfile.

Don't pass `.env` directly to `--env-file` for local Docker testing. Either strip quotes first (`sed -E 's/^([A-Z_]+)="(.*)"$/\1=\2/' .env > .env.docker`) or set values with `-e KEY=value` individually. This doesn't affect the recommended deployment paths above -- Cloud Run's `--set-env-vars`/`--set-secrets` and Streamlit Community Cloud's settings UI both take unquoted values directly, so the quoting mismatch never comes up there.

## What's not automated here
Provisioning either path (the actual `gcloud`/Neon/Supabase/Streamlit account setup) is infrastructure work outside this repo -- these are real commands to run, not something executed as part of building this project. The Dockerfile, `filings_rag.migrate`, and this document are what a user needs to actually do it.
