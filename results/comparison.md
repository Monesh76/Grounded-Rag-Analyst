# Experiment grid comparison (PLAN.md P6)

Cost is generation-only and comparable across all rows. Faithfulness/correctness ('n/a' below) need a separate judge pass -- see 'Judged'.

| Run | Chunking | Retrieval | Model | Judged | Recall@6 | MRR | Faithfulness | Correctness | Refusal accuracy | p95 latency (ms) | Generation cost/1K |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | fixed-512 | dense | anthropic/claude-haiku-4.5 | yes | 0.838 | 0.632 | 0.800 | 0.733 | 0.840 | 4096 | $4.02 |
| B | section-aware | dense | anthropic/claude-haiku-4.5 | yes | 0.787 | 0.602 | 0.850 | 0.760 | 0.860 | 5509 | $5.37 |
| C | section-aware | hybrid | anthropic/claude-haiku-4.5 | yes | 0.787 | 0.572 | 0.850 | 0.745 | 0.860 | 3426 | $5.44 |
| D | section-aware | hybrid_rerank | anthropic/claude-haiku-4.5 | no | 0.850 | 0.619 | n/a | n/a | 0.760 | 9211 | $5.48 |
| E | section-aware | hybrid_rerank | openai/gpt-4o-mini | no | 0.850 | 0.619 | n/a | n/a | 0.780 | 16565 | $0.69 |
