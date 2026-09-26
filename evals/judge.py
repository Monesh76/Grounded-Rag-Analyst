"""LLM-as-judge scoring (DeepEval) for eval-full: faithfulness (is every claim
supported by the retrieved context?) and answer correctness (does it match the
golden answer?). Judge is Claude Sonnet 5 via direct API -- deliberately a
different vendor/model than whichever provider generated the answer being
judged, to avoid a model favoring its own outputs. Temperature-pinned by
ClaudeProvider's own defaults, per PLAN.md's "fix the judge model and
temperature 0" for low noise between runs.
"""

from deepeval.metrics import FaithfulnessMetric, GEval
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, SingleTurnParams

from filings_rag.config import Settings
from filings_rag.generate.llm import ClaudeProvider

CORRECTNESS_CRITERIA = (
    "Determine whether the actual output is factually correct and consistent "
    "with the expected output, focusing on whether key facts, figures and "
    "conclusions match. Minor wording differences are fine as long as the "
    "substance agrees."
)


class ClaudeJudge(DeepEvalBaseLLM):
    """Adapts our own ClaudeProvider to DeepEval's judge-model interface.

    Tracks its own total_cost_usd/total_tokens: DeepEval only accrues a
    metric's cost when the wrapped model is one of its own "native" model
    integrations (see accrue_token_usage in its source), so a custom
    DeepEvalBaseLLM's spend is otherwise invisible -- a real gap found only
    after running eval-full for real and realizing the reported total cost
    silently excluded every judge call.
    """

    def __init__(self, provider: ClaudeProvider) -> None:
        self._provider = provider  # must be set before super().__init__() calls load_model()
        super().__init__(model=provider.model)
        self.total_cost_usd = 0.0
        self.total_tokens = 0

    def load_model(self) -> ClaudeProvider:
        return self._provider

    def generate(self, prompt: str) -> str:
        response = self._provider.complete(
            system="You are a careful, precise evaluation assistant.", user=prompt
        )
        self.total_cost_usd += response.cost_usd
        self.total_tokens += response.input_tokens + response.output_tokens
        return response.text

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self._provider.model


def build_judge(settings: Settings) -> ClaudeJudge:
    # Not ClaudeProvider.from_settings(): that uses llm_max_tokens, sized for a
    # normal answer. The judge needs judge_max_tokens instead -- see its
    # comment in config.py for the truncated-JSON failure that showed why.
    if not settings.anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY must be set in .env to use the Claude judge.")
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    provider = ClaudeProvider(client, settings.anthropic_model, settings.judge_max_tokens)
    return ClaudeJudge(provider)


class JudgeScores:
    def __init__(self, faithfulness: float, correctness: float) -> None:
        self.faithfulness = faithfulness
        self.correctness = correctness


def score_answer(
    judge: ClaudeJudge,
    question: str,
    answer_text: str,
    expected_answer: str,
    retrieved_texts: list[str],
) -> JudgeScores:
    test_case = LLMTestCase(
        input=question,
        actual_output=answer_text,
        expected_output=expected_answer,
        retrieval_context=retrieved_texts,
    )

    faithfulness_metric = FaithfulnessMetric(model=judge, include_reason=False, async_mode=False)
    correctness_metric = GEval(
        name="Correctness",
        criteria=CORRECTNESS_CRITERIA,
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        model=judge,
        async_mode=False,
    )

    faithfulness_metric.measure(test_case)
    correctness_metric.measure(test_case)
    return JudgeScores(
        faithfulness=faithfulness_metric.score or 0.0, correctness=correctness_metric.score or 0.0
    )
