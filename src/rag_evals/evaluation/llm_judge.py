"""LLM-as-judge with bias mitigation.

Two scorers:
- ``g_eval``: pointwise rubric scoring (1-5).
- ``pairwise``: A vs B preference. Order is randomised by default and the
  ``measure_position_bias`` helper runs both orders to quantify the bias
  the article warns about.

Self-preference bias is observable by running the same pair through
multiple judge models (different families) and reporting per-judge skew.
"""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from dataclasses import dataclass

from rag_evals._mock_warning import is_mock, warn_mock_eval
from rag_evals.generation.llm import LLM
from rag_evals.generation.models import Model

G_EVAL_PROMPT = """Score the answer below on the criterion '{criterion}', 1-5.
Respond with a single integer.

Question: {question}
Answer: {answer}"""

PAIRWISE_PROMPT = """You will be shown two answers to the same question.
Pick the better answer for the criterion '{criterion}'.
Respond with exactly one token: A or B.

Question: {question}

Answer A:
{a}

Answer B:
{b}"""


@dataclass
class PairwiseResult:
    winner: str  # 'A' | 'B' | 'TIE'
    raw: str


def g_eval(
    question: str, answer: str, *, criterion: str = "faithfulness", llm: LLM | None = None
) -> int:
    llm = llm or LLM()
    if is_mock(llm):
        warn_mock_eval("llm_judge.g_eval")
    raw = llm.ask(G_EVAL_PROMPT.format(criterion=criterion, question=question, answer=answer))
    m = re.search(r"[1-5]", raw)
    return int(m.group(0)) if m else 3


def pairwise(
    question: str,
    a: str,
    b: str,
    *,
    criterion: str = "faithfulness",
    llm: LLM | None = None,
) -> PairwiseResult:
    llm = llm or LLM()
    if is_mock(llm):
        warn_mock_eval("llm_judge.pairwise")
    raw = llm.ask(PAIRWISE_PROMPT.format(criterion=criterion, question=question, a=a, b=b))
    text = raw.strip().upper()
    if text.startswith("A"):
        return PairwiseResult(winner="A", raw=raw)
    if text.startswith("B"):
        return PairwiseResult(winner="B", raw=raw)
    return PairwiseResult(winner="TIE", raw=raw)


@dataclass
class BiasMeasurement:
    a_first_winrate: float
    b_first_winrate: float
    position_bias: float  # signed: positive => first-shown advantage
    n: int


def measure_position_bias(
    pairs: Sequence[tuple[str, str, str]],
    *,
    llm: LLM | None = None,
    criterion: str = "faithfulness",
) -> BiasMeasurement:
    """``pairs`` is [(question, a, b), ...]. We run each pair in both
    orderings and measure how often the *first-shown* answer wins.
    A pure quality judge with no position bias should show 50% in both.
    """
    llm = llm or LLM()
    if is_mock(llm):
        warn_mock_eval("llm_judge.measure_position_bias")
    a_first = b_first = 0
    n = 0
    for question, a, b in pairs:
        r1 = pairwise(question, a, b, criterion=criterion, llm=llm)
        r2 = pairwise(question, b, a, criterion=criterion, llm=llm)
        if r1.winner != "TIE":
            n += 1
            a_first += int(r1.winner == "A")
            b_first += int(r2.winner == "A")  # swapped: 'A' position is now b
    if not n:
        return BiasMeasurement(0.0, 0.0, 0.0, 0)
    a_rate = a_first / n
    b_rate = b_first / n
    return BiasMeasurement(
        a_first_winrate=a_rate,
        b_first_winrate=b_rate,
        position_bias=(a_rate + b_rate) / 2 - 0.5,
        n=n,
    )


def averaged_pairwise(
    question: str,
    a: str,
    b: str,
    *,
    criterion: str = "faithfulness",
    llm: LLM | None = None,
    rng: random.Random | None = None,
) -> PairwiseResult:
    """Mitigation from the article: run both orderings and aggregate."""
    llm = llm or LLM()
    if is_mock(llm):
        warn_mock_eval("llm_judge.averaged_pairwise")
    rng = rng or random.Random(0)
    if rng.random() < 0.5:
        first, second = a, b
        flip = False
    else:
        first, second = b, a
        flip = True
    r1 = pairwise(question, first, second, criterion=criterion, llm=llm)
    r2 = pairwise(question, second, first, criterion=criterion, llm=llm)
    score = 0
    for r, flipped in ((r1, flip), (r2, not flip)):
        if r.winner == "A":
            score += -1 if flipped else 1
        elif r.winner == "B":
            score += 1 if flipped else -1
    if score > 0:
        return PairwiseResult(winner="A", raw=f"{r1.raw}|{r2.raw}")
    if score < 0:
        return PairwiseResult(winner="B", raw=f"{r1.raw}|{r2.raw}")
    return PairwiseResult(winner="TIE", raw=f"{r1.raw}|{r2.raw}")


def cross_family_judges(generator: Model) -> list[Model]:
    """Pick judge models that aren't from the same family as ``generator``.

    Used by notebook 07 to demonstrate the 'never use a model to judge itself'
    rule.
    """
    if generator.value.startswith("gpt-"):
        return [Model.CLAUDE_HAIKU_4_5, Model.GEMINI_2_5_FLASH]
    if generator.value.startswith("claude-"):
        return [Model.GPT_5_MINI, Model.GEMINI_2_5_FLASH]
    if generator.value.startswith("gemini/"):
        return [Model.GPT_5_MINI, Model.CLAUDE_HAIKU_4_5]
    return [Model.GPT_5_MINI, Model.CLAUDE_HAIKU_4_5, Model.GEMINI_2_5_FLASH]
