"""Evidence-grounded pointwise SGR rubrics and mirrored pairwise comparisons.

These are not the probability-weighted G-Eval algorithm. ``g_eval`` remains a
compatibility alias for the demo's old pointwise API.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass

from rag_evals.evaluation.schemas import JUDGE_SYSTEM, Preference, Rating
from rag_evals.generation.llm import LLM
from rag_evals.generation.models import Model


@dataclass
class RatingResult:
    score: int | None
    status: str
    explanation: str = ""
    evidence_observation: str = ""


def pointwise(
    question: str,
    answer: str,
    *,
    context: str = "",
    reference: str = "",
    criterion: str = "faithfulness",
    llm: LLM | None = None,
) -> RatingResult:
    llm = llm or LLM()
    if "faithfulness" in criterion.lower() and not context:
        return RatingResult(None, "invalid", "Faithfulness requires retrieved context")
    try:
        result = llm.structured(
            json.dumps(
                {
                    "task": "Score only the named criterion: 1 fails, 2 mostly fails, 3 mixed, 4 mostly meets, 5 fully meets.",
                    "criterion": criterion,
                    "question": question,
                    "answer": answer,
                    "context": context,
                    "reference": reference,
                }
            ),
            Rating,
            system=JUDGE_SYSTEM,
        )
        return RatingResult(result.score, "ok", result.explanation, result.evidence_observation)
    except Exception as exc:
        return RatingResult(None, "invalid", type(exc).__name__)


def g_eval(
    question: str,
    answer: str,
    *,
    criterion: str = "faithfulness",
    llm: LLM | None = None,
    context: str = "",
    reference: str = "",
) -> int | None:
    """Compatibility alias; use pointwise() to retain invalid status and evidence."""
    return pointwise(
        question, answer, criterion=criterion, llm=llm, context=context, reference=reference
    ).score


@dataclass
class PairwiseResult:
    winner: str | None
    raw: str
    status: str = "ok"


def pairwise(
    question: str,
    a: str,
    b: str,
    *,
    criterion: str = "faithfulness",
    llm: LLM | None = None,
    context_a: str = "",
    context_b: str = "",
) -> PairwiseResult:
    llm = llm or LLM()
    if "faithfulness" in criterion.lower() and not (context_a and context_b):
        return PairwiseResult(None, "Faithfulness requires evidence for both answers", "invalid")
    try:
        result = llm.structured(
            json.dumps(
                {
                    "task": "Compare candidates on the criterion. TIE is allowed when quality is equal.",
                    "criterion": criterion,
                    "question": question,
                    "A": {"answer": a, "context": context_a},
                    "B": {"answer": b, "context": context_b},
                }
            ),
            Preference,
            system=JUDGE_SYSTEM,
        )
        return PairwiseResult(result.winner, result.model_dump_json())
    except Exception as exc:
        return PairwiseResult(None, type(exc).__name__, "invalid")


@dataclass
class BiasMeasurement:
    a_first_winrate: float | None
    b_first_winrate: float | None
    position_bias: float | None
    n: int
    swap_consistency: float | None = None
    n_invalid: int = 0
    n_ties: int = 0


def measure_position_bias(
    pairs: Sequence[tuple[str, str, str]],
    *,
    llm: LLM | None = None,
    criterion: str = "helpfulness",
    context: str = "",
) -> BiasMeasurement:
    """Map swapped labels to identity. Equal candidate win rates are not required."""
    llm = llm or LLM()
    a_first = b_first = consistent = n = invalid = ties = 0
    invert = {"A": "B", "B": "A", "TIE": "TIE"}
    for question, a, b in pairs:
        r1 = pairwise(
            question, a, b, criterion=criterion, llm=llm, context_a=context, context_b=context
        )
        r2 = pairwise(
            question, b, a, criterion=criterion, llm=llm, context_a=context, context_b=context
        )
        if r1.winner is None or r2.winner is None:
            invalid += 1
            continue
        n += 1
        consistent += r1.winner == invert[r2.winner]
        ties += (r1.winner == "TIE") + (r2.winner == "TIE")
        a_first += r1.winner == "A"
        b_first += r2.winner == "A"
    decisive = 2 * n - ties
    return BiasMeasurement(
        a_first / n if n else None,
        b_first / n if n else None,
        (a_first + b_first) / decisive - 0.5 if decisive else None,
        n,
        consistent / n if n else None,
        invalid,
        ties,
    )


def averaged_pairwise(
    question: str,
    a: str,
    b: str,
    *,
    criterion: str = "faithfulness",
    llm: LLM | None = None,
    rng: random.Random | None = None,
    context_a: str = "",
    context_b: str = "",
) -> PairwiseResult:
    """Score both orders; invalid calls remain invalid, disagreement is a tie."""
    llm = llm or LLM()
    r1 = pairwise(
        question, a, b, criterion=criterion, llm=llm, context_a=context_a, context_b=context_b
    )
    r2 = pairwise(
        question, b, a, criterion=criterion, llm=llm, context_a=context_b, context_b=context_a
    )
    raw = f"{r1.raw}|{r2.raw}"
    if r1.winner is None or r2.winner is None:
        return PairwiseResult(None, raw, "invalid")
    score = {"A": 1, "B": -1, "TIE": 0}[r1.winner] - {"A": 1, "B": -1, "TIE": 0}[r2.winner]
    return PairwiseResult("A" if score > 0 else "B" if score < 0 else "TIE", raw)


def alternate_judges(generator: Model | str) -> list[Model]:
    """Different OpenAI models, not independent provider families."""
    return [
        m
        for m in (Model.GPT_5_6_LUNA, Model.GPT_5_6_TERRA, Model.GPT_6_ASTRA)
        if m != str(generator).removeprefix("openai/")
    ]
