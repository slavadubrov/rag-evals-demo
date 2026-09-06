"""Atomic claim support, with explicit invalid/no-claim states and an offline proxy."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from rag_evals.evaluation.schemas import JUDGE_SYSTEM, Claims, Support
from rag_evals.generation.llm import LLM


def _split_sentences(text: str) -> list[str]:
    if re.fullmatch(r"\s*(I don['\u2019]t know|insufficient evidence)[.!]?\s*", text, re.I):
        return []
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p.strip()]


def extract_claims(answer: str, *, llm: LLM | None = None) -> list[str]:
    llm = llm or LLM()
    if llm.mode == "mock":
        return _split_sentences(answer)
    result = llm.structured(
        json.dumps(
            {
                "task": "Extract atomic factual claims. An abstention has no claims.",
                "answer": answer,
            }
        ),
        Claims,
        system=JUDGE_SYSTEM,
    )
    if any(not c.strip() for c in result.claims):
        raise ValueError("Empty atomic claim")
    return result.claims


@dataclass
class ClaimVerdict:
    claim: str
    supported: bool | None
    evidence_quote: str = ""
    explanation: str = ""
    status: str = "ok"


@dataclass
class FaithfulnessResult:
    score: float | None
    verdicts: list[ClaimVerdict]
    status: str = "ok"
    n_invalid: int = 0
    method: str = "sgr"
    errors: list[str] = field(default_factory=list)


def llm_verify(claims: list[str], context: str, *, llm: LLM | None = None) -> list[ClaimVerdict]:
    llm = llm or LLM()
    out = []
    for claim in claims:
        try:
            result = llm.structured(
                json.dumps(
                    {
                        "task": "Does context entail the claim? For SUPPORTED quote exact supporting text; otherwise quote may be empty.",
                        "context": context,
                        "claim": claim,
                    }
                ),
                Support,
                system=JUDGE_SYSTEM,
            )
            supported = result.verdict == "SUPPORTED"
            if supported and (not result.evidence_quote or result.evidence_quote not in context):
                raise ValueError("Supporting quote is absent from context")
            out.append(ClaimVerdict(claim, supported, result.evidence_quote, result.explanation))
        except Exception as exc:
            out.append(ClaimVerdict(claim, None, status="invalid", explanation=type(exc).__name__))
    return out


def heuristic_verify(claims: list[str], context: str) -> list[ClaimVerdict]:
    """Lexical diagnostic only: ignores negation, relations and semantic entailment."""
    ctx = set(re.findall(r"[a-z]+", context.lower()))
    out = []
    for claim in claims:
        words = {w for w in re.findall(r"[a-z]+", claim.lower()) if len(w) > 3}
        out.append(ClaimVerdict(claim, bool(words) and words <= ctx))
    return out


def faithfulness(
    answer: str, context: str, *, llm: LLM | None = None, use_heuristic: bool = False
) -> FaithfulnessResult:
    method = "lexical_proxy" if use_heuristic else "sgr"
    try:
        claims = _split_sentences(answer) if use_heuristic else extract_claims(answer, llm=llm)
    except Exception as exc:
        return FaithfulnessResult(None, [], "invalid", 1, method, [type(exc).__name__])
    if not claims:
        return FaithfulnessResult(None, [], "not_applicable", method=method)
    verdicts = (
        heuristic_verify(claims, context) if use_heuristic else llm_verify(claims, context, llm=llm)
    )
    invalid = sum(v.supported is None for v in verdicts)
    score = None if invalid else sum(v.supported is True for v in verdicts) / len(verdicts)
    return FaithfulnessResult(score, verdicts, "invalid" if invalid else "ok", invalid, method)
