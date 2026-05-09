"""Faithfulness — RAGAS-style decomposition into atomic claims, then
per-claim verification against the retrieved context.

Two backends:
- ``llm_verify``: uses the configured LLM (live or mock) to judge each claim.
- ``nli_verify``: uses a local cross-encoder NLI model. Cheap, deterministic.

The harness layout follows the article's code block: extract -> verify ->
aggregate. The aggregation is just (supported / total). When a claim is
unsupported we keep it in the output so the per-row breakdown is visible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag_evals._mock_warning import is_mock, warn_mock_eval
from rag_evals.generation.llm import LLM

CLAIM_EXTRACTION_PROMPT = """Decompose the answer below into a list of atomic factual claims.
Output one claim per line, no numbering, no extra text.

Answer:
{answer}"""

CLAIM_VERIFICATION_PROMPT = """Given the context below, decide whether the claim is supported.
Respond with exactly one word: SUPPORTED or NOT_SUPPORTED.

Context:
{context}

Claim:
{claim}"""


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def extract_claims(answer: str, *, llm: LLM | None = None) -> list[str]:
    """Decompose ``answer`` into atomic claims. Falls back to sentence split
    when ``llm`` is mock-mode and no fixture is found.
    """
    llm = llm or LLM()
    if is_mock(llm):
        warn_mock_eval("faithfulness.extract_claims")
    prompt = CLAIM_EXTRACTION_PROMPT.format(answer=answer)
    raw = llm.ask(prompt, system="You are an information extraction expert.")
    lines = [line.strip("-• \t") for line in raw.splitlines() if line.strip()]
    if not lines or lines[0].startswith("[mock:"):
        return _split_sentences(answer)
    return lines


@dataclass
class ClaimVerdict:
    claim: str
    supported: bool


@dataclass
class FaithfulnessResult:
    score: float
    verdicts: list[ClaimVerdict]


def llm_verify(claims: list[str], context: str, *, llm: LLM | None = None) -> list[ClaimVerdict]:
    llm = llm or LLM()
    if is_mock(llm):
        warn_mock_eval("faithfulness.llm_verify")
    out: list[ClaimVerdict] = []
    for claim in claims:
        verdict = llm.ask(
            CLAIM_VERIFICATION_PROMPT.format(context=context, claim=claim),
            system="You are a careful fact-checker.",
        )
        supported = "SUPPORTED" in verdict.upper() and "NOT_SUPPORTED" not in verdict.upper()
        out.append(ClaimVerdict(claim=claim, supported=supported))
    return out


def heuristic_verify(claims: list[str], context: str) -> list[ClaimVerdict]:
    """Deterministic fallback used by tests and offline notebooks: a claim
    is 'supported' if all of its content words (>3 chars) appear in context.
    """
    ctx = context.lower()
    out: list[ClaimVerdict] = []
    for claim in claims:
        words = [w.lower() for w in re.findall(r"[A-Za-z]+", claim) if len(w) > 3]
        ok = bool(words) and all(w in ctx for w in words)
        out.append(ClaimVerdict(claim=claim, supported=ok))
    return out


def faithfulness(
    answer: str,
    context: str,
    *,
    llm: LLM | None = None,
    use_heuristic: bool = False,
) -> FaithfulnessResult:
    claims = extract_claims(answer, llm=llm)
    verdicts = (
        heuristic_verify(claims, context) if use_heuristic else llm_verify(claims, context, llm=llm)
    )
    if not verdicts:
        return FaithfulnessResult(score=0.0, verdicts=[])
    score = sum(1 for v in verdicts if v.supported) / len(verdicts)
    return FaithfulnessResult(score=score, verdicts=verdicts)
