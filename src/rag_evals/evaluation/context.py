"""Semantic context utility, nugget coverage and citation support via SGR."""

from __future__ import annotations

import json

from rag_evals.evaluation.faithfulness import extract_claims, llm_verify
from rag_evals.evaluation.schemas import JUDGE_SYSTEM, NuggetCoverage
from rag_evals.generation.llm import LLM
from rag_evals.generation.rag import extract_citations


def coverage(answer: str, nuggets: list[list[str]], judge: LLM) -> dict:
    if not nuggets:
        return {"score": None, "status": "not_applicable"}
    try:
        result = judge.structured(
            json.dumps(
                {
                    "task": "List zero-based indices of nuggets fully conveyed by the answer. Each nested list contains alternative expressions of one nugget.",
                    "answer": answer,
                    "nuggets": nuggets,
                }
            ),
            NuggetCoverage,
            system=JUDGE_SYSTEM,
        )
        indices = result.covered_indices
        if len(set(indices)) != len(indices) or any(i < 0 or i >= len(nuggets) for i in indices):
            raise ValueError("Invalid nugget indices")
        return {"score": len(indices) / len(nuggets), "status": "ok", **result.model_dump()}
    except Exception as exc:
        return {"score": None, "status": "invalid", "error_type": type(exc).__name__}


def citation_support(answer: str, context: list[dict], judge: LLM) -> dict:
    """Support of atomic claims by the union of explicitly cited passages.

    This measures citation-set support, not sentence-to-citation attribution.
    """
    cited = set(extract_citations(answer))
    if not cited:
        return {"score": None, "status": "not_applicable"}
    texts = {h["doc_id"]: h["text"] for h in context}
    if not cited <= texts.keys():
        return {"score": 0.0, "status": "ok", "reason": "unknown citation"}
    try:
        claims = extract_claims(answer, llm=judge)
        if not claims:
            return {"score": None, "status": "not_applicable"}
        verdicts = llm_verify(claims, "\n\n".join(texts[d] for d in sorted(cited)), llm=judge)
        invalid = any(v.supported is None for v in verdicts)
        return {
            "score": None
            if invalid
            else sum(v.supported is True for v in verdicts) / len(verdicts),
            "status": "invalid" if invalid else "ok",
        }
    except Exception as exc:
        return {"score": None, "status": "invalid", "error_type": type(exc).__name__}
