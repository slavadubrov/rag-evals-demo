"""End-to-end RAG: query -> retrieve -> (rerank) -> generate."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable

from rag_evals._mock_warning import is_mock, warn_mock_eval
from rag_evals.evaluation.schemas import StrictModel
from rag_evals.generation.llm import LLM
from rag_evals.generation.prompts import RAG_SYSTEM, rag_user_prompt
from rag_evals.types import RAGAnswer, RetrievalHit

CITATION_GROUP_RE = re.compile(r"\[([^\[\]]+)\]")
CITATION_ID_RE = re.compile(r"[\w\-:]+")


def extract_citations(answer: str) -> list[str]:
    """Extract citation ids from forms like ``[d3]`` or ``[d3, d7]``."""
    seen: dict[str, None] = {}
    for group in CITATION_GROUP_RE.findall(answer):
        for token in CITATION_ID_RE.findall(group):
            seen.setdefault(token, None)
    return list(seen)


class GroundedAnswer(StrictModel):
    evidence_observations: list[str]
    abstain: bool
    answer: str
    cited_doc_ids: list[str]


def run_rag(
    qid: str,
    query: str,
    retrieve: Callable[[str], list[RetrievalHit]],
    *,
    rerank: Callable[[str, list[RetrievalHit]], list[RetrievalHit]] | None = None,
    llm: LLM | None = None,
    top_n_context: int = 5,
    sgr: bool = False,
) -> RAGAnswer:
    if top_n_context <= 0:
        raise ValueError("top_n_context must be positive")
    llm = llm or LLM()
    if is_mock(llm):
        warn_mock_eval("generation.run_rag")
    t0 = time.perf_counter()
    hits = retrieve(query)
    if rerank is not None:
        hits = rerank(query, hits)
    context = hits[:top_n_context]
    prompt = rag_user_prompt(query, context)
    if sgr:
        result = llm.structured(
            json.dumps(
                {
                    "question_and_context": prompt,
                    "task": "Record concise evidence observations, then answer with bracketed citations. Abstain on missing or conflicting evidence.",
                }
            ),
            GroundedAnswer,
            system=RAG_SYSTEM,
        )
        if not set(result.cited_doc_ids) <= {h.doc_id for h in context}:
            raise ValueError("SGR cited an unknown context ID")
        answer = "I don't know." if result.abstain else result.answer
        if not result.abstain and set(extract_citations(answer)) != set(result.cited_doc_ids):
            raise ValueError("SGR citation fields disagree with answer")
    else:
        answer = llm.ask(prompt, system=RAG_SYSTEM)
    return RAGAnswer(
        qid=qid,
        answer=answer,
        cited_doc_ids=extract_citations(answer),
        context=context,
        latency_ms=(time.perf_counter() - t0) * 1000,
    )
