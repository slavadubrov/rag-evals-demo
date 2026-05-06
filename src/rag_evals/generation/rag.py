"""End-to-end RAG: query -> retrieve -> (rerank) -> generate."""

from __future__ import annotations

import re
import time
from collections.abc import Callable

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


def run_rag(
    qid: str,
    query: str,
    retrieve: Callable[[str], list[RetrievalHit]],
    *,
    rerank: Callable[[str, list[RetrievalHit]], list[RetrievalHit]] | None = None,
    llm: LLM | None = None,
    top_n_context: int = 5,
) -> RAGAnswer:
    llm = llm or LLM()
    t0 = time.perf_counter()
    hits = retrieve(query)
    if rerank is not None:
        hits = rerank(query, hits)
    context = hits[:top_n_context]
    prompt = rag_user_prompt(query, context)
    answer = llm.ask(prompt, system=RAG_SYSTEM)
    return RAGAnswer(
        qid=qid,
        answer=answer,
        cited_doc_ids=extract_citations(answer),
        context=context,
        latency_ms=(time.perf_counter() - t0) * 1000,
    )
