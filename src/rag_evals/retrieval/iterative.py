"""Bounded query expansion; authorization predicates are immutable across rounds."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from rag_evals.retrieval.hybrid_rrf import fuse_hits
from rag_evals.types import RetrievalHit


@dataclass
class IterativeResult:
    hits: list[RetrievalHit]
    calls: int
    elapsed_ms: float
    stop_reason: str


def retrieve_iteratively(
    queries: Sequence[str],
    retrieve: Callable,
    *,
    predicates: dict[str, object] | None = None,
    max_calls: int = 3,
    max_seconds: float = 10,
    max_query_chars: int = 4000,
    max_tokens: int = 4000,
    count_tokens: Callable[[str], int] = lambda text: len(text.encode("utf-8")),
    limit: int = 10,
) -> IterativeResult:
    """Run supplied reformulations within call/time/input budgets and fuse results.

    No LLM query planner is implied. UTF-8 bytes conservatively bound query tokens
    by default; supply the target tokenizer for exact query counts. Check time between calls; the backend must
    enforce its own request timeout to bound a single blocking retrieval call.
    """
    if min(max_calls, max_seconds, max_query_chars, max_tokens, limit) <= 0:
        raise ValueError("Budgets and limit must be positive")
    start = time.monotonic()
    rankings: list[list[RetrievalHit]] = []
    used = tokens = 0
    reason = "queries_exhausted"
    for query in dict.fromkeys(queries):
        if len(rankings) >= max_calls:
            reason = "call_budget"
            break
        if time.monotonic() - start >= max_seconds:
            reason = "time_budget"
            break
        if used + len(query) > max_query_chars:
            reason = "input_budget"
            break
        query_tokens = count_tokens(query)
        if query_tokens < 0:
            raise ValueError("Token count must be nonnegative")
        if tokens + query_tokens > max_tokens:
            reason = "token_budget"
            break
        tokens += query_tokens
        used += len(query)
        rankings.append(retrieve(query, limit=limit, predicates=dict(predicates or {})))
    return IterativeResult(
        fuse_hits(rankings, limit=limit), len(rankings), (time.monotonic() - start) * 1000, reason
    )
