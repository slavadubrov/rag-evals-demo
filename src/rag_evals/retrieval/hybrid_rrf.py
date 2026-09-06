"""Hybrid retrieval via Reciprocal Rank Fusion (Cormack et al., SIGIR 2009).

RRF is rank-only and score-agnostic. It avoids the score-normalisation
disasters of linear fusion (z-score, min-max) when one lane has higher
score variance than the other.

    score(d) = sum over rankings of 1 / (k + rank_in_ranking(d))
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from rag_evals.types import RetrievalHit

DEFAULT_K = 60  # Cormack et al.'s canonical default


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], *, k: int = DEFAULT_K
) -> list[tuple[str, float]]:
    """Fuse N rank lists by RRF. Returns (doc_id, score) sorted desc."""
    if k < 0:
        raise ValueError("RRF k must be nonnegative")
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc in enumerate(dict.fromkeys(ranking), start=1):
            scores[doc] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def fuse_hits(
    rankings: Sequence[Sequence[RetrievalHit]],
    *,
    k: int = DEFAULT_K,
    limit: int | None = None,
) -> list[RetrievalHit]:
    """Same as ``reciprocal_rank_fusion`` but operates on RetrievalHit
    sequences and returns RetrievalHit (with the fused RRF score)."""
    if k < 0 or (limit is not None and limit < 0):
        raise ValueError("k and limit must be nonnegative")
    rrf_scores: dict[str, float] = defaultdict(float)
    representative: dict[str, RetrievalHit] = {}
    for ranking in rankings:
        unique = {h.doc_id: h for h in reversed(ranking)}
        ordered = [unique[d] for d in dict.fromkeys(h.doc_id for h in ranking)]
        for rank, hit in enumerate(ordered, start=1):
            rrf_scores[hit.doc_id] += 1.0 / (k + rank)
            representative.setdefault(hit.doc_id, hit)
    fused = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)
    if limit is not None:
        fused = fused[:limit]
    return [
        RetrievalHit(
            doc_id=doc_id,
            score=score,
            chunk_id=representative[doc_id].chunk_id,
            text=representative[doc_id].text,
            metadata=representative[doc_id].metadata,
        )
        for doc_id, score in fused
    ]


class HybridRetriever:
    """Combine a dense and a sparse retriever via RRF."""

    def __init__(self, dense, sparse, *, k: int = DEFAULT_K) -> None:
        self.dense = dense
        self.sparse = sparse
        self.k = k

    def __call__(
        self,
        query: str,
        *,
        limit: int = 10,
        per_lane: int = 50,
        predicates: dict[str, object] | None = None,
    ) -> list[RetrievalHit]:
        dense_hits = self.dense(query, limit=per_lane, predicates=predicates)
        sparse_hits = self.sparse(query, limit=per_lane, predicates=predicates)
        return fuse_hits([dense_hits, sparse_hits], k=self.k, limit=limit)
