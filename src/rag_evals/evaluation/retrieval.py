"""Retrieval metrics: Recall@k, Precision@k, MRR, nDCG@k, MAP, Hit Rate@k, Coverage.

Reproduces the article's Part 4 code block; the harness in `tests/test_retrieval_metrics.py`
asserts the exact numbers (Recall@5=0.750, MRR=0.625, nDCG@5=0.627).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import log2
from statistics import mean


def _unique(ranked: Sequence[str]) -> list[str]:
    """Drop chunk-level duplicates so doc-level metrics aren't inflated by
    multiple chunks of the same document landing in the top-k.
    """
    seen: set[str] = set()
    out: list[str] = []
    for d in ranked:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def recall_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    hits = sum(1 for d in _unique(ranked)[:k] if d in gold_set)
    return hits / len(gold_set)


def precision_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    if k <= 0:
        return 0.0
    gold_set = set(gold)
    hits = sum(1 for d in _unique(ranked)[:k] if d in gold_set)
    return hits / k


def reciprocal_rank(ranked: Sequence[str], gold: Iterable[str]) -> float:
    gold_set = set(gold)
    for r, d in enumerate(_unique(ranked), start=1):
        if d in gold_set:
            return 1.0 / r
    return 0.0


def hit_rate_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    gold_set = set(gold)
    return 1.0 if any(d in gold_set for d in _unique(ranked)[:k]) else 0.0


def ndcg_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    """Binary relevance nDCG@k."""
    gold_set = set(gold)
    gains = [1.0 if d in gold_set else 0.0 for d in _unique(ranked)[:k]]
    dcg = sum(g / log2(i + 2) for i, g in enumerate(gains))
    n_gold_in_topk = min(k, len(gold_set))
    idcg = sum(1.0 / log2(i + 2) for i in range(n_gold_in_topk))
    return dcg / idcg if idcg else 0.0


def average_precision(ranked: Sequence[str], gold: Iterable[str]) -> float:
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    hits = 0
    score = 0.0
    for r, d in enumerate(_unique(ranked), start=1):
        if d in gold_set:
            hits += 1
            score += hits / r
    return score / len(gold_set)


@dataclass
class RetrievalMetrics:
    recall_at_k: float
    precision_at_k: float
    mrr: float
    hit_rate_at_k: float
    ndcg_at_k: float
    map: float
    coverage: float
    k: int
    n_queries: int


def evaluate_runs(
    runs: dict[str, Sequence[str]],
    gold: dict[str, Iterable[str]],
    *,
    k: int = 10,
) -> RetrievalMetrics:
    """Evaluate ranked retrieval results against a gold map.

    runs: qid -> ranked list of doc_ids (chunk-level duplicates are deduped
        inside each per-query metric so chunk granularity doesn't inflate
        Recall etc.).
    gold: qid -> iterable of relevant doc_ids
    """
    qids = [q for q in runs if gold.get(q)]
    if not qids:
        raise ValueError("No queries with gold")
    recalls = [recall_at_k(runs[q], gold[q], k) for q in qids]
    precs = [precision_at_k(runs[q], gold[q], k) for q in qids]
    rrs = [reciprocal_rank(runs[q], gold[q]) for q in qids]
    hrs = [hit_rate_at_k(runs[q], gold[q], k) for q in qids]
    ndcgs = [ndcg_at_k(runs[q], gold[q], k) for q in qids]
    aps = [average_precision(runs[q], gold[q]) for q in qids]
    retrieved_universe = {d for q in qids for d in runs[q]}
    gold_universe = {d for q in qids for d in gold[q]}
    coverage = (
        len(retrieved_universe & gold_universe) / len(gold_universe) if gold_universe else 0.0
    )
    return RetrievalMetrics(
        recall_at_k=mean(recalls),
        precision_at_k=mean(precs),
        mrr=mean(rrs),
        hit_rate_at_k=mean(hrs),
        ndcg_at_k=mean(ndcgs),
        map=mean(aps),
        coverage=coverage,
        k=k,
        n_queries=len(qids),
    )
