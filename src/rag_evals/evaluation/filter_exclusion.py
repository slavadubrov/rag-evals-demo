"""Filter false-exclusion rate — the article's signature metric.

A hard metadata filter can drop effective recall to zero without changing
the standard retrieval metrics. The gold doc is excluded *before* ranking
starts, so Recall@k computed over the survivor set can look fine.

Two metric flavours live here:

1. ``filter_false_exclusion_rate`` — the metric from the article. Pass a
   list of queries with their (correct) filter predicates; we count how
   many had their gold doc removed pre-retrieval.

2. ``predicate_precision_recall`` — for systems where the predicate is
   produced by an LLM extractor, treat the extractor as a classifier and
   measure it against a labelled (query, correct predicate) set.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass


@dataclass
class FilterEvalRow:
    qid: str
    gold_excluded: bool
    reason: str  # 'gold-not-in-survivors' | 'gold-survives' | 'no-gold'


@dataclass
class FilterEvalResult:
    rate: float
    n_queries: int
    n_excluded: int
    rows: list[FilterEvalRow]


def filter_false_exclusion_rate(
    queries: Iterable[dict],
    survives: Callable[[dict, dict], bool],
) -> FilterEvalResult:
    """Compute the filter false-exclusion rate.

    Each query is a dict with ``gold_doc_ids`` (iterable of strings) and
    ``filter_predicate`` (the predicate that *was* applied — could be wrong).

    ``survives(doc_meta, predicate) -> bool`` — domain-supplied callable
    that says whether a given doc passes the predicate. For Qdrant-backed
    runs, use ``QdrantStore.survivor_ids`` to materialise the survivor set
    and pass a closure that just checks membership.

    Returns the rate plus per-row reasons so callers can debug which
    queries were silently broken.
    """
    rows: list[FilterEvalRow] = []
    for q in queries:
        if q.get("authorization_predicate") and "eligible_gold_doc_ids" not in q:
            raise ValueError("Authorization requires explicit eligible gold")
        gold = list(q.get("eligible_gold_doc_ids", q.get("gold_doc_ids")) or [])
        if not gold:
            rows.append(FilterEvalRow(qid=q["qid"], gold_excluded=False, reason="no-gold"))
            continue
        predicate = q.get("filter_predicate") or {}
        authorization = q.get("authorization_predicate") or {}
        if set(predicate) & set(authorization):
            raise ValueError("Search predicates must not override authorization")
        predicate = {**predicate, **authorization}
        # `survives` takes the gold doc payload-shaped dict; here we
        # interpret 'meta' as just the doc_id wrapped — the real Qdrant
        # path uses survivor_ids. See ``rate_against_survivors`` below.
        surviving = [d for d in gold if survives({"doc_id": d}, predicate)]
        excluded = not surviving
        rows.append(
            FilterEvalRow(
                qid=q["qid"],
                gold_excluded=excluded,
                reason="gold-not-in-survivors" if excluded else "gold-survives",
            )
        )
    n_with_gold = sum(1 for r in rows if r.reason != "no-gold")
    n_excluded = sum(1 for r in rows if r.gold_excluded)
    rate = n_excluded / n_with_gold if n_with_gold else 0.0
    return FilterEvalResult(rate=rate, n_queries=n_with_gold, n_excluded=n_excluded, rows=rows)


def rate_against_survivors(
    queries: Iterable[dict],
    survivors_for: Callable[[dict], set[str]],
) -> FilterEvalResult:
    """Like ``filter_false_exclusion_rate`` but expects a function that
    returns the *full survivor doc-id set* given a predicate dict.

    This is the production path: pass in
    ``lambda p: store.survivor_ids(p)`` for a real Qdrant collection.
    """
    rows: list[FilterEvalRow] = []
    cache: dict[str, set[str]] = {}
    for q in queries:
        if q.get("authorization_predicate") and "eligible_gold_doc_ids" not in q:
            raise ValueError("Authorization requires explicit eligible gold")
        gold = set(q.get("eligible_gold_doc_ids", q.get("gold_doc_ids")) or [])
        if not gold:
            rows.append(FilterEvalRow(qid=q["qid"], gold_excluded=False, reason="no-gold"))
            continue
        predicate = q.get("filter_predicate") or {}
        authorization = q.get("authorization_predicate") or {}
        if set(predicate) & set(authorization):
            raise ValueError("Search predicates must not override authorization")
        predicate = {**predicate, **authorization}
        key = json.dumps(predicate, sort_keys=True)
        if key not in cache:
            cache[key] = survivors_for(predicate)
        survivors = cache[key]
        excluded = not bool(gold & survivors)
        rows.append(
            FilterEvalRow(
                qid=q["qid"],
                gold_excluded=excluded,
                reason="gold-not-in-survivors" if excluded else "gold-survives",
            )
        )
    n_with_gold = sum(1 for r in rows if r.reason != "no-gold")
    n_excluded = sum(1 for r in rows if r.gold_excluded)
    rate = n_excluded / n_with_gold if n_with_gold else 0.0
    return FilterEvalResult(rate=rate, n_queries=n_with_gold, n_excluded=n_excluded, rows=rows)


@dataclass
class PredicateClassifierMetrics:
    precision: float
    recall: float
    f1: float
    n: int


def predicate_precision_recall(
    predicted: list[dict[str, str]],
    gold: list[dict[str, str]],
) -> PredicateClassifierMetrics:
    """Treat dynamic predicate extraction as a classification problem.

    Each predicate is a dict {field: value}; we count exact (field,value)
    matches as true positives. Field-only matches with wrong value count
    as false positives + false negatives.
    """
    if len(predicted) != len(gold):
        raise ValueError("predicted and gold must align")
    tp = fp = fn = 0
    for p, g in zip(predicted, gold, strict=True):
        p_pairs = set(p.items())
        g_pairs = set(g.items())
        tp += len(p_pairs & g_pairs)
        fp += len(p_pairs - g_pairs)
        fn += len(g_pairs - p_pairs)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return PredicateClassifierMetrics(precision=precision, recall=recall, f1=f1, n=len(predicted))
