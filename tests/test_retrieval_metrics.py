"""Reproduce the retrieval metrics from Part 4 of the article.

These exact numbers are quoted in the article body, so the test acts as a
canary against silent metric drift.
"""

from __future__ import annotations

import pytest

from rag_evals.evaluation.retrieval import (
    evaluate_runs,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)

GOLD = {
    "q1": {"d3"},
    "q2": {"d7", "d2"},
    "q3": {"d11"},
    "q4": {"d5"},
}

RUNS = {
    "q1": ["d8", "d3", "d1", "d4", "d2", "d9", "d6", "d10", "d12", "d13"],
    "q2": ["d2", "d6", "d4", "d7", "d1", "d3", "d8", "d11", "d5", "d9"],
    "q3": ["d11", "d2", "d3", "d4", "d1", "d6", "d7", "d8", "d10", "d12"],
    "q4": ["d1", "d2", "d3", "d6", "d8", "d9", "d10", "d12", "d13", "d14"],
}


def test_recall_at_5_matches_article() -> None:
    avg = sum(recall_at_k(RUNS[q], GOLD[q], 5) for q in GOLD) / len(GOLD)
    assert avg == pytest.approx(0.750, abs=1e-3)


def test_mrr_matches_article() -> None:
    avg = sum(reciprocal_rank(RUNS[q], GOLD[q]) for q in GOLD) / len(GOLD)
    assert avg == pytest.approx(0.625, abs=1e-3)


def test_ndcg_at_5_matches_article() -> None:
    avg = sum(ndcg_at_k(RUNS[q], GOLD[q], 5) for q in GOLD) / len(GOLD)
    assert avg == pytest.approx(0.627, abs=1e-3)


def test_evaluate_runs_returns_all_metrics() -> None:
    m = evaluate_runs(RUNS, GOLD, k=5)
    assert m.recall_at_k == pytest.approx(0.750, abs=1e-3)
    assert m.mrr == pytest.approx(0.625, abs=1e-3)
    assert m.ndcg_at_k == pytest.approx(0.627, abs=1e-3)
    assert m.k == 5
    assert m.n_queries == 4
