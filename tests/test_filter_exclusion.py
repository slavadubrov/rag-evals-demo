"""Reproduce the filter false-exclusion worked example from Part 5.

Half the queries lose their gold doc to the filter; the rate is 50%.
The test pinpoints the silent failure the article warns about.
"""

from __future__ import annotations

import pytest

from rag_evals.evaluation.filter_exclusion import (
    predicate_precision_recall,
    rate_against_survivors,
)

DOCS = [
    {"id": "d1", "tenant": "acme", "locale": "en-US"},
    {"id": "d2", "tenant": "acme", "locale": "en-GB"},
    {"id": "d3", "tenant": "globex", "locale": "en-US"},
    {"id": "d4", "tenant": "acme", "locale": "en-US"},
    {"id": "d5", "tenant": "acme", "locale": "de-DE"},
]


def survivors_for(predicate: dict) -> set[str]:
    return {d["id"] for d in DOCS if all(d.get(k) == v for k, v in predicate.items())}


def test_50_percent_exclusion_rate() -> None:
    queries = [
        {"qid": "q1", "gold_doc_ids": ["d2"], "filter_predicate": {"locale": "en-US"}},
        {"qid": "q2", "gold_doc_ids": ["d4"], "filter_predicate": {"tenant": "acme"}},
        {"qid": "q3", "gold_doc_ids": ["d3"], "filter_predicate": {"tenant": "acme"}},
        {"qid": "q4", "gold_doc_ids": ["d5"], "filter_predicate": {"locale": "de-DE"}},
    ]
    result = rate_against_survivors(queries, survivors_for)
    assert result.rate == pytest.approx(0.50)
    assert result.n_queries == 4
    assert result.n_excluded == 2
    excluded = {r.qid for r in result.rows if r.gold_excluded}
    assert excluded == {"q1", "q3"}


def test_predicate_precision_recall_perfect() -> None:
    pred = [{"tenant": "acme"}, {"locale": "en-US"}]
    gold = [{"tenant": "acme"}, {"locale": "en-US"}]
    m = predicate_precision_recall(pred, gold)
    assert m.precision == pytest.approx(1.0)
    assert m.recall == pytest.approx(1.0)
    assert m.f1 == pytest.approx(1.0)


def test_predicate_precision_recall_partial() -> None:
    pred = [{"tenant": "acme"}, {"locale": "en-US"}]
    gold = [{"tenant": "globex"}, {"locale": "en-US"}]
    m = predicate_precision_recall(pred, gold)
    assert m.precision == pytest.approx(0.5)
    assert m.recall == pytest.approx(0.5)
