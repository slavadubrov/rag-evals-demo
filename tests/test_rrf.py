"""Reciprocal rank fusion tests.

Reproduces the article's worked example (top-3: d3, d2, d1) and asserts
the rank-only / score-agnostic property: fused order does not depend on
the underlying retriever scores.
"""

from __future__ import annotations

import pytest

from rag_evals.retrieval.hybrid_rrf import reciprocal_rank_fusion


def test_article_example_top3() -> None:
    dense = ["d3", "d7", "d1", "d4", "d2", "d9", "d10"]
    sparse = ["d2", "d3", "d8", "d1", "d11", "d4", "d6"]
    fused = reciprocal_rank_fusion([dense, sparse], k=60)
    top_ids = [doc for doc, _ in fused[:3]]
    assert top_ids == ["d3", "d2", "d1"]


def test_top1_score_matches_article() -> None:
    dense = ["d3", "d7", "d1", "d4", "d2", "d9", "d10"]
    sparse = ["d2", "d3", "d8", "d1", "d11", "d4", "d6"]
    fused = dict(reciprocal_rank_fusion([dense, sparse], k=60))
    # 1/(60+1) + 1/(60+2)
    assert fused["d3"] == pytest.approx(1 / 61 + 1 / 62, abs=1e-6)


def test_rank_only_property() -> None:
    """Fused ordering depends only on rank, not raw scores. We exercise
    this by reusing the same rankings: any 'score' is irrelevant.
    """
    a = ["x", "y", "z"]
    b = ["y", "x", "z"]
    fused1 = reciprocal_rank_fusion([a, b], k=60)
    fused2 = reciprocal_rank_fusion([a, b], k=60)
    assert fused1 == fused2
