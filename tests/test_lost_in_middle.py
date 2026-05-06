from __future__ import annotations

from rag_evals.evaluation.lost_in_middle import _arrange
from rag_evals.types import RetrievalHit


def _hit(d: str) -> RetrievalHit:
    return RetrievalHit(doc_id=d, score=0.0, text=d)


def test_arrange_first() -> None:
    g = _hit("gold")
    ds = [_hit("a"), _hit("b")]
    out = _arrange(g, ds, "first")
    assert [h.doc_id for h in out] == ["gold", "a", "b"]


def test_arrange_last() -> None:
    g = _hit("gold")
    ds = [_hit("a"), _hit("b")]
    out = _arrange(g, ds, "last")
    assert [h.doc_id for h in out] == ["a", "b", "gold"]


def test_arrange_middle() -> None:
    g = _hit("gold")
    ds = [_hit("a"), _hit("b")]
    out = _arrange(g, ds, "middle")
    assert out[1].doc_id == "gold"
