from __future__ import annotations

from rag_evals.generation.rag import extract_citations


def test_extract_citations_basic() -> None:
    answer = "Phobos and Deimos orbit Mars [d3]. Mars has two moons [d3, d7]."
    assert extract_citations(answer) == ["d3", "d7"]


def test_extract_citations_handles_compound_ids() -> None:
    answer = "See [doc-1::2] and [doc-3]."
    assert extract_citations(answer) == ["doc-1::2", "doc-3"]


def test_extract_citations_empty() -> None:
    assert extract_citations("no citations here") == []
