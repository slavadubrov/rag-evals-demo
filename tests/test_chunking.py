from __future__ import annotations

from rag_evals.ingest.chunking import chunk_documents, recursive_split, structural_split
from rag_evals.types import Document


def test_short_text_is_single_chunk() -> None:
    text = "Hello world."
    chunks = recursive_split(text, target_tokens=512)
    assert chunks == [text]


def test_long_text_splits_into_multiple() -> None:
    paragraph = "Long sentence number {i} that talks about retrieval. "
    text = "\n\n".join(paragraph.format(i=i) for i in range(60))
    chunks = recursive_split(text, target_tokens=64, overlap_tokens=8)
    assert len(chunks) > 1
    # No chunk grossly exceeds the target.
    assert all(len(c) <= 64 * 4 * 2 for c in chunks)


def test_chunk_documents_carries_metadata() -> None:
    docs = [
        Document(doc_id="d1", title="t1", text="x" * 4000, metadata={"tenant": "acme"}),
    ]
    chunks = chunk_documents(docs, target_tokens=128, overlap_tokens=16)
    assert chunks
    assert all(c.metadata == {"tenant": "acme"} for c in chunks)
    assert all(c.doc_id == "d1" for c in chunks)


def test_structural_split_respects_headings() -> None:
    text = "# Section A\n\nA1.\n\n# Section B\n\nB1.\n"
    pieces = structural_split(text, target_tokens=512)
    assert len(pieces) == 2
    assert "Section A" in pieces[0]
    assert "Section B" in pieces[1]
