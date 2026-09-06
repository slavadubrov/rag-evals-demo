"""Ingestion pipeline: docs -> chunks -> dense + sparse vectors -> Qdrant."""

from __future__ import annotations

from collections.abc import Iterable

from tqdm import tqdm

from rag_evals.config import settings
from rag_evals.index.qdrant_store import QdrantStore
from rag_evals.ingest.chunking import chunk_documents
from rag_evals.types import Document


def dense_dim(model_name: str | None = None) -> int:
    """Resolve the dimension of the dense embedding model by probing it once."""
    from fastembed import TextEmbedding

    model_name = model_name or settings.embedding_model
    model = TextEmbedding(model_name=model_name)
    sample = next(iter(model.embed(["dim probe"])))
    return len(list(sample))


def ingest(
    documents: Iterable[Document],
    *,
    store: QdrantStore | None = None,
    chunk_strategy: str = "recursive",
    target_tokens: int = 256,
    overlap_tokens: int = 32,
    batch_size: int = 64,
    embedding_model: str | None = None,
) -> int:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    from fastembed import SparseTextEmbedding, TextEmbedding

    dense_model = TextEmbedding(model_name=embedding_model or settings.embedding_model)
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    chunks = chunk_documents(
        documents,
        strategy=chunk_strategy,
        target_tokens=target_tokens,
        overlap_tokens=overlap_tokens,
    )
    if not chunks:
        raise ValueError("Cannot index an empty corpus")
    owned = store is None
    store = store or QdrantStore(dense_dim=len(next(iter(dense_model.embed(["probe"])))))
    try:
        store.ensure_collection()
        if store.count():
            raise ValueError(
                "Ingest requires an empty collection; use a new collection/path to rebuild"
            )
        n = 0
        for i in tqdm(range(0, len(chunks), batch_size), desc="ingest"):
            batch = chunks[i : i + batch_size]
            texts = [c.text for c in batch]
            dense = [list(map(float, v)) for v in dense_model.passage_embed(texts)]
            sparse = [
                (list(map(int, v.indices)), list(map(float, v.values)))
                for v in sparse_model.passage_embed(texts)
            ]
            n += store.upsert(batch, dense, sparse, batch_size=batch_size)
        return n
    finally:
        if owned:
            store.close()
