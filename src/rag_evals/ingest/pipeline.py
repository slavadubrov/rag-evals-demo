"""Ingestion pipeline: docs -> chunks -> dense + sparse vectors -> Qdrant."""

from __future__ import annotations

from collections.abc import Iterable

from tqdm import tqdm

from rag_evals.config import settings
from rag_evals.index.qdrant_store import QdrantStore
from rag_evals.ingest.chunking import chunk_documents
from rag_evals.types import Chunk, Document


def _embed_dense(texts: list[str], model_name: str | None = None) -> list[list[float]]:
    from fastembed import TextEmbedding

    model_name = model_name or settings.embedding_model
    model = TextEmbedding(model_name=model_name)
    return [list(v) for v in model.embed(texts)]


def dense_dim(model_name: str | None = None) -> int:
    """Resolve the dimension of the dense embedding model by probing it once."""
    from fastembed import TextEmbedding

    model_name = model_name or settings.embedding_model
    model = TextEmbedding(model_name=model_name)
    sample = next(iter(model.embed(["dim probe"])))
    return len(list(sample))


def _embed_sparse(texts: list[str]) -> list[tuple[list[int], list[float]]]:
    from fastembed import SparseTextEmbedding

    model = SparseTextEmbedding(model_name="Qdrant/bm25")
    out: list[tuple[list[int], list[float]]] = []
    for v in model.embed(texts):
        out.append((list(map(int, v.indices)), list(map(float, v.values))))
    return out


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
    store = store or QdrantStore(dense_dim=dense_dim(embedding_model))
    store.ensure_collection()

    chunks: list[Chunk] = chunk_documents(
        documents,
        strategy=chunk_strategy,
        target_tokens=target_tokens,
        overlap_tokens=overlap_tokens,
    )
    n = 0
    for i in tqdm(range(0, len(chunks), batch_size), desc="ingest"):
        batch = chunks[i : i + batch_size]
        texts = [c.text for c in batch]
        dense = _embed_dense(texts, model_name=embedding_model)
        sparse = _embed_sparse(texts)
        n += store.upsert(batch, dense, sparse, batch_size=batch_size)
    return n
