"""Sparse / BM25 retriever via Qdrant sparse vectors (FastEmbed Qdrant/bm25)."""

from __future__ import annotations

from rag_evals.index.qdrant_store import QdrantStore
from rag_evals.types import RetrievalHit


class SparseRetriever:
    def __init__(self, store: QdrantStore | None = None) -> None:
        from fastembed import SparseTextEmbedding

        self.store = store or QdrantStore()
        self.model = SparseTextEmbedding(model_name="Qdrant/bm25")

    def __call__(
        self, query: str, *, limit: int = 10, predicates: dict[str, object] | None = None
    ) -> list[RetrievalHit]:
        vec = next(iter(self.model.query_embed(query)))
        idx = list(map(int, vec.indices))
        vals = list(map(float, vec.values))
        return self.store.search_sparse(idx, vals, limit=limit, predicates=predicates)
