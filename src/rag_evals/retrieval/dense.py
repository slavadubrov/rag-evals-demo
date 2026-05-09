"""Dense retriever — embed query, top-k cosine search."""

from __future__ import annotations

from rag_evals.config import settings
from rag_evals.index.qdrant_store import QdrantStore
from rag_evals.types import RetrievalHit


class DenseRetriever:
    def __init__(self, store: QdrantStore | None = None, model_name: str | None = None) -> None:
        from fastembed import TextEmbedding

        self.store = store or QdrantStore()
        self.model_name = model_name or settings.embedding_model
        self.model = TextEmbedding(model_name=self.model_name)

    def __call__(
        self, query: str, *, limit: int = 10, predicates: dict[str, object] | None = None
    ) -> list[RetrievalHit]:
        vec = list(next(iter(self.model.embed([query]))))
        return self.store.search_dense(vec, limit=limit, predicates=predicates)
