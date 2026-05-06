"""Cross-encoder reranker (BAAI/bge-reranker-v2-m3 by default)."""

from __future__ import annotations

from rag_evals.config import settings
from rag_evals.types import RetrievalHit


class CrossEncoderReranker:
    def __init__(self, model_name: str | None = None) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name or settings.reranker_model
        self.model = CrossEncoder(self.model_name)

    def __call__(
        self,
        query: str,
        hits: list[RetrievalHit],
        *,
        limit: int | None = None,
    ) -> list[RetrievalHit]:
        if not hits:
            return []
        pairs = [(query, h.text or "") for h in hits]
        scores = self.model.predict(pairs).tolist()
        rescored = [
            RetrievalHit(
                doc_id=h.doc_id,
                score=float(s),
                chunk_id=h.chunk_id,
                text=h.text,
                metadata=h.metadata,
            )
            for h, s in zip(hits, scores, strict=True)
        ]
        rescored.sort(key=lambda h: h.score, reverse=True)
        return rescored[:limit] if limit is not None else rescored
