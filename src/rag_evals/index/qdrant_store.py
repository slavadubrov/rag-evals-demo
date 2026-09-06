"""Thin Qdrant client wrapper with named dense + sparse vectors.

Encapsulates collection bootstrap, upsert, and the two search modes (dense,
sparse). Hybrid is fused client-side in ``rag_evals.retrieval.hybrid_rrf``.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Iterable

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from rag_evals.config import settings
from rag_evals.index.schema import (
    DENSE_DIM,
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
)
from rag_evals.types import Chunk, RetrievalHit


def _stable_uuid(chunk_id: str) -> str:
    h = hashlib.sha1(chunk_id.encode()).digest()
    return str(uuid.UUID(bytes=h[:16]))


class QdrantStore:
    """Wraps a Qdrant collection with named dense + sparse vectors.

    Defaults to embedded mode against ``settings.qdrant_path`` (a local file).
    Pass ``url=...`` or set ``QDRANT_URL`` to use a server instead.
    """

    def __init__(
        self,
        url: str | None = None,
        collection: str | None = None,
        path: str | None = None,
        dense_dim: int = DENSE_DIM,
    ) -> None:
        if url and path:
            raise ValueError("Choose url or path, not both")
        self.url = url if url is not None else (None if path else settings.qdrant_url)
        self.collection = collection or settings.qdrant_collection
        self.dense_dim = dense_dim
        if self.url:
            self.path = None
            self.client = QdrantClient(url=self.url)
        else:
            self.path = path or str(settings.qdrant_path)
            self.client = QdrantClient(path=self.path)

    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            params = self.client.get_collection(self.collection).config.params
            vectors = params.vectors
            if not isinstance(vectors, dict) or (
                DENSE_VECTOR_NAME not in vectors
                or vectors[DENSE_VECTOR_NAME].size != self.dense_dim
                or vectors[DENSE_VECTOR_NAME].distance != qm.Distance.COSINE
            ):
                raise ValueError("Dense dimension mismatch; rebuild the index")
            sparse = params.sparse_vectors or {}
            if (
                SPARSE_VECTOR_NAME not in sparse
                or sparse[SPARSE_VECTOR_NAME].modifier != qm.Modifier.IDF
            ):
                raise ValueError("BM25 index needs IDF; rebuild the index")
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                DENSE_VECTOR_NAME: qm.VectorParams(
                    size=self.dense_dim, distance=qm.Distance.COSINE
                ),
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: qm.SparseVectorParams(modifier=qm.Modifier.IDF),
            },
        )

    def upsert(
        self,
        chunks: Iterable[Chunk],
        dense_vectors: Iterable[list[float]],
        sparse_vectors: Iterable[tuple[list[int], list[float]]],
        batch_size: int = 64,
    ) -> int:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        points: list[qm.PointStruct] = []
        n = 0
        for chunk, dense, (idx, vals) in zip(chunks, dense_vectors, sparse_vectors, strict=True):
            payload = {
                **chunk.metadata,
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
            }
            points.append(
                qm.PointStruct(
                    id=_stable_uuid(chunk.chunk_id),
                    vector={
                        DENSE_VECTOR_NAME: dense,
                        SPARSE_VECTOR_NAME: qm.SparseVector(indices=idx, values=vals),
                    },
                    payload=payload,
                )
            )
            if len(points) >= batch_size:
                self.client.upsert(self.collection, points=points)
                n += len(points)
                points = []
        if points:
            self.client.upsert(self.collection, points=points)
            n += len(points)
        return n

    def _build_filter(self, predicates: dict[str, object] | None) -> qm.Filter | None:
        if not predicates:
            return None
        if any(not isinstance(v, (str, int, bool)) for v in predicates.values()):
            raise ValueError("Only exact string, integer and boolean filters are supported")
        return qm.Filter(
            must=[
                qm.FieldCondition(key=k, match=qm.MatchValue(value=v))
                for k, v in predicates.items()
                if isinstance(v, (str, int, bool))
            ]
        )

    def search_dense(
        self,
        vector: list[float],
        *,
        limit: int = 10,
        predicates: dict[str, object] | None = None,
    ) -> list[RetrievalHit]:
        flt = self._build_filter(predicates)
        results = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            using=DENSE_VECTOR_NAME,
            limit=limit,
            query_filter=flt,
            with_payload=True,
        ).points
        return [self._to_hit(p) for p in results]

    def search_sparse(
        self,
        indices: list[int],
        values: list[float],
        *,
        limit: int = 10,
        predicates: dict[str, object] | None = None,
    ) -> list[RetrievalHit]:
        flt = self._build_filter(predicates)
        results = self.client.query_points(
            collection_name=self.collection,
            query=qm.SparseVector(indices=indices, values=values),
            using=SPARSE_VECTOR_NAME,
            limit=limit,
            query_filter=flt,
            with_payload=True,
        ).points
        return [self._to_hit(p) for p in results]

    def survivor_ids(self, predicates: dict[str, object]) -> set[str]:
        """All doc IDs that pass ``predicates`` — used by the filter
        false-exclusion harness. Scrolls without vector search.
        """
        flt = self._build_filter(predicates)
        out: set[str] = set()
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=flt,
                limit=512,
                offset=offset,
                with_payload=["doc_id"],
                with_vectors=False,
            )
            for p in points:
                if p.payload:
                    out.add(p.payload["doc_id"])
            if offset is None:
                return out

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> QdrantStore:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def count(self) -> int:
        return self.client.count(self.collection).count

    @staticmethod
    def _to_hit(point) -> RetrievalHit:
        payload = point.payload or {}
        return RetrievalHit(
            doc_id=payload.get("doc_id", ""),
            score=float(point.score),
            chunk_id=payload.get("chunk_id"),
            text=payload.get("text"),
            metadata={k: payload[k] for k in ("tenant", "locale", "domain") if k in payload},
        )
