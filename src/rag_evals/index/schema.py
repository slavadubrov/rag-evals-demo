"""Qdrant collection schema.

One collection, named vectors:
- "dense": cosine similarity over bge-small-en-v1.5 (384 dims)
- "sparse": Qdrant/bm25 sparse vectors

Payload carries doc_id, title, text and the synthesized metadata.
"""

from __future__ import annotations

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
DENSE_DIM = 384  # bge-small-en-v1.5

PAYLOAD_FIELDS = ("doc_id", "chunk_id", "title", "text", "tenant", "locale", "domain")
