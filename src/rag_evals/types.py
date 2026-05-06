from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    position: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Query:
    qid: str
    text: str
    gold_doc_ids: frozenset[str] = frozenset()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalHit:
    doc_id: str
    score: float
    chunk_id: str | None = None
    text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    qid: str
    hits: list[RetrievalHit]
    filter_predicates: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0


@dataclass
class RAGAnswer:
    qid: str
    answer: str
    cited_doc_ids: list[str]
    context: list[RetrievalHit]
    latency_ms: float = 0.0
