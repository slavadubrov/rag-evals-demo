"""Fetch and cache BEIR scifact via Hugging Face datasets.

scifact has 5,183 docs and ~300 test queries with relevance judgments. We
expose three iterators: documents, queries, qrels. Cached to ``data/cache``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from rag_evals.config import settings
from rag_evals.data.metadata import synthesize
from rag_evals.types import Document, Query

CORPUS_REPO = "BeIR/scifact"
QUERIES_REPO = "BeIR/scifact"
QRELS_REPO = "BeIR/scifact-qrels"


def _cache_path(name: str) -> Path:
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    return settings.cache_dir / f"scifact-{name}.jsonl"


def _download() -> None:
    from datasets import load_dataset  # local import to avoid cold pyproject hit

    corpus = load_dataset(CORPUS_REPO, "corpus", split="corpus")
    queries = load_dataset(QUERIES_REPO, "queries", split="queries")
    qrels = load_dataset(QRELS_REPO, split="test")

    with _cache_path("corpus").open("w") as f:
        for row in corpus:
            f.write(json.dumps(row) + "\n")
    with _cache_path("queries").open("w") as f:
        for row in queries:
            f.write(json.dumps(row) + "\n")
    with _cache_path("qrels").open("w") as f:
        for row in qrels:
            f.write(json.dumps(row) + "\n")


def _ensure_cached() -> None:
    if not all(_cache_path(n).exists() for n in ("corpus", "queries", "qrels")):
        _download()


def documents() -> Iterator[Document]:
    _ensure_cached()
    with _cache_path("corpus").open() as f:
        for line in f:
            row = json.loads(line)
            doc_id = row["_id"]
            yield Document(
                doc_id=doc_id,
                title=row.get("title", ""),
                text=row.get("text", ""),
                metadata=synthesize(doc_id),
            )


def queries() -> Iterator[Query]:
    """All queries (train + test). Use qrels() to attach gold doc ids."""
    _ensure_cached()
    qrels_map = _qrels_map()
    with _cache_path("queries").open() as f:
        for line in f:
            row = json.loads(line)
            qid = row["_id"]
            yield Query(
                qid=qid,
                text=row["text"],
                gold_doc_ids=frozenset(qrels_map.get(qid, set())),
            )


def _qrels_map() -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    with _cache_path("qrels").open() as f:
        for line in f:
            row = json.loads(line)
            qid = str(row["query-id"])
            doc_id = str(row["corpus-id"])
            score = int(row.get("score", 1))
            if score > 0:
                out.setdefault(qid, set()).add(doc_id)
    return out


def test_queries() -> Iterator[Query]:
    """Queries that have at least one positive qrel — the eval set."""
    return (q for q in queries() if q.gold_doc_ids)
