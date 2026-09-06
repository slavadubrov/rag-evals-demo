"""Safe run identity: input/source hashes and metric definitions, never secrets."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from datetime import UTC, datetime
from pathlib import Path

from rag_evals.config import ROOT, settings
from rag_evals.data.scifact import CORPUS_REVISION, QRELS_REVISION
from rag_evals.evaluation.schemas import PROMPT_VERSION

SUITE_VERSION = "2.0"


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def provenance(rows: object) -> dict:
    sources = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT / "src").rglob("*.py"))
    }
    return {
        "corpus_revision": CORPUS_REVISION,
        "qrels_revision": QRELS_REVISION,
        "modality": "text",
        "suite_version": SUITE_VERSION,
        "prompt_version": PROMPT_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_sha256": digest(rows),
        "source_sha256": digest(sources),
        "lock_sha256": hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest()
        if (ROOT / "uv.lock").exists()
        else None,
        "dependencies": {p: importlib.metadata.version(p) for p in ("openai", "qdrant-client")},
        "retrieval": {
            "embedding": settings.embedding_model,
            "collection": settings.qdrant_collection,
            "rrf_k": 60,
            "sparse": "Qdrant/bm25+IDF",
        },
        "metric_definitions": {
            "retrieval": "binary qrels; unique document ranks; missing runs score zero; MRR/MAP use returned depth",
            "coverage": "intersection of retrieved and gold document universes at returned depth",
            "faithfulness": "supported atomic claims / all claims; any invalid judgment makes score null; no claims is N/A",
            "citation_id_validity": "cited IDs present in context / cited IDs; not semantic citation support",
            "filter": "no eligible gold survives immutable authorization plus non-security search predicates",
            "confidence_intervals": "query bootstrap 95% percentile, seed=7, 1000 draws; small samples are descriptive",
        },
    }


def write_report(path: Path, result: dict, markdown: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for target, content in (
        (path.with_suffix(".json"), json.dumps(result, indent=2, allow_nan=False)),
        (path, markdown),
    ):
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(target)
