"""Build the three golden sets used by the eval suite.

- ``retrieval.jsonl``: {qid, query, gold_doc_ids}
- ``filter_aware.jsonl``: {qid, query, gold_doc_ids, filter_predicate} where
  ``filter_predicate`` is a serialized dict, e.g. {"tenant": "acme"}.
- ``generation.jsonl``: {qid, query, gold_doc_ids, evidence_excerpt}; excerpts are not reference answers.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from rag_evals.config import settings
from rag_evals.data import scifact
from rag_evals.data.metadata import synthesize


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def build_retrieval() -> Path:
    out = settings.golden_dir / "retrieval.jsonl"
    rows = [
        {"qid": q.qid, "query": q.text, "gold_doc_ids": sorted(q.gold_doc_ids)}
        for q in scifact.test_queries()
    ]
    _write_jsonl(out, rows)
    return out


def build_filter_aware(seed: int = 7) -> Path:
    """For each query, attach the metadata of one of its gold docs as the
    *correct* filter predicate. With probability ``corruption_rate`` we
    deliberately swap to a wrong value — that's what notebook 04 detects.
    """
    rng = random.Random(seed)
    out = settings.golden_dir / "filter_aware.jsonl"
    options = {
        "locale": ["en-US", "en-GB", "de-DE"],
        "domain": ["biomed", "clinical", "public-health"],
    }
    rows: list[dict] = []
    for q in scifact.test_queries():
        gold = sorted(q.gold_doc_ids)[0]
        meta = synthesize(gold)
        # Pick the predicate field (rotates) and apply truth or corruption.
        field = rng.choice(["locale", "domain"])
        if rng.random() < 0.3:  # 30% deliberately corrupted predicates
            wrong = [v for v in options[field] if v != meta[field]]
            value = rng.choice(wrong)
            corrupted = True
        else:
            value = meta[field]
            corrupted = False
        rows.append(
            {
                "qid": q.qid,
                "query": q.text,
                "gold_doc_ids": sorted(q.gold_doc_ids),
                "filter_predicate": {field: value},
                "corrupted": corrupted,
                "authorization_predicate": {"tenant": meta["tenant"]},
                "eligible_gold_doc_ids": [
                    d for d in sorted(q.gold_doc_ids) if synthesize(d)["tenant"] == meta["tenant"]
                ],
            }
        )
    _write_jsonl(out, rows)
    return out


def build_generation(limit: int = 50) -> Path:
    """SciFact evidence excerpts, NOT reviewed QA answers."""
    out = settings.golden_dir / "generation.jsonl"
    docs_by_id = {d.doc_id: d for d in scifact.documents()}
    rows: list[dict] = []
    for q in scifact.test_queries():
        if len(rows) >= limit:
            break
        gold_id = sorted(q.gold_doc_ids)[0]
        if gold_id not in docs_by_id:
            continue
        rows.append(
            {
                "qid": q.qid,
                "query": q.text,
                "gold_doc_ids": sorted(q.gold_doc_ids),
                "evidence_excerpt": docs_by_id[gold_id].text,
                "reference_status": "unreviewed_evidence_not_answer",
            }
        )
    _write_jsonl(out, rows)
    return out


def build_all() -> dict[str, Path]:
    return {
        "retrieval": build_retrieval(),
        "filter_aware": build_filter_aware(),
        "generation": build_generation(),
    }


if __name__ == "__main__":
    paths = build_all()
    for name, path in paths.items():
        n = sum(1 for _ in path.open())
        print(f"  {name}: {n:>4} rows -> {path}")
