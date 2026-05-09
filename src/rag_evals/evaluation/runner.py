"""Orchestrates a full eval suite end-to-end.

Reads the golden JSONL files, runs hybrid retrieval against Qdrant,
collects retrieval / filter-exclusion / latency metrics, applies threshold
gates, and writes ``report.md`` + ``report.json``. Exits non-zero on
regression so it can drop into CI as-is.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import typer

from rag_evals.config import settings
from rag_evals.evaluation import (
    filter_exclusion as fx,
)
from rag_evals.evaluation import (
    latency,
    report,
)
from rag_evals.evaluation import (
    retrieval as rmetrics,
)
from rag_evals.index.qdrant_store import QdrantStore
from rag_evals.retrieval.dense import DenseRetriever
from rag_evals.retrieval.hybrid_rrf import HybridRetriever
from rag_evals.retrieval.sparse import SparseRetriever

app = typer.Typer(add_completion=False)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def run_suite(
    *,
    suite: str = "all",
    limit: int | None = None,
    k: int = 10,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    store = QdrantStore()
    retrieve = HybridRetriever(DenseRetriever(store=store), SparseRetriever(store=store))

    if suite in ("all", "retrieval"):
        retrieval_path = settings.golden_dir / "retrieval.jsonl"
        rows = _load_jsonl(retrieval_path)
        if limit:
            rows = rows[:limit]
        runs: dict[str, list[str]] = {}
        gold: dict[str, list[str]] = {}
        tracers: list[latency.Tracer] = []
        for row in rows:
            qid = row["qid"]
            tr = latency.Tracer()
            with tr.stage("retrieve"):
                # Over-retrieve at the chunk level so that doc-level dedup in
                # `evaluate_runs` still yields ≥ k unique documents per query.
                hits = retrieve(row["query"], limit=k * 3)
            tracers.append(tr)
            runs[qid] = [h.doc_id for h in hits]
            gold[qid] = row["gold_doc_ids"]
        rm = rmetrics.evaluate_runs(runs, gold, k=k)
        out["retrieval"] = asdict(rm)
        out["latency"] = {
            stage: asdict(stats) for stage, stats in latency.summarise(tracers).items()
        }

    if suite in ("all", "filter"):
        path = settings.golden_dir / "filter_aware.jsonl"
        rows = _load_jsonl(path)
        if limit:
            rows = rows[:limit]
        result = fx.rate_against_survivors(rows, lambda p: store.survivor_ids(p))
        out["filter_exclusion"] = {
            "rate": result.rate,
            "n_queries": result.n_queries,
            "n_excluded": result.n_excluded,
        }

    out["gates"] = _check_gates(out)
    return out


def _check_gates(suite: dict[str, Any]) -> list[dict[str, Any]]:
    gates: list[dict[str, Any]] = []
    if "retrieval" in suite:
        gates.append(
            {
                "name": "Recall@10",
                "observed": suite["retrieval"]["recall_at_k"],
                "threshold": settings.threshold_recall_at_10,
                "pass": suite["retrieval"]["recall_at_k"] >= settings.threshold_recall_at_10,
            }
        )
        gates.append(
            {
                "name": "MRR",
                "observed": suite["retrieval"]["mrr"],
                "threshold": settings.threshold_mrr,
                "pass": suite["retrieval"]["mrr"] >= settings.threshold_mrr,
            }
        )
    if "filter_exclusion" in suite:
        gates.append(
            {
                "name": "Filter false-exclusion",
                "observed": suite["filter_exclusion"]["rate"],
                "threshold": settings.threshold_filter_false_exclusion,
                "pass": suite["filter_exclusion"]["rate"]
                <= settings.threshold_filter_false_exclusion,
            }
        )
    return gates


@app.command()
def main(
    suite: str = typer.Option("all", help="all|retrieval|filter"),
    limit: int = typer.Option(0, help="limit queries; 0 = no limit"),
    report_path: Path = typer.Option(Path("report.md"), "--report"),  # noqa: B008
    k: int = typer.Option(10),
) -> None:
    result = run_suite(suite=suite, limit=limit or None, k=k)
    report_path.write_text(report.render_markdown(result))
    json_path = report_path.with_suffix(".json")
    json_path.write_text(json.dumps(result, indent=2))
    typer.echo(f"wrote {report_path} and {json_path}")
    failures = [g for g in result.get("gates", []) if not g["pass"]]
    if failures:
        for f in failures:
            typer.echo(f"FAIL: {f['name']} = {f['observed']:.4f}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
