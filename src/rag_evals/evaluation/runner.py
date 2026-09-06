"""Explicit suites, complete denominators, and opt-in dataset-specific quality gates."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import typer

from rag_evals.config import settings
from rag_evals.evaluation import filter_exclusion as fx
from rag_evals.evaluation import latency, report
from rag_evals.evaluation import retrieval as rmetrics
from rag_evals.evaluation.generation import calibrate, load_cases, run_generation
from rag_evals.evaluation.provenance import provenance, write_report
from rag_evals.generation.llm import LLM
from rag_evals.index.qdrant_store import QdrantStore
from rag_evals.retrieval.dense import DenseRetriever
from rag_evals.retrieval.hybrid_rrf import HybridRetriever
from rag_evals.retrieval.sparse import SparseRetriever

app = typer.Typer(add_completion=False)
SUITES = {"all", "retrieval", "filter", "generation", "offline"}


def run_suite(
    *,
    suite: str = "all",
    limit: int | None = None,
    k: int = 10,
    live: bool = False,
    quality_gates: bool = False,
    generator_model: str | None = None,
    judge_model: str | None = None,
) -> dict[str, Any]:
    if suite not in SUITES:
        raise ValueError(f"Unknown suite {suite}; choose {sorted(SUITES)}")
    if k <= 0 or (limit is not None and limit <= 0):
        raise ValueError("k and limit must be positive")
    if live and suite == "offline":
        raise ValueError("offline suite cannot make live calls")
    out: dict[str, Any] = {"suite": suite, "quality_gates_enabled": quality_gates}
    inputs = {}
    if suite in {"all", "retrieval", "filter"}:
        with QdrantStore() as store:
            if not store.client.collection_exists(store.collection) or not store.count():
                raise ValueError("Missing index: run make index first")
            store.ensure_collection()
            if suite in {"all", "retrieval"}:
                rows = load_cases(settings.golden_dir / "retrieval.jsonl")[:limit]
                inputs["retrieval"] = rows
                retrieve = HybridRetriever(
                    DenseRetriever(store=store), SparseRetriever(store=store)
                )
                runs, gold, errors, tracers = {}, {}, [], []
                for row in rows:
                    qid = row["qid"]
                    gold[qid] = row["gold_doc_ids"]
                    tr = latency.Tracer()
                    try:
                        with tr.stage("retrieve"):
                            runs[qid] = [h.doc_id for h in retrieve(row["query"], limit=k * 3)]
                    except Exception as exc:
                        errors.append({"qid": qid, "error_type": type(exc).__name__})
                    tracers.append(tr)
                out["retrieval"] = asdict(rmetrics.evaluate_runs(runs, gold, k=k))
                out["retrieval"]["errors"] = errors
                out["latency"] = {s: asdict(v) for s, v in latency.summarise(tracers).items()}
            if suite in {"all", "filter"}:
                rows = load_cases(settings.golden_dir / "filter_aware.jsonl")[:limit]
                if not rows:
                    raise ValueError("Empty filter dataset")
                if any(
                    "eligible_gold_doc_ids" not in r or "authorization_predicate" not in r
                    for r in rows
                ):
                    raise ValueError("Stale filter gold: run make golden")
                inputs["filter"] = rows
                out["filter_exclusion"] = asdict(
                    fx.rate_against_survivors(rows, store.survivor_ids)
                )
                out["filter_exclusion"]["population"] = (
                    "synthetic non-security predicate corruption"
                )
                clean = [r for r in rows if not r.get("corrupted", False)]
                out["filter_exclusion"]["clean"] = asdict(
                    fx.rate_against_survivors(clean, store.survivor_ids)
                )
    if suite in {"all", "generation", "offline"}:
        generator = LLM(
            generator_model or settings.rag_evals_default_model, mode="live" if live else "mock"
        )
        judge = LLM(judge_model or settings.rag_evals_judge_model, mode="live" if live else "mock")
        if live:
            out["calibration"] = calibrate(judge)
        out["generation"] = run_generation(
            generator=generator, judge=judge, limit=limit, replay=not live
        )
    out["provenance"] = provenance(inputs)
    out["gates"] = _check_gates(out)
    return out


def _check_gates(suite: dict[str, Any]) -> list[dict[str, Any]]:
    gates = []

    def gate(name, observed, threshold, *, maximum=False):
        passed = observed is not None and (
            observed <= threshold if maximum else observed >= threshold
        )
        gates.append({"name": name, "observed": observed, "threshold": threshold, "pass": passed})

    if r := suite.get("retrieval"):
        gate("Missing retrieval outputs", r["n_missing"], 0, maximum=True)
        if suite.get("quality_gates_enabled"):
            if r["k"] == 10:
                gate("Recall@10", r["recall_at_k"], settings.threshold_recall_at_10)
            gate("MRR", r["mrr"], settings.threshold_mrr)
    if (f := suite.get("filter_exclusion")) and suite.get("quality_gates_enabled"):
        clean = f["clean"]
        gate(
            "Clean filter false-exclusion",
            clean["rate"] if clean["n_queries"] else None,
            settings.threshold_filter_false_exclusion,
            maximum=True,
        )
    if g := suite.get("generation"):
        gate("Generation failures", g["n_failures"], 0, maximum=True)
        if not g["replay"]:
            gate(
                "Invalid SGR judgments",
                sum(m.get("n_invalid", 0) for m in g["metrics"].values()),
                0,
                maximum=True,
            )
            if suite.get("quality_gates_enabled"):
                gate(
                    "Faithfulness",
                    g["metrics"]["faithfulness"]["mean"],
                    settings.threshold_faithfulness,
                )
    if c := suite.get("calibration"):
        gate("Invalid calibration judgments", c["n_invalid"], 0, maximum=True)
    return gates


@app.command()
def main(
    suite: str = typer.Option("all", help="all|retrieval|filter|generation|offline"),
    limit: int = typer.Option(0, min=0),
    report_path: Path = typer.Option(Path("report.md"), "--report"),  # noqa: B008
    k: int = typer.Option(10, min=1),
    live: bool = False,
    quality_gates: bool = False,
    generator_model: str | None = None,
    judge_model: str | None = None,
) -> None:
    result = run_suite(
        suite=suite,
        limit=limit or None,
        k=k,
        live=live,
        quality_gates=quality_gates,
        generator_model=generator_model,
        judge_model=judge_model,
    )
    write_report(report_path, result, report.render_markdown(result))
    typer.echo(f"wrote {report_path} and {report_path.with_suffix('.json')}")
    if any(not g["pass"] for g in result["gates"]):
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
