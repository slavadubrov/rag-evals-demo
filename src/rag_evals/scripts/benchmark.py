"""End-to-end benchmark: chunking sweep x embedding sweep x LLM sweep.

Outputs:
- ``report/benchmark.json`` — every numeric result.
- ``report/benchmark.md`` — human-readable summary tables.

The chunking + embedding sweeps re-index a small slice of scifact into a
fresh Qdrant collection per config, then run the retrieval suite. The LLM
sweep keeps the index fixed (default) and varies the generator across the
configured OpenAI models, recording latency, faithfulness (heuristic +
LLM-judge), and a pairwise-judge head-to-head.

The benchmark is intentionally bounded so it finishes in a few minutes:
``--n-docs`` (default 800) caps the corpus slice for index sweeps, and
``--n-queries`` (default 30) caps the eval set. Override via CLI for a
heavier run.
"""

from __future__ import annotations

import itertools
import json
import statistics
import time
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import typer
from rich import print as rprint
from rich.console import Console

from rag_evals.config import settings
from rag_evals.data import scifact
from rag_evals.evaluation.generation import FIXTURES, calibrate, load_cases, run_generation
from rag_evals.evaluation.latency import quantiles
from rag_evals.evaluation.llm_judge import averaged_pairwise
from rag_evals.evaluation.provenance import provenance, write_report
from rag_evals.evaluation.retrieval import evaluate_runs
from rag_evals.generation.llm import LLM
from rag_evals.generation.models import Model
from rag_evals.index.qdrant_store import QdrantStore
from rag_evals.ingest.pipeline import dense_dim, ingest
from rag_evals.retrieval.dense import DenseRetriever
from rag_evals.retrieval.hybrid_rrf import HybridRetriever
from rag_evals.retrieval.sparse import SparseRetriever
from rag_evals.types import Document

app = typer.Typer(add_completion=False)
console = Console()

# ---------------------------------------------------------------------------
# Configurations to sweep
# ---------------------------------------------------------------------------

CHUNKING_CONFIGS: list[dict[str, Any]] = [
    {
        "name": "recursive_256_32",
        "strategy": "recursive",
        "target_tokens": 256,
        "overlap_tokens": 32,
    },
    {
        "name": "recursive_128_16",
        "strategy": "recursive",
        "target_tokens": 128,
        "overlap_tokens": 16,
    },
    {
        "name": "recursive_512_64",
        "strategy": "recursive",
        "target_tokens": 512,
        "overlap_tokens": 64,
    },
    {"name": "structural_256", "strategy": "structural", "target_tokens": 256, "overlap_tokens": 0},
]

EMBEDDING_CONFIGS: list[dict[str, Any]] = [
    {"name": "bge-small-en-v1.5", "model": "BAAI/bge-small-en-v1.5"},
    {"name": "all-MiniLM-L6-v2", "model": "sentence-transformers/all-MiniLM-L6-v2"},
    {"name": "bge-base-en-v1.5", "model": "BAAI/bge-base-en-v1.5"},
]

LLM_MODELS: list[Model] = [Model.GPT_5_6_LUNA, Model.GPT_5_6_TERRA]

REPORT_DIR = Path("report")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class IndexBenchResult:
    config: str
    chunking: dict[str, Any]
    embedding: str
    n_chunks: int
    index_secs: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float
    map: float
    coverage: float
    n_queries: int
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMBenchResult:
    model: str
    backend_mode: str
    n_queries: int
    avg_latency_ms: float | None
    p95_latency_ms: float | None
    faithfulness_heuristic: float | None
    faithfulness_llm: float | None
    n_failures: int
    sample_answer: str = ""
    # True when any LLM involved in producing this row's numbers (the
    # generator itself or the same-provider judge used for faithfulness)
    # ran in mock mode. Mock-derived numbers are not real evaluations.
    is_mock: bool = False
    n_attempted: int = 0
    n_invalid: int = 0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class PairwiseRow:
    judge: str
    a: str
    b: str
    a_wins: int
    b_wins: int
    ties: int
    is_mock: bool = False
    n_attempted: int = 0
    n_invalid: int = 0
    n_unpaired: int = 0
    rows: list[dict[str, Any]] = field(default_factory=list)

    def winrate_a(self) -> float:
        total = self.a_wins + self.b_wins + self.ties
        return self.a_wins / total if total else 0.0


@dataclass
class BenchmarkReport:
    chunking_sweep: list[IndexBenchResult] = field(default_factory=list)
    embedding_sweep: list[IndexBenchResult] = field(default_factory=list)
    llm_sweep: list[LLMBenchResult] = field(default_factory=list)
    pairwise: list[PairwiseRow] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)

    @property
    def has_mock_data(self) -> bool:
        return any(r.is_mock for r in self.llm_sweep) or any(r.is_mock for r in self.pairwise)


def _take_docs(n: int) -> list[Document]:
    out: list[Document] = []
    for d in scifact.documents():
        out.append(d)
        if len(out) >= n:
            break
    return out


def _doc_universe(docs: Iterable[Document]) -> set[str]:
    return {d.doc_id for d in docs}


def _select_queries(docs: Iterable[Document], n: int) -> list[dict[str, Any]]:
    """Pick golden retrieval rows whose gold doc_ids fall inside our slice."""
    universe = _doc_universe(docs)
    rows: list[dict[str, Any]] = []
    path = settings.golden_dir / "retrieval.jsonl"
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            if any(g in universe for g in row["gold_doc_ids"]):
                row["gold_doc_ids"] = [g for g in row["gold_doc_ids"] if g in universe]
                if row["gold_doc_ids"]:
                    rows.append(row)
            if len(rows) >= n:
                break
    return rows


def _eval_retrieval(rows: list[dict[str, Any]], retriever, *, k: int = 10) -> dict[str, Any]:
    runs: dict[str, list[str]] = {}
    gold: dict[str, list[str]] = {}
    for r in rows:
        try:
            hits = retriever(r["query"], limit=k * 3)
            runs[r["qid"]] = [h.doc_id for h in hits]
        except Exception:
            pass  # Missing runs remain in the metric denominator.
        gold[r["qid"]] = r["gold_doc_ids"]
    m = evaluate_runs(runs, gold, k=k)
    return asdict(m)


def _ingest_into(
    docs: list[Document],
    *,
    collection: str,
    chunk_strategy: str,
    target_tokens: int,
    overlap_tokens: int,
    embedding_model: str,
) -> tuple[QdrantStore, int, float]:
    dim = dense_dim(embedding_model)
    # Embedded Qdrant locks one storage folder per process, so each
    # benchmark variant gets its own subdirectory.
    collection = f"{collection}_{uuid.uuid4().hex[:8]}"
    bench_path = settings.qdrant_path.parent / "qdrant_bench" / collection
    bench_path.mkdir(parents=True, exist_ok=True)
    store = QdrantStore(
        collection=collection,
        dense_dim=dim,
        path=str(bench_path) if not settings.qdrant_url else None,
    )
    try:
        store.ensure_collection()
        t0 = time.perf_counter()
        n_chunks = ingest(
            docs,
            store=store,
            chunk_strategy=chunk_strategy,
            target_tokens=target_tokens,
            overlap_tokens=overlap_tokens,
            embedding_model=embedding_model,
        )
        return store, n_chunks, time.perf_counter() - t0
    except Exception:
        store.close()
        raise


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------


def chunking_sweep(
    docs: list[Document],
    rows: list[dict[str, Any]],
    *,
    embedding_model: str = "BAAI/bge-small-en-v1.5",
) -> list[IndexBenchResult]:
    """Hold embedding fixed, vary chunking strategy + window size."""
    out: list[IndexBenchResult] = []
    for cfg in CHUNKING_CONFIGS:
        rprint(f"[cyan]chunking[/cyan]  {cfg['name']}  -> indexing {len(docs)} docs")
        store, n_chunks, secs = _ingest_into(
            docs,
            collection=f"bench_chunk_{cfg['name']}",
            chunk_strategy=cfg["strategy"],
            target_tokens=cfg["target_tokens"],
            overlap_tokens=cfg["overlap_tokens"],
            embedding_model=embedding_model,
        )
        with store:
            retriever = HybridRetriever(
                DenseRetriever(store=store, model_name=embedding_model),
                SparseRetriever(store=store),
            )
            m = _eval_retrieval(rows, retriever)
        out.append(
            IndexBenchResult(
                config=cfg["name"],
                chunking=cfg,
                embedding=embedding_model,
                n_chunks=n_chunks,
                index_secs=round(secs, 2),
                recall_at_10=round(m["recall_at_k"], 4),
                mrr=round(m["mrr"], 4),
                ndcg_at_10=round(m["ndcg_at_k"], 4),
                map=round(m["map"], 4),
                coverage=round(m["coverage"], 4),
                n_queries=m["n_queries"],
                metrics=m,
            )
        )
    return out


def embedding_sweep(
    docs: list[Document],
    rows: list[dict[str, Any]],
    *,
    chunk_cfg: dict[str, Any] | None = None,
) -> list[IndexBenchResult]:
    """Hold chunking fixed, vary embedding model (different dimensions)."""
    cfg = chunk_cfg or CHUNKING_CONFIGS[0]
    out: list[IndexBenchResult] = []
    for emb in EMBEDDING_CONFIGS:
        rprint(f"[cyan]embedding[/cyan] {emb['name']}  -> indexing {len(docs)} docs")
        store, n_chunks, secs = _ingest_into(
            docs,
            collection=f"bench_emb_{emb['name'].replace('/', '_').replace('.', '_')}",
            chunk_strategy=cfg["strategy"],
            target_tokens=cfg["target_tokens"],
            overlap_tokens=cfg["overlap_tokens"],
            embedding_model=emb["model"],
        )
        with store:
            retriever = HybridRetriever(
                DenseRetriever(store=store, model_name=emb["model"]),
                SparseRetriever(store=store),
            )
            m = _eval_retrieval(rows, retriever)
        out.append(
            IndexBenchResult(
                config=emb["name"],
                chunking=cfg,
                embedding=emb["model"],
                n_chunks=n_chunks,
                index_secs=round(secs, 2),
                recall_at_10=round(m["recall_at_k"], 4),
                mrr=round(m["mrr"], 4),
                ndcg_at_10=round(m["ndcg_at_k"], 4),
                map=round(m["map"], 4),
                coverage=round(m["coverage"], 4),
                n_queries=m["n_queries"],
                metrics=m,
            )
        )
    return out


def llm_sweep(rows: list[dict[str, Any]], *, store: QdrantStore | None = None):
    """Fixed-context heldout comparison; every model sees the same labeled cases."""
    results, comparisons = [], []
    cases = load_cases(FIXTURES / "generation.jsonl", "heldout")[: len(rows)]
    outputs = {}
    for model in LLM_MODELS:
        generator = LLM(model, mode="live")
        judge = LLM(settings.rag_evals_judge_model, mode="live")
        result = run_generation(generator=generator, judge=judge, limit=len(cases))
        outputs[str(model)] = result
        durations = [r["latency_ms"] for r in result["rows"] if r.get("latency_ms") is not None]
        results.append(
            LLMBenchResult(
                model=str(model),
                backend_mode="live",
                n_queries=result["n_scored"],
                avg_latency_ms=statistics.fmean(durations) if durations else None,
                p95_latency_ms=quantiles(durations).p95 if durations else None,
                faithfulness_heuristic=None,
                faithfulness_llm=result["metrics"]["faithfulness"]["mean"],
                n_failures=result["n_failures"],
                n_attempted=result["n_attempted"],
                n_invalid=sum(m.get("n_invalid", 0) for m in result["metrics"].values()),
                details=result,
            )
        )
    for ma, mb in itertools.combinations(LLM_MODELS, 2):
        judge = LLM(settings.rag_evals_third_judge, mode="live")
        comparison = PairwiseRow(str(judge.model), str(ma), str(mb), 0, 0, 0)
        aa = {r["qid"]: r for r in outputs[str(ma)]["rows"] if r["status"] == "ok"}
        bb = {r["qid"]: r for r in outputs[str(mb)]["rows"] if r["status"] == "ok"}
        comparison.n_unpaired = len(cases) - len(aa.keys() & bb.keys())
        for case in cases:
            qid = case["qid"]
            if qid not in aa or qid not in bb:
                continue
            context = "\n\n".join(f"[{h['doc_id']}] {h['text']}" for h in case["context"])
            verdict = averaged_pairwise(
                case["query"],
                aa[qid]["answer"],
                bb[qid]["answer"],
                llm=judge,
                context_a=context,
                context_b=context,
            )
            comparison.n_attempted += 1
            comparison.n_invalid += verdict.status != "ok"
            comparison.a_wins += verdict.winner == "A"
            comparison.b_wins += verdict.winner == "B"
            comparison.ties += verdict.winner == "TIE"
            comparison.rows.append({"qid": qid, **asdict(verdict)})
        comparisons.append(comparison)
    return results, comparisons


def serialise(result: BenchmarkReport) -> dict[str, Any]:
    return {
        **asdict(result),
        "has_mock_data": result.has_mock_data,
        "pairwise": [{**asdict(r), "winrate_a": r.winrate_a()} for r in result.pairwise],
    }


def render_markdown(result: BenchmarkReport) -> str:
    lines = [
        "# RAG benchmark",
        "",
        "Fresh run; skipped arms are absent. No results are merged from older runs.",
        "LLM means are conditional on valid results. Pairwise comparisons use paired successful cases.",
        "All judges use OpenAI; this is not a cross-provider or self-preference study.",
        "",
    ]
    for title, rows in (("Chunking", result.chunking_sweep), ("Embedding", result.embedding_sweep)):
        lines += [
            f"## {title}",
            "",
            "| config | n | Recall@10 | nDCG@10 | MRR |",
            "| --- | --- | --- | --- | --- |",
        ]
        lines += [
            f"| {r.config} | {r.n_queries} | {r.recall_at_10:.3f} | {r.ndcg_at_10:.3f} | {r.mrr:.3f} |"
            for r in rows
        ]
    lines += [
        "",
        "## Generation",
        "",
        "| model | attempts | failures | invalid judgments | faithfulness |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {r.model} | {r.n_attempted} | {r.n_failures} | {r.n_invalid} | {r.faithfulness_llm} |"
        for r in result.llm_sweep
    ]
    lines += [
        "",
        "## Mirrored pairwise",
        "",
        "| A | B | A wins | B wins | ties | invalid | unpaired |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {r.a} | {r.b} | {r.a_wins} | {r.b_wins} | {r.ties} | {r.n_invalid} | {r.n_unpaired} |"
        for r in result.pairwise
    ]
    return "\n".join(lines)


@app.command()
def main(
    n_docs: int = typer.Option(800, min=1),
    n_queries: int = typer.Option(30, min=1),
    n_llm_queries: int = typer.Option(5, min=1),
    skip_chunking: bool = False,
    skip_embedding: bool = False,
    skip_llm: bool = False,
    live: bool = False,
    out: Path = typer.Option(REPORT_DIR),  # noqa: B008
) -> None:
    result = BenchmarkReport(
        settings={"n_docs": n_docs, "n_queries": n_queries, "n_llm_queries": n_llm_queries}
    )
    inputs = {}
    if not (skip_chunking and skip_embedding):
        docs = _take_docs(n_docs)
        rows = _select_queries(docs, n_queries)
        inputs = {"corpus": [asdict(d) for d in docs], "queries": rows}
        if not skip_chunking:
            result.chunking_sweep = chunking_sweep(docs, rows)
        if not skip_embedding:
            result.embedding_sweep = embedding_sweep(docs, rows)
    if not skip_llm and live:
        result.settings["calibration"] = calibrate(LLM(settings.rag_evals_judge_model, mode="live"))
        result.llm_sweep, result.pairwise = llm_sweep([{}] * n_llm_queries)
    result.settings["provenance"] = provenance(inputs)
    write_report(out / "benchmark.md", serialise(result), render_markdown(result))
    typer.echo(f"wrote {out / 'benchmark.md'}")
    if any(r.n_failures or r.n_invalid for r in result.llm_sweep) or any(
        r.n_invalid or r.n_unpaired for r in result.pairwise
    ):
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
