"""End-to-end benchmark: chunking sweep × embedding sweep × LLM sweep.

Outputs:
- ``report/benchmark.json`` — every numeric result.
- ``report/benchmark.md`` — human-readable summary tables.

The chunking + embedding sweeps re-index a small slice of scifact into a
fresh Qdrant collection per config, then run the retrieval suite. The LLM
sweep keeps the index fixed (default) and varies the generator across the
three configured providers, recording latency, faithfulness (heuristic +
LLM-judge), and a pairwise-judge head-to-head.

The benchmark is intentionally bounded so it finishes in a few minutes:
``--n-docs`` (default 800) caps the corpus slice for index sweeps, and
``--n-queries`` (default 30) caps the eval set. Override via CLI for a
heavier run.
"""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from rag_evals._mock_warning import (
    MOCK_BANNER_MD,
    MOCK_BANNER_TEXT,
    is_mock,
    reset_mock_warnings,
    warn_mock_eval,
)
from rag_evals.config import settings
from rag_evals.data import scifact
from rag_evals.evaluation import faithfulness as fa
from rag_evals.evaluation.llm_judge import pairwise
from rag_evals.evaluation.retrieval import evaluate_runs
from rag_evals.generation.llm import LLM
from rag_evals.generation.models import Model, required_env_var
from rag_evals.generation.rag import run_rag
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

LLM_MODELS: list[Model] = [Model.GPT_5_MINI, Model.CLAUDE_HAIKU_4_5, Model.GEMINI_2_5_FLASH]

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


@dataclass
class LLMBenchResult:
    model: str
    backend_mode: str
    n_queries: int
    avg_latency_ms: float
    p95_latency_ms: float
    faithfulness_heuristic: float
    faithfulness_llm: float | None
    n_failures: int
    sample_answer: str = ""
    # True when any LLM involved in producing this row's numbers (the
    # generator itself or the cross-family judge used for faithfulness)
    # ran in mock mode. Mock-derived numbers are not real evaluations.
    is_mock: bool = False


@dataclass
class PairwiseRow:
    judge: str
    a: str
    b: str
    a_wins: int
    b_wins: int
    ties: int
    is_mock: bool = False  # generator(s) or judge ran in mock mode

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
        hits = retriever(r["query"], limit=k * 3)
        runs[r["qid"]] = [h.doc_id for h in hits]
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
    bench_path = settings.qdrant_path.parent / "qdrant_bench" / collection
    bench_path.mkdir(parents=True, exist_ok=True)
    store = QdrantStore(
        collection=collection,
        dense_dim=dim,
        path=str(bench_path) if not settings.qdrant_url else None,
    )
    if store.client.collection_exists(collection):
        store.client.delete_collection(collection)
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
        retriever = HybridRetriever(
            DenseRetriever(store=store, model_name=embedding_model),
            SparseRetriever(store=store),
        )
        m = _eval_retrieval(rows, retriever)
        _close(store)
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
        retriever = HybridRetriever(
            DenseRetriever(store=store, model_name=emb["model"]),
            SparseRetriever(store=store),
        )
        m = _eval_retrieval(rows, retriever)
        _close(store)
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
            )
        )
    return out


def llm_sweep(
    rows: list[dict[str, Any]],
    *,
    store: QdrantStore | None = None,
) -> tuple[list[LLMBenchResult], list[PairwiseRow]]:
    """Compare each LLM as the answer generator. Live where keys exist, mock otherwise.

    Faithfulness is computed two ways, both using a cross-family judge for
    claim extraction so the answer is decomposed by a model that didn't write
    it: (1) heuristic — deterministic content-word overlap of each claim
    against the retrieved context; (2) LLM-judge — the cross-family judge
    also verifies each claim. Heuristic gives a deterministic baseline that
    the LLM judge can be compared against.
    """
    store = store or QdrantStore()
    retriever = HybridRetriever(DenseRetriever(store=store), SparseRetriever(store=store))

    # 1) Generate answers per model
    answers: dict[str, list[dict[str, Any]]] = {}
    out_rows: list[LLMBenchResult] = []
    mock_modes: dict[str, bool] = {}  # model.value -> generator-was-mock
    for model in LLM_MODELS:
        llm = LLM(model)
        if is_mock(llm):
            warn_mock_eval(f"benchmark.llm_sweep.generator[{model.value}]")
            rprint(
                f"[bold red]   ⚠ MOCK MODE[/bold red] for {model.value} — "
                "rows produced will be flagged [MOCK] in the report."
            )
        rprint(f"[cyan]llm[/cyan]       {model.value}  (mode={llm.mode})  -> {len(rows)} queries")
        mock_modes[model.value] = is_mock(llm)
        # Cross-family judge resolved once per generator. Used for claim
        # extraction in both faithfulness paths (so the heuristic is scored
        # against real atomic claims, not a raw sentence split) and as the
        # verifier in the LLM-judge path. Falls back to mock when no live
        # cross-family key is available, which makes claim extraction
        # deterministic-and-free via the sentence-split path.
        judge_llm: LLM | None = None
        if llm.mode == "live":
            for m in LLM_MODELS:
                if _same_family(m, model):
                    continue
                candidate = LLM(m)
                if candidate.mode == "live":
                    judge_llm = candidate
                    break
        extractor_llm = judge_llm or LLM(Model.MOCK)
        per_q: list[dict[str, Any]] = []
        latencies: list[float] = []
        failures = 0
        faith_h: list[float] = []
        faith_llm: list[float] = []
        for r in rows:
            try:
                rag = run_rag(r["qid"], r["query"], lambda q: retriever(q, limit=10), llm=llm)
            except Exception as exc:  # noqa: BLE001
                failures += 1
                rprint(f"[red]  failure[/red] qid={r['qid']}: {exc}")
                continue
            latencies.append(rag.latency_ms)
            ctx = "\n\n".join(h.text or "" for h in rag.context)
            f_h = fa.faithfulness(rag.answer, ctx, llm=extractor_llm, use_heuristic=True).score
            faith_h.append(f_h)
            if judge_llm is not None:
                f_l = fa.faithfulness(rag.answer, ctx, llm=judge_llm).score
                faith_llm.append(f_l)
            per_q.append({"qid": r["qid"], "query": r["query"], "answer": rag.answer})
        answers[model.value] = per_q
        # Row is mock-tainted if either the generator OR the cross-family
        # judge ran in mock mode (LLM-faithfulness column is None in that
        # case, but the heuristic column still depends on extractor mode).
        row_is_mock = is_mock(llm) or is_mock(extractor_llm)
        out_rows.append(
            LLMBenchResult(
                model=model.value,
                backend_mode=llm.mode,
                n_queries=len(per_q),
                avg_latency_ms=round(statistics.mean(latencies), 1) if latencies else 0.0,
                p95_latency_ms=round(_percentile(latencies, 95), 1) if latencies else 0.0,
                faithfulness_heuristic=round(statistics.mean(faith_h), 4) if faith_h else 0.0,
                faithfulness_llm=round(statistics.mean(faith_llm), 4) if faith_llm else None,
                n_failures=failures,
                sample_answer=(per_q[0]["answer"][:240] + "…") if per_q else "",
                is_mock=row_is_mock,
            )
        )

    # 2) Pairwise head-to-head: each pair, judged by the third model (cross-family).
    pairwise_rows: list[PairwiseRow] = []
    if any(LLM(m).mode == "live" for m in LLM_MODELS):
        for i, ma in enumerate(LLM_MODELS):
            for mb in LLM_MODELS[i + 1 :]:
                if not (answers.get(ma.value) and answers.get(mb.value)):
                    continue
                judge_model = _third_model(ma, mb)
                judge_llm = LLM(judge_model)
                if judge_llm.mode != "live":
                    continue
                rprint(
                    f"[magenta]pairwise[/magenta] A={ma.value}  B={mb.value}  judge={judge_model.value}"
                )
                a_wins = b_wins = ties = 0
                qid_to_a = {a["qid"]: a for a in answers[ma.value]}
                qid_to_b = {b["qid"]: b for b in answers[mb.value]}
                for qid, a in qid_to_a.items():
                    b = qid_to_b.get(qid)
                    if not b:
                        continue
                    res = pairwise(
                        a["query"],
                        a["answer"],
                        b["answer"],
                        criterion="faithfulness and helpfulness",
                        llm=judge_llm,
                    )
                    if res.winner == "A":
                        a_wins += 1
                    elif res.winner == "B":
                        b_wins += 1
                    else:
                        ties += 1
                # Pairwise is gated on a live judge already, but the
                # generators feeding A and B answers may have been mock —
                # propagate that into the row so the report flags it.
                pairwise_rows.append(
                    PairwiseRow(
                        judge=judge_model.value,
                        a=ma.value,
                        b=mb.value,
                        a_wins=a_wins,
                        b_wins=b_wins,
                        ties=ties,
                        is_mock=mock_modes.get(ma.value, False) or mock_modes.get(mb.value, False),
                    )
                )
    return out_rows, pairwise_rows


def _same_family(a: Model, b: Model) -> bool:
    return required_env_var(a) == required_env_var(b)


def _third_model(a: Model, b: Model) -> Model:
    for m in LLM_MODELS:
        if m is not a and m is not b:
            return m
    return LLM_MODELS[0]


def _close(store: QdrantStore) -> None:
    """Release the embedded Qdrant lock so the next variant can open its own."""
    try:
        store.client.close()
    except Exception:  # noqa: BLE001
        pass


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    i = int(round((p / 100) * (len(s) - 1)))
    return s[i]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _mock_tag(is_mock_row: bool) -> str:
    return "[MOCK] " if is_mock_row else ""


def render_markdown(report: BenchmarkReport) -> str:
    lines: list[str] = ["# RAG benchmark report", ""]
    lines.append(
        f"_n-docs={report.settings.get('n_docs')} · n-queries={report.settings.get('n_queries')}_"
    )
    lines.append("")
    if report.has_mock_data:
        lines.append(MOCK_BANNER_MD)
        lines.append("")

    if report.chunking_sweep:
        lines.append("## Chunking sweep (embedding fixed)")
        lines.append("")
        lines.append("| config | chunks | index s | Recall@10 | MRR | nDCG@10 | MAP |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for r in report.chunking_sweep:
            lines.append(
                f"| {r.config} | {r.n_chunks} | {r.index_secs:.1f} | "
                f"{r.recall_at_10:.3f} | {r.mrr:.3f} | {r.ndcg_at_10:.3f} | {r.map:.3f} |"
            )
        lines.append("")

    if report.embedding_sweep:
        lines.append("## Embedding sweep (chunking fixed)")
        lines.append("")
        lines.append("| embedding | chunks | index s | Recall@10 | MRR | nDCG@10 | MAP |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for r in report.embedding_sweep:
            lines.append(
                f"| {r.config} | {r.n_chunks} | {r.index_secs:.1f} | "
                f"{r.recall_at_10:.3f} | {r.mrr:.3f} | {r.ndcg_at_10:.3f} | {r.map:.3f} |"
            )
        lines.append("")

    if report.llm_sweep:
        lines.append("## LLM sweep (retriever fixed, generator varies)")
        lines.append("")
        if any(r.is_mock for r in report.llm_sweep):
            lines.append(
                "_Rows tagged `[MOCK]` were produced with a mock LLM — "
                "faithfulness, latency, and the sample answer for those rows "
                "are NOT real evaluations._"
            )
            lines.append("")
        lines.append(
            "| model | mode | n | avg latency ms | p95 latency ms | faith (heuristic) | faith (LLM judge) | failures |"
        )
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for r in report.llm_sweep:
            faith_llm = "—" if r.faithfulness_llm is None else f"{r.faithfulness_llm:.3f}"
            lines.append(
                f"| {_mock_tag(r.is_mock)}{r.model} | {r.backend_mode} | {r.n_queries} | "
                f"{r.avg_latency_ms:.0f} | {r.p95_latency_ms:.0f} | "
                f"{r.faithfulness_heuristic:.3f} | {faith_llm} | {r.n_failures} |"
            )
        lines.append("")

    if report.pairwise:
        lines.append("## Pairwise judge (cross-family judge)")
        lines.append("")
        if any(r.is_mock for r in report.pairwise):
            lines.append(
                "_Rows tagged `[MOCK]` had at least one mock generator — "
                "win counts reflect the deterministic mock stub, NOT real model preference._"
            )
            lines.append("")
        lines.append("| judge | A | B | A wins | B wins | ties | A win-rate |")
        lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: |")
        for r in report.pairwise:
            lines.append(
                f"| {r.judge} | {_mock_tag(r.is_mock)}{r.a} | {_mock_tag(r.is_mock)}{r.b} | "
                f"{r.a_wins} | {r.b_wins} | {r.ties} | "
                f"{r.winrate_a():.2f} |"
            )
        lines.append("")

    if report.llm_sweep:
        lines.append("## Sample answers")
        lines.append("")
        for r in report.llm_sweep:
            mock_note = " — **MOCK ANSWER, not real model output**" if r.is_mock else ""
            lines.append(f"### {_mock_tag(r.is_mock)}{r.model}{mock_note}")
            lines.append("")
            lines.append(f"> {r.sample_answer}")
            lines.append("")

    return "\n".join(lines)


def print_console(report: BenchmarkReport) -> None:
    if report.has_mock_data:
        bar = "!" * 78
        console.print(f"[bold red]{bar}[/bold red]")
        console.print(f"[bold red]!!! {MOCK_BANNER_TEXT}[/bold red]")
        console.print(
            "[bold red]!!! Rows tagged [MOCK] below were produced (in whole or "
            "in part) by a mock LLM.[/bold red]"
        )
        console.print(f"[bold red]{bar}[/bold red]\n")
    if report.chunking_sweep:
        t = Table(title="Chunking sweep")
        for c in ("config", "chunks", "Recall@10", "MRR", "nDCG@10"):
            t.add_column(c)
        for r in report.chunking_sweep:
            t.add_row(
                r.config,
                str(r.n_chunks),
                f"{r.recall_at_10:.3f}",
                f"{r.mrr:.3f}",
                f"{r.ndcg_at_10:.3f}",
            )
        console.print(t)
    if report.embedding_sweep:
        t = Table(title="Embedding sweep")
        for c in ("embedding", "chunks", "Recall@10", "MRR", "nDCG@10"):
            t.add_column(c)
        for r in report.embedding_sweep:
            t.add_row(
                r.config,
                str(r.n_chunks),
                f"{r.recall_at_10:.3f}",
                f"{r.mrr:.3f}",
                f"{r.ndcg_at_10:.3f}",
            )
        console.print(t)
    if report.llm_sweep:
        title = "LLM sweep"
        if any(r.is_mock for r in report.llm_sweep):
            title += "  (⚠ contains MOCK rows — flagged below)"
        t = Table(title=title)
        for c in ("model", "mode", "avg ms", "p95 ms", "faith(h)", "faith(llm)"):
            t.add_column(c)
        for r in report.llm_sweep:
            faith_llm = "—" if r.faithfulness_llm is None else f"{r.faithfulness_llm:.3f}"
            label = f"[bold red][MOCK][/bold red] {r.model}" if r.is_mock else r.model
            t.add_row(
                label,
                r.backend_mode,
                f"{r.avg_latency_ms:.0f}",
                f"{r.p95_latency_ms:.0f}",
                f"{r.faithfulness_heuristic:.3f}",
                faith_llm,
            )
        console.print(t)
    if report.pairwise:
        title = "Pairwise (cross-family judge)"
        if any(r.is_mock for r in report.pairwise):
            title += "  (⚠ contains MOCK rows — flagged below)"
        t = Table(title=title)
        for c in ("judge", "A", "B", "A wins", "B wins", "ties", "A wr"):
            t.add_column(c)
        for r in report.pairwise:
            a_label = f"[bold red][MOCK][/bold red] {r.a}" if r.is_mock else r.a
            b_label = f"[bold red][MOCK][/bold red] {r.b}" if r.is_mock else r.b
            t.add_row(
                r.judge,
                a_label,
                b_label,
                str(r.a_wins),
                str(r.b_wins),
                str(r.ties),
                f"{r.winrate_a():.2f}",
            )
        console.print(t)
    if report.has_mock_data:
        console.print(
            "\n[bold red]⚠ This benchmark contains mock data — see the warning "
            "banner above. Numbers in [MOCK]-tagged rows are not real model "
            "evaluations.[/bold red]"
        )


def serialise(report: BenchmarkReport) -> dict[str, Any]:
    return {
        "settings": report.settings,
        # Top-level mock-data flag + warning text. Downstream readers (the
        # benchmark notebook, CI dashboards) MUST surface this when true.
        "has_mock_data": report.has_mock_data,
        "mock_warning": MOCK_BANNER_TEXT if report.has_mock_data else None,
        "chunking_sweep": [asdict(r) for r in report.chunking_sweep],
        "embedding_sweep": [asdict(r) for r in report.embedding_sweep],
        "llm_sweep": [asdict(r) for r in report.llm_sweep],
        "pairwise": [{**asdict(r), "winrate_a": r.winrate_a()} for r in report.pairwise],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_existing(out_dir: Path) -> dict[str, Any] | None:
    """Reuse prior sweep results when re-running a single arm so we don't
    have to re-index just to refresh the LLM numbers (or vice versa).
    """
    path = out_dir / "benchmark.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _run(
    n_docs: int,
    n_queries: int,
    n_llm_queries: int,
    skip_chunking: bool,
    skip_embedding: bool,
    skip_llm: bool,
    out_dir: Path,
) -> BenchmarkReport:
    out_dir.mkdir(parents=True, exist_ok=True)
    prior = _load_existing(out_dir) if any((skip_chunking, skip_embedding, skip_llm)) else None
    report = BenchmarkReport(
        settings={
            "n_docs": n_docs,
            "n_queries": n_queries,
            "n_llm_queries": n_llm_queries,
            "default_embedding": settings.embedding_model,
            "default_collection": settings.qdrant_collection,
        }
    )

    docs = _take_docs(n_docs)
    rows = _select_queries(docs, n_queries)
    rprint(f"[bold]docs[/bold]={len(docs)}  [bold]queries[/bold]={len(rows)}")

    if not skip_chunking:
        report.chunking_sweep = chunking_sweep(docs, rows)
    elif prior:
        report.chunking_sweep = [IndexBenchResult(**r) for r in prior.get("chunking_sweep", [])]
    if not skip_embedding:
        report.embedding_sweep = embedding_sweep(docs, rows)
    elif prior:
        report.embedding_sweep = [IndexBenchResult(**r) for r in prior.get("embedding_sweep", [])]
    if not skip_llm:
        # LLM sweep uses the *default* collection, so we don't need to re-index.
        store = QdrantStore()
        store.ensure_collection()
        rprint(f"[cyan]llm sweep collection[/cyan] {store.collection!r} ({store.count()} points)")
        # LLM sweep uses the full retrieval golden set, capped by --n-llm-queries.
        all_path = settings.golden_dir / "retrieval.jsonl"
        with all_path.open() as f:
            llm_rows = [json.loads(line) for line in f][:n_llm_queries]
        report.llm_sweep, report.pairwise = llm_sweep(llm_rows, store=store)
        _close(store)
    elif prior:
        report.llm_sweep = [LLMBenchResult(**r) for r in prior.get("llm_sweep", [])]
        report.pairwise = [
            PairwiseRow(**{k: v for k, v in r.items() if k != "winrate_a"})
            for r in prior.get("pairwise", [])
        ]
    return report


@app.command()
def main(
    n_docs: int = typer.Option(800, help="Slice of scifact for index sweeps"),
    n_queries: int = typer.Option(30, help="Eval queries per chunking/embedding config"),
    n_llm_queries: int = typer.Option(15, help="Queries per LLM (live API calls)"),
    skip_chunking: bool = typer.Option(False),
    skip_embedding: bool = typer.Option(False),
    skip_llm: bool = typer.Option(False),
    out: Path = typer.Option(REPORT_DIR, help="Output directory"),  # noqa: B008
) -> None:
    # Re-arm the one-shot mock warnings so a fresh run gets a fresh banner.
    reset_mock_warnings()
    report = _run(n_docs, n_queries, n_llm_queries, skip_chunking, skip_embedding, skip_llm, out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(serialise(report), indent=2))
    (out / "benchmark.md").write_text(render_markdown(report))
    rprint(f"\n[green]wrote[/green] {out / 'benchmark.md'} and {out / 'benchmark.json'}")
    print_console(report)
    if report.has_mock_data:
        warn_mock_eval("benchmark.main.summary")


if __name__ == "__main__":
    app()
