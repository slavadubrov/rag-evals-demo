"""Generate the notebooks. Run once after editing — outputs go straight to
``notebooks/*.ipynb``. Each notebook imports from ``src/rag_evals/`` so
the metric logic lives in one place.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from textwrap import dedent

OUT = Path(__file__).resolve().parent


def _cell_id() -> str:
    return uuid.uuid4().hex[:12]


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": _cell_id(),
        "metadata": {},
        "source": [line + "\n" for line in dedent(text).strip().splitlines()],
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "id": _cell_id(),
        "metadata": {},
        "source": [line + "\n" for line in dedent(text).strip().splitlines()],
        "outputs": [],
        "execution_count": None,
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write(path: Path, cells: list[dict]) -> None:
    source = "".join("".join(c["source"]) for c in cells)
    if "store = QdrantStore()" in source:
        cells.append(code("store.close()"))
    path.write_text(json.dumps(notebook(cells), indent=1))
    print(f"  wrote {path}")


# -----------------------------------------------------------------------------
# 00 - Setup and index
# -----------------------------------------------------------------------------
nb00 = [
    md(
        """
        # 00 — Setup and index

        Spin Qdrant, ingest scifact, sanity-check the collection. Run this once before the rest of the tour.
        """
    ),
    code(
        """
        from rag_evals.config import settings
        from rag_evals.index.qdrant_store import QdrantStore

        store = QdrantStore()
        store.ensure_collection()
        print(f"collection={store.collection!r} url={store.url!r}")
        """
    ),
    md(
        """
        Run `make index` from the shell to ingest scifact and build golden sets — that step is heavy
        (downloads embeddings + reranker the first time), so we do it outside the notebook.

        Below: verify what landed in the collection.
        """
    ),
    code(
        """
        from rag_evals.data import scifact
        from rag_evals.data.metadata import synthesize

        n = sum(1 for _ in scifact.documents())
        print(f"scifact corpus size: {n}")

        # Show metadata distribution on a sample
        sample = list(scifact.documents())[:200]
        from collections import Counter
        for field in ("tenant", "locale", "domain"):
            counts = Counter(synthesize(d.doc_id)[field] for d in sample)
            print(f"  {field:>7}: {dict(counts)}")
        """
    ),
    code(
        """
        # Smoke retrieval if the index has been seeded
        try:
            count = store.count()
            print(f"qdrant collection has {count} points")
        except Exception as e:
            print(f"collection empty or unreachable: {e}")
        """
    ),
]

# -----------------------------------------------------------------------------
# 01 - Retrieval metrics
# -----------------------------------------------------------------------------
nb01 = [
    md(
        """
        # 01 — Retrieval metrics

        Reproduce the article's Recall@k / MRR / nDCG@k example, then run the same metrics on a small
        scifact slice with the dense retriever.
        """
    ),
    code(
        """
        from rag_evals.evaluation.retrieval import evaluate_runs

        gold = {
            "q1": {"d3"},
            "q2": {"d7", "d2"},
            "q3": {"d11"},
            "q4": {"d5"},
        }
        runs = {
            "q1": ["d8", "d3", "d1", "d4", "d2", "d9", "d6", "d10", "d12", "d13"],
            "q2": ["d2", "d6", "d4", "d7", "d1", "d3", "d8", "d11", "d5", "d9"],
            "q3": ["d11", "d2", "d3", "d4", "d1", "d6", "d7", "d8", "d10", "d12"],
            "q4": ["d1", "d2", "d3", "d6", "d8", "d9", "d10", "d12", "d13", "d14"],
        }
        m = evaluate_runs(runs, gold, k=5)
        print(f"Recall@5 = {m.recall_at_k:.3f}  (article: 0.750)")
        print(f"MRR      = {m.mrr:.3f}  (article: 0.625)")
        print(f"nDCG@5   = {m.ndcg_at_k:.3f}  (article: 0.627)")
        """
    ),
    md("Now apply the same metrics to live retrieval against scifact."),
    code(
        """
        import json
        from pathlib import Path
        from rag_evals.config import settings

        retrieval_path = settings.golden_dir / "retrieval.jsonl"
        rows = [json.loads(l) for l in retrieval_path.open()][:50]  # smoke subset
        print(f"loaded {len(rows)} queries from {retrieval_path.name}")
        """
    ),
    code(
        """
        from rag_evals.retrieval.dense import DenseRetriever

        dense = DenseRetriever()
        runs_dense, gold = {}, {}
        for r in rows:
            hits = dense(r["query"], limit=30)
            runs_dense[r["qid"]] = [h.doc_id for h in hits]
            gold[r["qid"]] = r["gold_doc_ids"]

        m = evaluate_runs(runs_dense, gold, k=10)
        print(f"dense Recall@10 = {m.recall_at_k:.3f}")
        print(f"dense MRR       = {m.mrr:.3f}")
        print(f"dense nDCG@10   = {m.ndcg_at_k:.3f}")
        """
    ),
]

# -----------------------------------------------------------------------------
# 02 - Hybrid + RRF
# -----------------------------------------------------------------------------
nb02 = [
    md(
        """
        # 02 — Hybrid retrieval and RRF

        Reproduce the article's RRF top-3 (`d3`, `d2`, `d1`), then compare dense / sparse / hybrid on
        the live scifact slice.
        """
    ),
    code(
        """
        from rag_evals.retrieval.hybrid_rrf import reciprocal_rank_fusion

        dense  = ["d3", "d7", "d1", "d4", "d2", "d9", "d10"]
        sparse = ["d2", "d3", "d8", "d1", "d11", "d4", "d6"]
        for doc, score in reciprocal_rank_fusion([dense, sparse], k=60)[:5]:
            print(f"  {doc}  score={score:.5f}")
        """
    ),
    code(
        """
        import json
        from rag_evals.config import settings
        from rag_evals.evaluation.retrieval import evaluate_runs
        from rag_evals.index.qdrant_store import QdrantStore
        from rag_evals.retrieval.dense import DenseRetriever
        from rag_evals.retrieval.sparse import SparseRetriever
        from rag_evals.retrieval.hybrid_rrf import HybridRetriever

        rows = [json.loads(l) for l in (settings.golden_dir / "retrieval.jsonl").open()][:50]

        # Share a single store across retrievers — embedded Qdrant only allows
        # one client per storage folder.
        store = QdrantStore()
        dense = DenseRetriever(store=store)
        sparse = SparseRetriever(store=store)
        hybrid = HybridRetriever(dense, sparse, k=60)

        results = {"dense": {}, "sparse": {}, "hybrid": {}}
        gold = {}
        for r in rows:
            qid, q = r["qid"], r["query"]
            results["dense"][qid]  = [h.doc_id for h in dense(q, limit=30)]
            results["sparse"][qid] = [h.doc_id for h in sparse(q, limit=30)]
            results["hybrid"][qid] = [h.doc_id for h in hybrid(q, limit=30)]
            gold[qid] = r["gold_doc_ids"]

        for name, runs in results.items():
            m = evaluate_runs(runs, gold, k=10)
            print(f"{name:>7}: Recall@10={m.recall_at_k:.3f} MRR={m.mrr:.3f} nDCG@10={m.ndcg_at_k:.3f}")
        """
    ),
    md(
        """
        On the article's claim: "hybrid Recall@10 ≥ max(dense, sparse)" — true on most corpora;
        verify on yours before trusting it.
        """
    ),
]

# -----------------------------------------------------------------------------
# 03 - Reranking
# -----------------------------------------------------------------------------
nb03 = [
    md(
        """
        # 03 — Reranking

        Stack a cross-encoder (`bge-reranker-v2-m3`) on top of hybrid and measure ΔnDCG / ΔPrecision@1.
        """
    ),
    code(
        """
        import json
        from rag_evals.config import settings
        from rag_evals.evaluation.retrieval import evaluate_runs, precision_at_k
        from rag_evals.retrieval.dense import DenseRetriever
        from rag_evals.retrieval.sparse import SparseRetriever
        from rag_evals.retrieval.hybrid_rrf import HybridRetriever
        from rag_evals.retrieval.reranker import CrossEncoderReranker
        from rag_evals.index.qdrant_store import QdrantStore

        rows = [json.loads(l) for l in (settings.golden_dir / "retrieval.jsonl").open()][:30]
        store = QdrantStore()
        hybrid = HybridRetriever(DenseRetriever(store=store), SparseRetriever(store=store), k=60)
        rerank = CrossEncoderReranker()

        before, after, gold = {}, {}, {}
        for r in rows:
            hits = hybrid(r["query"], limit=20, per_lane=50)
            before[r["qid"]] = [h.doc_id for h in hits]
            after[r["qid"]]  = [h.doc_id for h in rerank(r["query"], hits, limit=10)]
            gold[r["qid"]]   = r["gold_doc_ids"]

        before_m = evaluate_runs(before, gold, k=10)
        after_m  = evaluate_runs(after, gold, k=10)
        print(f"before rerank: nDCG@10={before_m.ndcg_at_k:.3f} P@10={before_m.precision_at_k:.3f}")
        print(f" after rerank: nDCG@10={after_m.ndcg_at_k:.3f} P@10={after_m.precision_at_k:.3f}")
        print(f"   ΔnDCG@10 = {after_m.ndcg_at_k - before_m.ndcg_at_k:+.3f}")
        """
    ),
]

# -----------------------------------------------------------------------------
# 04 - Filter false-exclusion (the centrepiece)
# -----------------------------------------------------------------------------
nb04 = [
    md(
        """
        # 04 — Filter false-exclusion rate

        The article's signature metric. A hard metadata filter can drop effective recall to zero
        without changing the standard retrieval metrics. The gold doc is excluded *before* ranking
        starts, so Recall@k computed over survivors looks fine.
        """
    ),
    code(
        """
        # First: reproduce the article's worked example exactly
        from rag_evals.evaluation.filter_exclusion import rate_against_survivors

        DOCS = [
            {"id": "d1", "tenant": "acme",   "locale": "en-US"},
            {"id": "d2", "tenant": "acme",   "locale": "en-GB"},
            {"id": "d3", "tenant": "globex", "locale": "en-US"},
            {"id": "d4", "tenant": "acme",   "locale": "en-US"},
            {"id": "d5", "tenant": "acme",   "locale": "de-DE"},
        ]

        def survivors_for(predicate):
            return {d["id"] for d in DOCS if all(d.get(k) == v for k, v in predicate.items())}

        queries = [
            {"qid": "q1", "gold_doc_ids": ["d2"], "filter_predicate": {"locale": "en-US"}},
            {"qid": "q2", "gold_doc_ids": ["d4"], "filter_predicate": {"tenant": "acme"}},
            {"qid": "q3", "gold_doc_ids": ["d3"], "filter_predicate": {"tenant": "acme"}},
            {"qid": "q4", "gold_doc_ids": ["d5"], "filter_predicate": {"locale": "de-DE"}},
        ]
        result = rate_against_survivors(queries, survivors_for)
        print(f"filter_false_exclusion_rate = {result.rate:.0%}")
        print(f"excluded queries: {[r.qid for r in result.rows if r.gold_excluded]}")
        """
    ),
    md(
        """
        Now run on the live golden set. 30% of rows have a deliberately corrupted predicate
        (see `data/golden.py`); the harness should detect them.
        """
    ),
    code(
        """
        import json
        from rag_evals.config import settings
        from rag_evals.index.qdrant_store import QdrantStore
        from rag_evals.evaluation.filter_exclusion import rate_against_survivors

        rows = [json.loads(l) for l in (settings.golden_dir / "filter_aware.jsonl").open()][:100]
        store = QdrantStore()

        result = rate_against_survivors(rows, lambda p: store.survivor_ids(p))
        print(f"filter_false_exclusion_rate = {result.rate:.2%}  ({result.n_excluded}/{result.n_queries})")
        """
    ),
    md(
        """
        The "aha": with a *bad* predicate, standard Recall@10 over the survivor set looks plausible —
        but the gold doc was already gone. Compare:
        """
    ),
    code(
        """
        from rag_evals.retrieval.dense import DenseRetriever
        from rag_evals.evaluation.retrieval import recall_at_k

        # Reuse the store opened above — embedded Qdrant locks one client per folder.
        dense = DenseRetriever(store=store)
        recalls = []
        for r in rows[:30]:
            hits = dense(r["query"], limit=10, predicates=r["filter_predicate"])
            ranked = [h.doc_id for h in hits]
            recalls.append(recall_at_k(ranked, r["gold_doc_ids"], 10))

        print(f"Recall@10 over survivor set: {sum(recalls)/len(recalls):.2%}")
        print("Without filter_false_exclusion_rate, this number hides the broken queries.")
        """
    ),
]

# -----------------------------------------------------------------------------
# 05 - Faithfulness
# -----------------------------------------------------------------------------
nb05 = [
    md(
        """
        # 05 — Faithfulness

        Decompose answers into atomic claims, verify each against the retrieved context. Uses the
        deterministic heuristic verifier so this notebook runs offline.
        """
    ),
    code(
        """
        from rag_evals.evaluation.faithfulness import faithfulness

        context = (
            "Mars has two moons, Phobos and Deimos. NASA's Curiosity rover landed on Mars in 2012."
        )
        answer = (
            "Mars has two moons. Phobos and Deimos orbit Mars. "
            "Mars has a thick atmosphere. Curiosity landed in 2012."
        )

        result = faithfulness(answer, context, use_heuristic=True)
        for v in result.verdicts:
            mark = "✓" if v.supported else "✗"
            print(f"  [{mark}] {v.claim}")
        print(f"faithfulness = {result.score:.2f}")
        """
    ),
    md("Switch `use_heuristic=False` (with API keys) to use the OpenAI SGR judge."),
]

# -----------------------------------------------------------------------------
# 06 - Lost-in-the-middle
# -----------------------------------------------------------------------------
nb06 = [
    md(
        """
        # 06 — Lost-in-the-middle

        Place the gold chunk at positions {first, middle, last} and measure correctness. With
        `RAG_EVALS_BACKEND=mock`, the LLM returns deterministic stubs — the structure of the
        eval is the point, not the absolute numbers.
        """
    ),
    code(
        """
        from rag_evals.evaluation.lost_in_middle import position_stratified_eval
        from rag_evals.types import RetrievalHit

        gold = RetrievalHit(doc_id="gold", score=1.0, text="The capital of France is Paris.")
        distractors = [
            RetrievalHit(doc_id=f"d{i}", score=0.0, text=f"Distractor passage {i}.")
            for i in range(6)
        ]

        result = position_stratified_eval(
            "What is the capital of France?",
            gold,
            distractors,
            is_correct=lambda a: "paris" in a.lower(),
        )
        for pos, acc in result.accuracy_by_position.items():
            print(f"  {pos:>6}: accuracy={acc:.2f}")
        """
    ),
]

# -----------------------------------------------------------------------------
# 07 - LLM-as-judge w/ bias mitigation
# -----------------------------------------------------------------------------
nb07 = [
    md(
        """
        # 07 — LLM-as-judge

        SGR pointwise scoring, mirrored pairwise comparison and same-provider limitations.
        """
    ),
    code(
        """
        from rag_evals.evaluation.llm_judge import (
            alternate_judges, g_eval, measure_position_bias, pairwise,
        )
        from rag_evals.generation.models import Model

        # Alternative OpenAI judges for each generator
        for gen in (Model.GPT_5_6_LUNA, Model.GPT_5_6_TERRA, Model.GPT_6_ASTRA):
            judges = alternate_judges(gen)
            print(f"generator={gen.value:<25}  judges={[j.value for j in judges]}")
        """
    ),
    code(
        """
        # Position-bias measurement on a tiny pair set (uses MockBackend if no API key)
        pairs = [
            (
                "What is the capital of France?",
                "Paris is the capital of France.",
                "France's capital is the city of Paris.",
            ),
            (
                "Who painted the Mona Lisa?",
                "Leonardo da Vinci painted the Mona Lisa.",
                "The Mona Lisa was painted by Leonardo da Vinci.",
            ),
        ]
        bias = measure_position_bias(pairs)
        print(bias)
        """
    ),
    md(
        """
        With API keys set, swap the default model in `LLM(...)` to a real model and watch
        the bias measurement change as you change the judge model.
        """
    ),
]

# -----------------------------------------------------------------------------
# 08 - Full eval dashboard
# -----------------------------------------------------------------------------
nb08 = [
    md(
        """
        # 08 — Full eval dashboard

        One-pager: retrieval metrics, filter false-exclusion, latency p50/p95/p99 — the same
        numbers `make eval` writes to `report.md`, plotted side by side.
        """
    ),
    code(
        """
        from rag_evals.evaluation.runner import run_suite

        result = run_suite(suite="all", limit=50, k=10)
        print(result.keys())
        """
    ),
    code(
        """
        import pandas as pd
        if "retrieval" in result:
            df = pd.DataFrame([result["retrieval"]])
            display(df.T.rename(columns={0: "value"}))
        """
    ),
    code(
        """
        if "latency" in result:
            df = pd.DataFrame(result["latency"]).T
            display(df)
        """
    ),
    code(
        """
        if "gates" in result:
            df = pd.DataFrame(result["gates"])
            display(df)
        """
    ),
]


nb09 = [
    md(
        """
        # 09 — Benchmark report

        Visualises the chunking × embedding × LLM sweep produced by
        `python -m rag_evals.scripts.benchmark`. The script writes
        `report/benchmark.json`; this notebook loads it and renders the same
        tables + a few charts.

        Re-run `make benchmark` (or the script directly) to refresh the JSON.
        """
    ),
    code(
        """
        import json, pandas as pd
        from pathlib import Path
        from IPython.display import Markdown, display
        bench = json.loads(Path("../report/benchmark.json").read_text())
        print({k: type(v).__name__ for k, v in bench.items()})
        bench["settings"]
        """
    ),
    md("## ⚠ Mock-data warning"),
    code(
        """
        # If the underlying benchmark run involved mock LLMs, every LLM-derived
        # number below (faithfulness, latency, win-rates, sample answers) is
        # a deterministic stub or fixture replay — NOT a real model evaluation.
        if bench.get("has_mock_data"):
            display(Markdown(
                "> ⚠️  **MOCK DATA — NOT A REAL EVALUATION** ⚠️\\n"
                ">\\n"
                f"> {bench.get('mock_warning', '')}\\n"
                ">\\n"
                "> Rows tagged `[MOCK]` below were produced (in whole or in part) by a mock LLM."
            ))
        else:
            print("benchmark.json reports no mock data — all rows are live.")
        """
    ),
    md("## Chunking sweep — embedding fixed, chunking varied"),
    code(
        """
        df_chunk = pd.DataFrame(bench["chunking_sweep"])
        if not df_chunk.empty:
            cols = ["config", "n_chunks", "index_secs", "recall_at_10", "mrr", "ndcg_at_10", "map", "coverage"]
            display(df_chunk[cols].set_index("config"))
        """
    ),
    code(
        """
        import matplotlib.pyplot as plt
        if not df_chunk.empty:
            ax = df_chunk.set_index("config")[["recall_at_10", "mrr", "ndcg_at_10"]].plot(
                kind="bar", figsize=(8, 4), title="Chunking sweep — retrieval metrics"
            )
            ax.set_ylabel("score")
            ax.set_ylim(0, 1)
            plt.xticks(rotation=20, ha="right")
            plt.tight_layout()
            plt.show()
        """
    ),
    md("## Embedding sweep — chunking fixed, embedding varied"),
    code(
        """
        df_emb = pd.DataFrame(bench["embedding_sweep"])
        if not df_emb.empty:
            cols = ["config", "n_chunks", "index_secs", "recall_at_10", "mrr", "ndcg_at_10", "map", "coverage"]
            display(df_emb[cols].set_index("config"))
        """
    ),
    code(
        """
        if not df_emb.empty:
            ax = df_emb.set_index("config")[["recall_at_10", "mrr", "ndcg_at_10"]].plot(
                kind="bar", figsize=(8, 4), title="Embedding sweep — retrieval metrics"
            )
            ax.set_ylabel("score")
            ax.set_ylim(0, 1)
            plt.xticks(rotation=20, ha="right")
            plt.tight_layout()
            plt.show()
        """
    ),
    md("## LLM sweep — generator varies"),
    code(
        """
        df_llm = pd.DataFrame(bench["llm_sweep"])
        if not df_llm.empty:
            # Tag mock rows so the dataframe view makes the contamination obvious.
            if "is_mock" in df_llm.columns:
                df_llm["model"] = df_llm.apply(
                    lambda r: ("[MOCK] " if r.get("is_mock") else "") + r["model"], axis=1
                )
            display(df_llm.set_index("model"))
            if df_llm.get("is_mock", pd.Series(dtype=bool)).any():
                display(Markdown(
                    "**⚠ Rows prefixed `[MOCK]` are not real evaluations — "
                    "their faithfulness, latency, and sample answer come from "
                    "MockBackend. Re-run with live API keys to replace.**"
                ))
        """
    ),
    code(
        """
        if not df_llm.empty:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            df_llm.set_index("model")[["faithfulness_heuristic", "faithfulness_llm"]].plot(
                kind="bar", ax=axes[0], title="Faithfulness by model"
            )
            axes[0].set_ylim(0, 1)
            df_llm.set_index("model")[["avg_latency_ms", "p95_latency_ms"]].plot(
                kind="bar", ax=axes[1], title="Latency (ms)"
            )
            if bench.get("has_mock_data"):
                for ax in axes:
                    ax.set_title(ax.get_title() + "  (⚠ contains MOCK rows)")
            for ax in axes:
                ax.tick_params(axis="x", rotation=20)
            plt.tight_layout()
            plt.show()
        """
    ),
    md("## Pairwise judging (same-provider judge)"),
    code(
        """
        df_pw = pd.DataFrame(bench.get("pairwise", []))
        if not df_pw.empty:
            if "is_mock" in df_pw.columns and df_pw["is_mock"].any():
                df_pw = df_pw.copy()
                df_pw["a"] = df_pw.apply(lambda r: ("[MOCK] " if r["is_mock"] else "") + r["a"], axis=1)
                df_pw["b"] = df_pw.apply(lambda r: ("[MOCK] " if r["is_mock"] else "") + r["b"], axis=1)
                display(df_pw)
                display(Markdown(
                    "**⚠ `[MOCK]`-tagged rows had at least one mock generator — "
                    "win counts reflect a deterministic stub, not real preference.**"
                ))
            else:
                display(df_pw)
        else:
            print("no pairwise data — re-run with live LLM keys to populate")
        """
    ),
]


def main() -> None:
    write(OUT / "00_setup_and_index.ipynb", nb00)
    write(OUT / "01_retrieval_metrics.ipynb", nb01)
    write(OUT / "02_hybrid_and_rrf.ipynb", nb02)
    write(OUT / "03_reranking.ipynb", nb03)
    write(OUT / "04_filter_false_exclusion.ipynb", nb04)
    write(OUT / "05_faithfulness.ipynb", nb05)
    write(OUT / "06_lost_in_the_middle.ipynb", nb06)
    write(OUT / "07_llm_as_judge.ipynb", nb07)
    write(OUT / "08_full_eval_dashboard.ipynb", nb08)
    write(OUT / "09_benchmark.ipynb", nb09)


if __name__ == "__main__":
    main()
