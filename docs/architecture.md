# Architecture

`rag-evals` exists to make every metric in the *Evaluating RAG* article runnable on a real corpus. Production code lives in `src/rag_evals/`. Notebooks in `notebooks/` import from `src/` and visualise; no metric logic is duplicated in a notebook.

## Three layers

The article's central frame is that evaluation has three layers, each catching a different failure class:

- **Offline** — *was the knowledge base prepared correctly?* Parsing, cleaning, chunking, embedding, indexing.
- **Online** — *was the right evidence found and used for this query?* Query rewriting, retrieval, reranking, context assembly.
- **Post-generation** — *is the answer faithful and verifiable?* Faithfulness, citation accuracy, drift, telemetry.

Each layer maps onto a directory under `src/rag_evals/`:

| Layer            | Code                                                    | Notebooks |
| ---------------- | ------------------------------------------------------- | --------- |
| Offline          | `data/`, `ingest/`, `index/`                            | 00        |
| Online           | `retrieval/`, `generation/rag.py`                       | 01–04     |
| Post-generation  | `evaluation/faithfulness.py`, `evaluation/llm_judge.py` | 05–07     |
| Cross-cutting    | `evaluation/retrieval.py`, `evaluation/filter_exclusion.py`, `evaluation/latency.py` | 04, 08    |

## Single-query trace

```
make eval ─► runner.run_suite()
            │
            ▼
   for row in golden/retrieval.jsonl:
            │
            ▼
   ┌─ HybridRetriever ─────────────────────┐
   │  dense  : query  → bge-small-en-v1.5  │
   │           → Qdrant.search_dense       │
   │  sparse : query  → BM25               │
   │           → Qdrant.search_sparse      │
   │  fuse   : reciprocal_rank_fusion(k=60)│
   └────────────────────────────────────────┘
            │
            ▼
   evaluation.retrieval.evaluate_runs(runs, gold)
   evaluation.latency.summarise(tracers)
   evaluation.filter_exclusion.rate_against_survivors(...)
            │
            ▼
   evaluation.runner._check_gates  ──► report.md / report.json
                                       exit non-zero if any gate fails
```

## Qdrant collection

One collection (`scifact` by default), two named vectors per point:

- `dense`: 384-dim, cosine — `BAAI/bge-small-en-v1.5`.
- `sparse`: BM25 sparse vectors — FastEmbed's `Qdrant/bm25`.

Payload keeps the doc id, chunk id, raw text, and synthesized `tenant`/`locale`/`domain`. The metadata is what powers notebook 04 (filter false-exclusion) without needing a parallel index.

## LLM routing

Everything flows through a single `LLM` class wrapping `litellm.completion`. Model selection is the `Model` enum in `src/rag_evals/generation/models.py`:

```python
class Model(StrEnum):
    GPT_5_MINI = "gpt-5-mini"
    CLAUDE_HAIKU_4_5 = "claude-haiku-4-5"
    GEMINI_2_5_FLASH = "gemini/gemini-2.5-flash"
    ...
    MOCK = "mock"
```

The judge notebook picks judges from a *different family* than the generator (`cross_family_judges`) to demonstrate the article's "never use a model to judge itself" rule.

When `RAG_EVALS_BACKEND=auto` and the relevant API key is missing, the `LLM` falls back to `MockBackend`, a SHA1-keyed deterministic stub. The whole notebook tour runs offline.

## Why this layout

- **One module per metric family.** `evaluation/retrieval.py`, `evaluation/filter_exclusion.py`, `evaluation/faithfulness.py`, etc. Each is independently importable from a notebook in two lines and has a tests/ mirror.
- **Notebooks are thin.** They import, run, and visualise. A reader can drop any metric into their own pipeline by importing the same module.
- **`runner.py` is the CI surface.** One command writes `report.md` and exits non-zero on threshold violations. Drop into a GitHub Action without modification.
