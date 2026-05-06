# rag-evals

Companion demo to *Evaluating RAG: Metrics for Every Stage of a Production RAG System*. A runnable harness for the metrics the article discusses, on a real corpus.

## Quickstart

```bash
cp .env.example .env       # adjust keys; the suite runs offline if you skip them
uv sync --all-extras
make up                    # docker compose up -d qdrant
make index                 # ingest scifact + build golden sets (~3-5 min on CPU)
make eval                  # full suite -> report.md, exits non-zero on regressions
```

## Architecture

Three layers, mirroring the article's evaluation discipline:

- **Offline** — corpus, chunking, embedding, indexing into Qdrant (one collection, named dense + sparse vectors).
- **Online** — dense + sparse retrieval, hybrid fusion via RRF, optional cross-encoder reranking.
- **Post-generation** — faithfulness, lost-in-the-middle, LLM-as-judge with bias mitigation, latency telemetry.

See [`docs/architecture.md`](docs/architecture.md) for the full request/response trace.

```
query -> [retrieve dense] + [retrieve sparse] -> RRF fuse -> rerank -> generate -> evaluate
                                                                          |
                                              filter false-exclusion <----+----> faithfulness
```

## What's evaluated

| Metric                            | Module                                 | Notebook | Docs                                                  |
| --------------------------------- | -------------------------------------- | -------- | ----------------------------------------------------- |
| Recall@k, MRR, nDCG, MAP, etc.    | `evaluation/retrieval.py`              | 01       | [retrieval](docs/metrics/retrieval.md)                |
| Hybrid + RRF                      | `retrieval/hybrid_rrf.py`              | 02       | [retrieval](docs/metrics/retrieval.md)                |
| Reranker uplift                   | `retrieval/reranker.py`                | 03       | [retrieval](docs/metrics/retrieval.md)                |
| **Filter false-exclusion rate**   | `evaluation/filter_exclusion.py`       | 04       | [filter false-exclusion](docs/metrics/filter-false-exclusion.md) |
| Faithfulness (RAGAS-style)        | `evaluation/faithfulness.py`           | 05       | [faithfulness](docs/metrics/faithfulness.md)          |
| Lost-in-the-middle                | `evaluation/lost_in_middle.py`         | 06       | [lost-in-the-middle](docs/metrics/lost-in-the-middle.md) |
| LLM-as-judge w/ bias mitigation   | `evaluation/llm_judge.py`              | 07       | [llm-as-judge](docs/metrics/llm-as-judge.md)          |
| Latency p50/p95/p99               | `evaluation/latency.py`                | 08       | [latency-and-cost](docs/metrics/latency-and-cost.md)  |

The metric coverage is intentional: every section of the article that *can* be demoed on a single corpus has a destination here. Out of scope (called out so it's explicit): OCR / CER / WER, entity-linking F1, ontology hierarchical metrics, production drift, A/B testing.

## Notebooks tour

- **00 — Setup and index.** Spin Qdrant, ingest scifact, sanity-check counts.
- **01 — Retrieval metrics.** Dense baseline; Recall@k / MRR / nDCG sweep.
- **02 — Hybrid + RRF.** Dense vs BM25 vs RRF, per-query deltas.
- **03 — Reranking.** ΔnDCG and ΔPrecision@1 from the cross-encoder.
- **04 — Filter false-exclusion.** *Does my filter silently drop the right document before retrieval ever runs?* The article's signature metric on real metadata.
- **05 — Faithfulness.** Claim extraction + verification on generated answers.
- **06 — Lost-in-the-middle.** Position-stratified placement of the gold chunk.
- **07 — LLM-as-judge.** G-Eval, pairwise, position-bias measurement, cross-family judges.
- **08 — Full eval dashboard.** Every metric on the same eval set.

Notebooks run offline against `MockBackend` if you don't set any LLM API keys (`make nb` forces this).

## Reproducing the article numbers

```bash
# Recall@5 = 0.750, MRR = 0.625, nDCG@5 = 0.627
uv run pytest tests/test_retrieval_metrics.py -v

# Filter false-exclusion = 0.50 (the worked example from Part 5)
uv run pytest tests/test_filter_exclusion.py::test_50_percent_exclusion_rate -v

# RRF ordering d3 / d2 / d1 with k=60
uv run pytest tests/test_rrf.py -v
```

## Configuration

All knobs live in `.env` (read via Pydantic Settings).

- `RAG_EVALS_DEFAULT_MODEL` — generator + claim extractor (e.g. `gpt-5-mini`).
- `RAG_EVALS_JUDGE_MODEL` — second model for cross-family judging (e.g. `claude-haiku-4-5`).
- `RAG_EVALS_THIRD_JUDGE` — third leg for self-preference measurement (e.g. `gemini/gemini-3-flash`).
- `RAG_EVALS_BACKEND` — `auto` | `live` | `mock`. `auto` falls back to `mock` when API keys are missing.
- `EMBEDDING_MODEL`, `RERANKER_MODEL`, `NLI_MODEL` — Hugging Face IDs.
- `THRESHOLD_*` — pass/fail gates used by `make eval` to exit non-zero on regression.

Adding a new LLM is one enum line in `src/rag_evals/generation/models.py` — LiteLLM does the rest.

## Project layout

```
src/rag_evals/
  config.py             settings (Pydantic, .env)
  types.py              Document, Chunk, Query, RetrievalHit, RAGAnswer
  data/                 scifact loader, synthetic metadata, golden sets
  ingest/               chunking + ingest pipeline
  index/                Qdrant store (named dense + sparse vectors)
  retrieval/            dense, sparse, hybrid_rrf, reranker, filters
  generation/           Model enum, LiteLLM-backed LLM, prompts, end-to-end RAG
  evaluation/           retrieval, filter_exclusion, faithfulness,
                        lost_in_middle, llm_judge, latency, runner, report
notebooks/              demo tour, 00-08
tests/                  unit tests, including article-fidelity fixtures
docs/                   architecture + per-metric reference pages
```

## License

MIT. Companion to [the article](https://slavadubrov.github.io/) — see the `Evaluating RAG` post.
