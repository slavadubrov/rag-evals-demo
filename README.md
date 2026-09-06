# rag-evals

A small, inspectable RAG evaluation demo. It separates **retrieval quality on SciFact**, **generation quality on labeled synthetic QA**, and **offline contract checks**. The scores describe these datasets, not production readiness.

Python 3.12+, [uv](https://docs.astral.sh/uv/), embedded Qdrant, native OpenAI SDK. No service or Docker is required.

## Start without API calls

```bash
uv sync --extra dev
# Only if you do not already have .env:
cp -n .env.example .env
make test
make lint
uv run python -m rag_evals.evaluation.runner --suite offline --report report/offline.md
```

The offline suite replays five committed QA answers and checks deterministic metrics. It downloads no corpus or models and makes no API calls after dependencies are installed. It does **not** measure an LLM. Unrecorded mock judgments are invalid, never invented scores.

## Evaluate with OpenAI

Set `OPENAI_API_KEY` in `.env`. The default generator and SGR judge are `gpt-5.6-luna`; use `gpt-5.6-terra` for a separate model comparison. Existing `.env` model settings override defaults. No Anthropic or Gemini key is used.

```bash
uv run python -m rag_evals.scripts.probe_models gpt-5.6-luna gpt-5.6-terra
uv run python -m rag_evals.evaluation.runner \
  --suite generation --live --report report/generation.md
# Explicit alternative judge:
uv run python -m rag_evals.evaluation.runner \
  --suite generation --live --judge-model gpt-5.6-terra --report report/terra.md
```

Each live generation run first evaluates six labeled support/contradiction fixtures, then the heldout QA split. The five QA cases include answerable questions, missing information, and conflicting sources. Calibration accuracy is reported; these hand-authored checks are not a human-panel validation study. Two OpenAI models still share provider biases.

`--live` is required by the CLI. `LLM_TIMEOUT`, `LLM_MAX_TOKENS`, and `LLM_MAX_CALLS` bound each adapter instance; retries are disabled. Each judgment can require several calls. Provider failures, refusals, truncations, malformed schemas and unsupported evidence quotes remain visible as invalid results. Reports retain token usage and elapsed time; `cost_usd=null` means unpriced, not free. No streaming TTFT is measured.

## Evaluate retrieval

```bash
make index       # Downloads pinned SciFact and embedding weights; builds local Qdrant and gold files
make eval        # Retrieval + filter diagnostics + generation fixture replay
# Add live generation to the same report:
uv run python -m rag_evals.evaluation.runner --suite all --live --report report/all.md
# An explicit retrieval depth:
uv run python -m rag_evals.evaluation.runner --suite retrieval --k 5 --limit 30
```

The first index build needs network access, disk space and CPU time. Subsequent evaluation uses local caches. Ingest requires an empty collection to prevent stale chunks from surviving a rebuild. Set a fresh `QDRANT_PATH` or `QDRANT_COLLECTION` to rebuild. An incompatible vector dimension, distance or BM25 IDF configuration is rejected. Do not run two processes against the same embedded path simultaneously. Store ownership is explicit; use `with QdrantStore() as store` when composing components.

Execution failures fail the command. Quality thresholds are opt-in via `--quality-gates`, and must be calibrated on your development data. The Recall threshold applies only at k=10. Deliberately corrupted filter cases are diagnostics; only the clean subset is eligible for its quality gate. Authorization gold is immutable and excludes unauthorized documents from the false-exclusion denominator.

## Metrics and their limits

| Layer | Measures | Interpretation |
| --- | --- | --- |
| Retrieval | Recall@k, Precision@k, Hit@k, nDCG@k, MRR, MAP, universe coverage | Binary document qrels; deduplicate chunks; missing queries score zero. MRR/MAP use returned depth. |
| Filters | False exclusion; predicate precision/recall/F1 | Synthetic tenant/locale/domain metadata; search corruption never changes authorization. |
| Deterministic answers | Exact match, token F1, lexical nugget coverage, abstention correctness | Lexical agreement does not establish semantic correctness. |
| Citations | ID validity, gold-document recall, SGR citation-set support | Semantic support uses the union of cited passages, not per-sentence attribution. |
| SGR judge | Atomic-claim faithfulness, answer relevance, correctness, semantic nugget coverage, context nugget recall | Strict schema and short evidence observations; still fallible judgments. No claims is N/A. |
| Comparison | Both pairwise orders, swap consistency, slot bias | Invalid calls are not ties. This does not isolate self-preference or verbosity bias. |
| Diagnostics | Gold at first/middle/last; rerankers on identical candidate pools; bounded iterative retrieval | Explicit library experiments, not automatic claims about model quality. |
| Statistics | Query bootstrap 95% intervals; attempted/scored/missing/invalid counts | Small synthetic samples provide descriptive intervals, not release guarantees. |
| Telemetry | Request latency, stage p50/p95/p99, API token usage | Non-streaming latency; cost is unknown unless priced separately. |

SGR means schema-guided reasoning: observations and evidence precede a bounded decision in a provider-constrained schema, then Pydantic validates the response. It does not request private chain of thought. `g_eval()` is a compatibility alias for pointwise scoring, **not** the probability-weighted G-Eval algorithm. See [judge contracts](docs/metrics/llm-as-judge.md).

## Benchmarks

```bash
make benchmark  # Chunking and embedding sweeps; no paid calls
uv run python -m rag_evals.scripts.benchmark \
  --skip-chunking --skip-embedding --live --n-llm-queries 5
```

Each indexing variant uses a fresh collection and the same corpus slice/query population. The LLM arm compares Luna and Terra on the same fixed-context heldout QA cases. Pairwise evaluation mirrors both presentation orders and reports unpaired and invalid cases separately. Means are conditional on valid scores. Skipped arms stay absent; older results are never silently merged into a fresh report.

Results are written to Markdown and JSON under `report/`. JSON includes row-level evidence, counts, confidence intervals, requested/resolved model identifiers, prompt/suite versions, source/dataset hashes and dependency versions. Keep the JSON alongside any published numbers. Corpus and qrels revisions are pinned in `data/scifact.py`.

## Follow and extend the code

```text
src/rag_evals/
  data/          pinned SciFact loader, deterministic metadata and gold builders
  ingest/        bounded separator-aware chunking and embedding batches
  index/         Qdrant schema, storage lifecycle and exact metadata filters
  retrieval/     dense, BM25, document-level RRF, reranking, bounded query expansion
  generation/    OpenAI adapter, prompts, structured answer and citation validation
  evaluation/    pure metrics, SGR judges, runners, reports and provenance
  scripts/       indexing, model access probe, comparison sweeps
```

Follow `evaluation/runner.py` → retriever or `evaluation/generation.py` → metric → report. Modules accept retrievers, rerankers or LLM instances directly; there is no plugin framework. Add a metric as a pure function where possible, define N/A and invalid behavior, add one meaningful regression check, then wire it into aggregation/reporting.

To add QA cases, append records to `data/fixtures/generation.jsonl` with unique `qid`, `split`, evidence `context`, `reference_answers`, nugget aliases, `answerable`, `gold_doc_ids`, and `fixture_answer`. Keep calibration and heldout IDs disjoint; use your own human-reviewed data before drawing domain conclusions. SciFact abstracts in `data/golden/generation.jsonl` are evidence excerpts, never reference answers.

For reranker experiments use `evaluation.reranking.compare_rerankers(rows, retrieve, {name: reranker})`: retrieval happens once and every reranker receives the same candidate pool. Pass model IDs to `CrossEncoderReranker(model_name=...)`. For iterative retrieval, `retrieve_iteratively()` accepts explicit query reformulations, immutable predicates and call/time/token/input budgets; it does not invent an agent planner. The blocking retriever needs its own timeout.

[Architecture](docs/architecture.md) · [Corpus](docs/corpus.md)

## Notebooks and checks

```bash
uv sync --all-extras
uv run python notebooks/_build.py  # Regenerate notebooks from their source
make index                       # Required by notebooks 00–04 and 08
make benchmark                   # Required by notebook 09
make nb                          # Mock backend: examples, not model-quality evidence
make test
make lint
uv run python -m ruff format --check src tests
```

Notebooks are optional walkthroughs; CLI and tests define the executable contracts. CI runs lint, typing, tests and offline replay without API credentials. Python follows Ruff formatting/import/PEP checks and mypy; no claim of compliance with every PEP is meaningful.

## Sources

- [OpenAI Luna model](https://developers.openai.com/api/docs/models/gpt-5.6-luna) and [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs).
- [Ragas metric taxonomy](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/): this demo has small explicit evaluators, not a Ragas dependency or a claim of algorithmic equivalence.

MIT. Companion demo for *Evaluating RAG*.
