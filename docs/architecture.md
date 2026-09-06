# Architecture

The project keeps six boundaries: data, ingestion, storage, retrieval, generation,
and evaluation. Each has a concrete responsibility; orchestration lives in runners.

1. `data/scifact.py` loads pinned corpus/qrels revisions and writes complete cache files.
2. `ingest/pipeline.py` builds bounded chunks and initializes dense/sparse embedders once per ingest. A nonempty index is rejected to prevent stale points.
3. `index/qdrant_store.py` owns named vectors, cosine/IDF validation, filtering and explicit closure. It supports local and server Qdrant.
4. `retrieval/` embeds queries, fuses unique document ranks and optionally reranks. RRF retains one representative chunk per document; multi-passage aggregation is a separate design choice.
5. `generation/llm.py` is the only paid-call boundary. OpenAI JSON Schema constrains structured output; Pydantic and evidence checks validate it. There is no live-to-mock fallback after a failed call.
6. `evaluation/` defines denominators, nullable results and evidence. `runner.py` assembles suites; `provenance.py` writes Markdown/JSON via temporary files.

The fixed-context generation fixture isolates generation and judging. SciFact
retrieval uses real relevance labels. Neither population is silently substituted
for the other. The all-suite includes both but does not pretend to provide an
end-to-end human-reviewed SciFact answer benchmark.

Use direct dependency injection: pass a retriever callable, reranker or LLM object.
Tests substitute these boundaries rather than invoking paid services. A new metric
needs a result contract and denominator, not a registry or framework.

SGR contracts are in `evaluation/schemas.py`; the answer contract is in
`generation/rag.py`. Schema validity is distinct from truth. Judges retain invalid
status and coverage. Pointwise ratings are ordinal 1–5 rubrics; support is a strict
enum with an exact evidence quote. Abstention is measured separately.

`compare_rerankers` materializes candidate pools once. `retrieve_iteratively`
uses explicit reformulations with call, elapsed-time and character budgets; the
backend must bound each individual request. Model-generated expansion would also
need the existing LLM completion-token and call budgets.

Limits: embedded Qdrant has a single-process lock; use explicit `with` ownership.
Character-based chunks approximate token size. Local synthetic QA is a contract
fixture, not a representative domain benchmark. Same-provider judgments need
independent human calibration. Cost is unpriced and no TTFT is measured.
