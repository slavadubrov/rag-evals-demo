# Latency and cost

> p50 / p95 / p99 per pipeline stage, plus a hook for token-cost accounting.

## What it measures

Wall-clock time per pipeline stage (retrieve, rerank, generate, post-process) and aggregate quantiles. The article's recommended SLO target is p95; p99 is what you alert on. Token counts feed cost: `litellm` exposes per-call usage that plugs into the `Tracer.add_tokens` hook.

## Why it matters

p95 spikes are almost always rerankers running on CPU. You will not see this without a per-stage breakdown. Cache hit rates at the embedding, retrieval, and KV-cache levels are usually the cheapest single optimisation in a RAG stack — and you cannot tune them without measuring the relevant stage.

## Implementation

`src/rag_evals/evaluation/latency.py`:

```python
@dataclass
class Tracer:
    @contextmanager
    def stage(name: str)
    def add_tokens(model: str, n: int)

def quantiles(values: list[float]) -> StageStats
def summarise(tracers: list[Tracer]) -> dict[str, StageStats]
def mean_total(tracers: list[Tracer]) -> float
```

Usage:

```python
tr = Tracer()
with tr.stage("retrieve"):
    hits = retrieve(query)
with tr.stage("rerank"):
    hits = rerank(query, hits)
with tr.stage("generate"):
    answer = llm.ask(prompt)
```

`runner.py` aggregates a list of `Tracer` instances across the eval set and emits a per-stage p50/p95/p99 table to `report.md`.

## How to run

The latency block is part of the default eval suite:

```bash
make eval                # writes per-stage p50/p95/p99 to report.md
```

For deep dives, notebook 08 (`08_full_eval_dashboard.ipynb`) plots per-stage timing distributions side by side.

## Notes on production telemetry

The demo deliberately stays light. For production, swap `Tracer` for OpenTelemetry. Both Phoenix and TruLens speak OTEL natively, and `litellm` ships an OTEL exporter. The shape of the metric does not change, only the transport.

## References

- Article: [§ System-Level Evaluation: Latency and cost](../../README.md#whats-evaluated).
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — OTEL-native tracing if you want to graduate.
