# Filter false-exclusion rate

> The article's signature metric. Catches the silent failure where a hard metadata filter removes the right document *before* ranking starts.

## What it measures

Per query, did the applied filter predicate exclude the gold document from the candidate set? The metric is the fraction of queries (with at least one gold doc) where this happened.

```text
filter_false_exclusion_rate =
    | { q : gold(q) ∩ survivors(predicate(q)) = ∅ } |
    / | { q : gold(q) ≠ ∅ } |
```

## Why it matters

A wrong tag, a brittle hard predicate, or a buggy LLM-driven extractor can collapse effective recall to zero. Standard retrieval metrics will not see it: Recall@k is computed over the surviving candidate set, so it can look fine. Faithfulness can also look fine — the model faithfully said "I don't know."

This is the metric the article argues most teams skip and most production failures hide behind.

## Implementation

`src/rag_evals/evaluation/filter_exclusion.py`. Two scorers and a classifier:

```python
def filter_false_exclusion_rate(queries, survives) -> FilterEvalResult
def rate_against_survivors(queries, survivors_for) -> FilterEvalResult
def predicate_precision_recall(predicted, gold) -> PredicateClassifierMetrics
```

`rate_against_survivors` is the production path: pass a closure like `lambda p: store.survivor_ids(p)` for a real Qdrant collection. `QdrantStore.survivor_ids` scrolls the index without vector search and returns the survivor doc-id set, which is exactly what the metric needs.

The result includes a per-query `reason` (`gold-not-in-survivors` vs `gold-survives`) so you can debug *which* queries are silently broken instead of staring at an aggregate.

## How to run

```bash
# Notebook 04 walks through this end-to-end on scifact + synthesized metadata
jupyter notebook notebooks/04_filter_false_exclusion.ipynb

# CI gate (uses data/golden/filter_aware.jsonl)
uv run python -m rag_evals.evaluation.runner --suite filter --report report.md
```

The default golden set is built with a 30% deliberately-corrupted predicate rate (see `src/rag_evals/data/golden.py`), so the demo always has a non-trivial exclusion rate to detect.

## Decision rule: hard filter vs soft boost

The article's measurable rule:

```text
For each filter predicate F:
  hard_recall_F   = retrieval recall@k with F as a hard filter
  soft_recall_F   = retrieval recall@k with F as a +0.X rerank boost
  hard_precision  = relevant_in_top_k / k under hard filter
  soft_precision  = relevant_in_top_k / k under soft boost
  exclusion_rate  = filter_false_exclusion_rate

Use hard filter only if exclusion_rate < ε AND hard_precision >> soft_precision.
Otherwise prefer soft boost.
```

ε in 1–2% range is a reasonable default; lower for high-stakes domains. `THRESHOLD_FILTER_FALSE_EXCLUSION` in `.env` enforces this in CI.

## Companion metric: predicate precision/recall

When the predicate comes from an LLM extractor, treat the extractor as a classifier and evaluate it as one. `predicate_precision_recall` operates on lists of predicate dicts and counts exact (field, value) matches. If the extractor is wrong 8% of the time and applies hard filters, you have a hard ceiling on recall around 92% — no amount of reranking helps.

## References

- Article: [§ Part 5 — The Filter False-Exclusion Rate](../../README.md#whats-evaluated).
- Tested in `tests/test_filter_exclusion.py::test_50_percent_exclusion_rate` — reproduces the article's worked example exactly.
