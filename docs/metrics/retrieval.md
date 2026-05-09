# Retrieval metrics

> Recall@k, Precision@k, MRR, nDCG@k, MAP, Hit Rate@k, Coverage. The baseline metrics every RAG eval starts with.

## What they measure

Given a ranked list of retrieved doc ids and a set of gold doc ids per query, these metrics quantify *how high up* the relevant docs land. They cover the full retrieval pipeline (dense, sparse, hybrid, reranked) without knowing anything about generation.

## Why they matter

If retrieval is broken, generation cannot recover. The article argues these are the first metrics to wire up — earlier than any vector-database decision. Recall@k is the headline: a low Recall@k caps everything downstream.

## Definitions

```text
recall_at_k    = | top_k ∩ gold | / | gold |
precision_at_k = | top_k ∩ gold | / k
hit_rate_at_k  = 1 if top_k ∩ gold else 0
mrr            = mean over queries of 1 / rank_of_first_relevant
map            = mean over queries of average_precision
ndcg_at_k      = dcg_at_k / idcg_at_k     (binary relevance variant)
coverage       = | retrieved_universe ∩ gold_universe | / | gold_universe |
```

## Implementation

`src/rag_evals/evaluation/retrieval.py`. Public surface:

```python
def recall_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float
def precision_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float
def reciprocal_rank(ranked: Sequence[str], gold: Iterable[str]) -> float
def hit_rate_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float
def ndcg_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float
def average_precision(ranked: Sequence[str], gold: Iterable[str]) -> float
def evaluate_runs(runs, gold, *, k=10) -> RetrievalMetrics
```

## How to run

```bash
# Notebook tour
jupyter notebook notebooks/01_retrieval_metrics.ipynb

# CI gate
uv run python -m rag_evals.evaluation.runner --suite retrieval --report report.md
```

## Reasonable targets

The article quotes these as illustrative, not universal: Recall@10 ≥ 0.85, MRR ≥ 0.6, nDCG@10 ≥ 0.7. Calibrate to your domain. A medical RAG at 95% Recall@10 may still be unsafe; a brainstorm assistant at 70% is probably fine.

`make eval` enforces these via `THRESHOLD_RECALL_AT_10` and `THRESHOLD_MRR` from `.env`.

## References

- Cormack, Clarke, Buettcher, [*Reciprocal Rank Fusion Outperforms Condorcet…*](https://doi.org/10.1145/1571941.1572114), SIGIR 2009.
- Thakur et al., [*BEIR*](https://arxiv.org/abs/2104.08663), NeurIPS 2021.
- Article: [§ Retrieval metrics](../../README.md#whats-evaluated).
