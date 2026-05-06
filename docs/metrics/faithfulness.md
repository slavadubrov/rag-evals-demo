# Faithfulness

> RAGAS-style: decompose the answer into atomic claims, verify each against the retrieved context, report the fraction supported.

## What it measures

For a generated answer `A` and the retrieved context `C`, faithfulness is the fraction of atomic claims in `A` that are supported by `C`. The structure (per-claim verdicts) is more useful than the score alone — it tells you *which* claims drift.

## Why it matters

Faithfulness measures the last link in the chain. It cannot detect a parsing bug, a chunking bug, an embedding drift, or a filter exclusion. But it does catch the most visible failure mode: the model stating things the context does not support.

## Definition

```text
claims          = decompose(answer)            # LLM-driven atomic-claim extraction
verdicts        = [verify(c, context) for c in claims]
faithfulness    = #supported / #claims
```

## Implementation

`src/rag_evals/evaluation/faithfulness.py`. Three pieces:

```python
def extract_claims(answer, *, llm=None) -> list[str]
def llm_verify(claims, context, *, llm=None) -> list[ClaimVerdict]
def heuristic_verify(claims, context) -> list[ClaimVerdict]
def faithfulness(answer, context, *, llm=None, use_heuristic=False) -> FaithfulnessResult
```

- `llm_verify` — uses the configured LLM (`Model.GPT_5_MINI` by default) for a single-token SUPPORTED/NOT_SUPPORTED verdict per claim.
- `heuristic_verify` — deterministic stand-in that flags a claim supported when all its content words (>3 chars) appear in the context. Used by tests and offline notebook runs.

Switching to a local NLI model (`cross-encoder/nli-deberta-v3-base`, the article's recommendation) is a one-function swap — replace `llm_verify` with an NLI scorer.

## How to run

```bash
jupyter notebook notebooks/05_faithfulness.ipynb
```

`make eval` does not currently invoke this; it runs offline-first metrics (retrieval, filter exclusion, latency) that don't need a live LLM. Add it as a gate by extending `runner._check_gates` — the threshold is already in `.env` (`THRESHOLD_FAITHFULNESS=0.85`).

## Reasonable targets

A medical or legal RAG below ~0.95 is unsafe. A general assistant at 0.85 is usually fine. Score by claim type as well as overall: a score of 0.9 with the unsupported claims being the only ones with numbers is much worse than 0.9 with the unsupported claims being throwaway transitions.

## Alternative: HHEM

The article cites Vectara's [HHEM-2.1-Open](https://huggingface.co/vectara/hallucination_evaluation_model) as a purpose-built classifier alternative. It is a 600 MB model that runs on CPU and reportedly outperforms generic LLM judges on AggreFact and RAGTruth. Worth swapping in when your eval volume makes LLM-judge cost dominate.

## References

- Es et al., [*RAGAS*](https://arxiv.org/abs/2309.15217), 2023.
- Min et al., [*FActScore*](https://arxiv.org/abs/2305.14251), EMNLP 2023.
- [Vectara HHEM](https://huggingface.co/vectara/hallucination_evaluation_model).
- Article: [§ Faithfulness and groundedness](../../README.md#whats-evaluated).
