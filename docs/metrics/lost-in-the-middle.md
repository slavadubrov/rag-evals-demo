# Lost-in-the-middle

> Position-stratified eval. When the gold chunk sits in the *middle* of the context, models often miss it. The U-shaped degradation that Liu et al. (TACL 2023) first documented is real, and it persists in 2026 models.

## What it measures

For a query whose gold chunk is known, build the context three different ways:

- **first.** Gold chunk at position 0, distractors after.
- **middle.** Gold chunk in the middle of the distractors.
- **last.** Distractors first, gold chunk at the end.

Generate an answer from each arrangement; score correctness per position. A model with no position bias should score equally across all three. In practice, middle is meaningfully worse.

## Why it matters

This is one of the most common "good retrieval, bad answer" failure modes. Retrieval can put the gold chunk in your top-5, but if it lands at position 3 in a 5-chunk context the model sometimes misses it. The standard mitigation in production is **rerank then reorder**: place the highest-scored chunk first or last in the prompt, not by retrieval rank.

## Implementation

`src/rag_evals/evaluation/lost_in_middle.py`:

```python
def position_stratified_eval(
    query: str,
    gold_chunk: RetrievalHit,
    distractors: Sequence[RetrievalHit],
    *,
    is_correct: Callable[[str], bool],
    llm: LLM | None = None,
    positions: Sequence[str] = ("first", "middle", "last"),
) -> PositionEvalResult
```

Pass an `is_correct` callable so the harness stays domain-agnostic: exact match for short answers, embedding cosine threshold for open-ended ones, an LLM judge if you need it.

## How to run

```bash
jupyter notebook notebooks/06_lost_in_the_middle.ipynb
```

The notebook visualises accuracy by position. Expect a U-shape: first ≥ last > middle.

## Mitigation: rerank + reorder

If you have a cross-encoder reranker (which you should — see `retrieval/reranker.py`), the simplest mitigation is to take the reranked top-k and *reorder* them so the highest-scored chunk is first or last in the prompt. Compress the middle chunks aggressively if context is tight.

## References

- Liu et al., [*Lost in the Middle*](https://arxiv.org/abs/2307.03172), TACL 2023.
- Article: [§ Context construction and lost-in-the-middle](../../README.md#whats-evaluated).
