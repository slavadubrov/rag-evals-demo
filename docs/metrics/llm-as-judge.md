# LLM-as-judge

> G-Eval and pairwise preference, plus the three biases the article warns about — position, verbosity, self-preference — and the mitigations that close them.

## What it measures

Two scorers:

- **G-Eval** — pointwise rubric scoring. The judge produces an integer 1–5 against a natural-language criterion ("faithfulness", "completeness", "concision").
- **Pairwise** — the judge picks the better answer (A vs B). Bypasses absolute-score calibration. Roughly 80% human-judge agreement at the GPT-4-class tier under MT-Bench conditions.

## Why it matters

Holistic answer quality is hard to capture in a single number, and traditional NLP metrics (BLEU, ROUGE) correlate poorly with human judgement on open-ended generations. An LLM judge does much better — but it inherits its base model's biases. The article calls out three.

## Bias 1: position

Judges prefer the first or second answer regardless of quality. **Mitigation:** randomize order, or run both orderings and average.

```python
def measure_position_bias(pairs, *, llm, criterion="faithfulness") -> BiasMeasurement
def averaged_pairwise(question, a, b, *, llm, criterion="faithfulness") -> PairwiseResult
```

`measure_position_bias` runs each pair in both orderings and reports how often the first-shown answer wins. A clean judge sits at ~50% in both directions; a biased one shows a measurable skew.

## Bias 2: verbosity

Judges prefer longer answers. The 2025–2026 research is more nuanced — modern instruction-tuned judges actually penalize filler on length-controlled tests but reward genuine completeness. **Mitigation:** explicitly tell the judge how to treat length in the rubric, and consider length-controlled win rates.

## Bias 3: self-preference

GPT-4 prefers GPT-4 outputs. The bias correlates with output perplexity — judges prefer text that is familiar to them. **Mitigation:** never use a model to judge itself.

```python
def cross_family_judges(generator: Model) -> list[Model]
```

`cross_family_judges` returns judges that are not in the same family as the generator. Notebook 07 uses this to run three judges (`GPT_5_MINI`, `CLAUDE_HAIKU_4_5`, `GEMINI_2_5_FLASH`) against each generator and visualises the per-judge win-rate skew.

## Implementation

`src/rag_evals/evaluation/llm_judge.py`:

```python
def g_eval(question, answer, *, criterion="faithfulness", llm=None) -> int
def pairwise(question, a, b, *, criterion="faithfulness", llm=None) -> PairwiseResult
def measure_position_bias(pairs, *, llm, criterion="faithfulness") -> BiasMeasurement
def averaged_pairwise(question, a, b, *, llm, criterion="faithfulness") -> PairwiseResult
def cross_family_judges(generator: Model) -> list[Model]
```

## How to run

```bash
jupyter notebook notebooks/07_llm_as_judge.ipynb
```

The notebook walks through: a baseline pairwise run, position-bias measurement, averaging-mitigation, and cross-family judging. With the default `RAG_EVALS_BACKEND=mock`, the judges return SHA1-keyed deterministic verdicts so the bias visualisation is reproducible.

## References

- Zheng et al., [*Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*](https://arxiv.org/abs/2306.05685), NeurIPS 2023.
- Upadhyay et al., [*Support Evaluation for the TREC 2024 RAG Track*](https://arxiv.org/abs/2504.15205), SIGIR 2025.
- Article: [§ LLM-as-judge](../../README.md#whats-evaluated).
