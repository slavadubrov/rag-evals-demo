# Context-position diagnostic

`position_stratified_eval()` puts a fixed gold passage first, middle and last among
the same distractors, generates one response per position and applies the supplied
correctness function. Unknown positions are rejected. API failures propagate rather
than silently dropping an arrangement.

The notebook demonstrates the experiment with a mock backend. Its numbers are not
model accuracy. Use a live LLM and many labeled questions to measure a real position
effect. Do not assume a U-shape or prescribe reordering based on this tiny fixture.
A single question per position cannot support a robust comparison.

The correctness function belongs to the application. Exact answer matching can
work for short factual answers; lexical/embedding similarity alone does not prove
correctness for open-ended responses.

Original research: Liu et al., [Lost in the Middle](https://arxiv.org/abs/2307.03172).
The library diagnostic is separate from the CLI all-suite; see README for coverage.
