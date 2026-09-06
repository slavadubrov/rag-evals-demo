# Schema-guided judges

`pointwise()` returns a bounded 1–5 rating, status, evidence observation and explanation.
Faithfulness requires the actual retrieved context. Correctness requires references
and evidence. Invalid output returns a null score; it never becomes a midpoint.
`g_eval()` is a compatibility alias, not the probability-weighted G-Eval algorithm.

`pairwise()` takes evidence for both answers. `averaged_pairwise()` evaluates both
orders and maps labels to candidate identity. Disagreement becomes an explicit tie;
an invalid call remains invalid. `measure_position_bias()` reports swap consistency,
invalid pairs and ties. Slot preference is the first-position selection rate among
decisive choices minus 0.5; it does not demand that equally many candidates win.

All presets use OpenAI. `alternate_judges()` lists different model IDs, not independent
families. Neither different IDs nor swapped order eliminates self-preference or
verbosity bias. No controlled self-preference experiment is claimed.

Live suites calibrate atomic support on six hand-authored logical cases before
heldout QA. Replace these with domain-specific human labels for real model selection.
The report retains all attempted/invalid counts; calibration is not hidden by filtering.

[OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
constrains the schema, not factual validity. Evidence observations are short,
checkable statements, not a private reasoning transcript.
