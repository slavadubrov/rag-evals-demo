# Faithfulness

`faithfulness(answer, context, llm=judge)` extracts atomic claims through SGR and
verifies each with `SUPPORTED` or `NOT_SUPPORTED`. A supported claim must include
an exact quote from context. Any invalid judgment makes the answer score null;
no claims produces `not_applicable`, not zero. Abstention correctness is separate.

The score is supported claims divided by all claims. A copied quote is a necessary
check, not proof of entailment. Calibration includes negation and prompt-injection
cases. `use_heuristic=True` uses sentence splitting and lexical overlap with no API
call; it ignores negation and relations and must not be interpreted as faithfulness.

Citation-set support uses only the union of explicitly cited passages. ID validity
only checks whether a citation exists in context. These are different measurements.

No local NLI implementation is included. See the [Ragas taxonomy](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/)
for the distinction between faithfulness, relevance, correctness and context metrics.
