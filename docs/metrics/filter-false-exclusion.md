# Filter false exclusion

For each query, count whether no **eligible relevant** document survives the
combined authorization and search predicates. Divide by queries with nonempty
eligible gold. Queries with no eligible gold are reported as `no-gold`, not errors.

Authorization is immutable: never soften tenant/access-control predicates into a
ranking boost. The synthetic experiment corrupts only locale/domain search metadata.
If authorization predicates are supplied, eligible gold must be explicit. A search
predicate cannot override an authorization field.

The standard recall metric against original eligible gold also detects lost
results. False exclusion isolates the loss before ranking; recomputing gold only
from survivors would hide it.

`rate_against_survivors()` asks Qdrant for all surviving IDs and caches repeated
predicates within a run. `filter_false_exclusion_rate()` accepts a per-document
predicate callback. Both return per-query reasons and population counts.
`predicate_precision_recall()` separately scores exact extracted field/value pairs;
its error rate is not automatically an equivalent bound on retrieval recall.

```bash
uv run python -m rag_evals.evaluation.runner --suite filter --report report/filter.md
```

The generated set deliberately corrupts about 30% of search predicates. That
population is diagnostic. With `--quality-gates`, only uncorrupted rows use the
configured false-exclusion threshold. Calibrate thresholds to your own data.

For non-security metadata, compare hard filtering and soft ranking on the same
eligible gold population before choosing. The demo does not automatically soften
filters. `tests/test_filter_exclusion.py` retains the article's 50% worked example;
`tests/test_audit_contracts.py` checks authorization boundaries.
