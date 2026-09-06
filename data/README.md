# data/

`data/cache/` — fetched scifact (corpus, queries, qrels). Built lazily on first access.
`data/golden/` — generated golden sets used by the eval suite. Build with `make golden` or `python -m rag_evals.data.golden`.

Both subtrees are gitignored. See [`docs/corpus.md`](../docs/corpus.md) for the full corpus design.

`data/fixtures/` is committed: labeled synthetic QA with calibration/heldout splits, plus support-judge calibration. These are small logical fixtures, not human-panel judgments or SciFact reference answers.
