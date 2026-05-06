# Corpus

The demo runs against **BEIR scifact** — a 5,183-document scientific-claim verification corpus from [BEIR (Thakur et al., NeurIPS 2021)](https://arxiv.org/abs/2104.08663). Loaded via `datasets` and cached under `data/cache/`.

## Why scifact

- Small. Embeds in <1 minute on CPU; the full eval suite runs on a laptop.
- Has real qrels — relevance judgements for 300 test queries — so Recall / MRR / nDCG numbers actually mean something.
- Domain-specific (biomedical claims), which exposes how a general-purpose embedding model behaves out of distribution.

## Synthetic metadata

Every document is augmented with deterministic metadata in `src/rag_evals/data/metadata.py`:

- `tenant ∈ {acme, globex, initech}`
- `locale ∈ {en-US, en-GB, de-DE}`
- `domain ∈ {biomed, clinical, public-health}`

The bucket is a SHA-1 of `f"{salt}:{doc_id}"`, so the same doc id always lands in the same triple. No randomness across runs.

This metadata is what notebook 04 (filter false-exclusion) operates on. With three orthogonal dimensions and three buckets each, every dimension splits the corpus into roughly equal thirds — a corrupted predicate has a real chance of zeroing out gold doc recall.

## Golden sets

`make golden` (or `python -m rag_evals.data.golden`) builds three JSONL files from scifact qrels:

- `data/golden/retrieval.jsonl` — `{qid, query, gold_doc_ids}`. Used by notebooks 01–03 and the retrieval suite.
- `data/golden/filter_aware.jsonl` — adds `filter_predicate`. 30% of rows have a deliberately corrupted predicate so notebook 04 has a non-zero false-exclusion rate to detect.
- `data/golden/generation.jsonl` — small generation eval set (~50 rows); `gold_answer` is the first 200 chars of the most-relevant doc.

The corruption rate is hard-coded in `data/golden.py`. Tune it there if you want to stress-test the harness more aggressively.

## Substituting another corpus

The data layer hides behind two iterators in `src/rag_evals/data/scifact.py`: `documents()` and `test_queries()`. Replace those with iterators over your own corpus and the rest of the pipeline keeps working — `metadata.synthesize` is keyed by `doc_id`, not by anything scifact-specific.

If you ship your own metadata in the source corpus, drop the call to `synthesize` in `data/scifact.py` and pass through the real fields.
