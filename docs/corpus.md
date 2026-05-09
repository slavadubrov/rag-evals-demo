# Corpus

The demo runs against **BEIR scifact**, a 5,183-document scientific-claim verification corpus from [BEIR (Thakur et al., NeurIPS 2021)](https://arxiv.org/abs/2104.08663). It loads via `datasets` and is cached under `data/cache/`.

## Why scifact

- It's small. The whole corpus embeds in under a minute on CPU, and the full eval suite runs comfortably on a laptop.
- It comes with real qrels (relevance judgements for 300 test queries), so Recall / MRR / nDCG numbers actually mean something instead of measuring noise.
- It's domain-specific (biomedical claims), which exposes how a general-purpose embedding model behaves out of distribution — useful, since most production RAG systems run a general embedder over a niche corpus.

## Synthetic metadata

Every document is augmented with deterministic metadata in `src/rag_evals/data/metadata.py`:

- `tenant ∈ {acme, globex, initech}`
- `locale ∈ {en-US, en-GB, de-DE}`
- `domain ∈ {biomed, clinical, public-health}`

The bucket is a SHA-1 of `f"{salt}:{doc_id}"`, so the same doc id always lands in the same triple. No randomness across runs.

This metadata is what notebook 04 (filter false-exclusion) operates on. Three orthogonal dimensions with three buckets each split the corpus into roughly equal thirds along every axis, which gives a corrupted predicate a real chance of zeroing out gold-doc recall — exactly the failure the metric is designed to catch.

## Golden sets

`make golden` (or `python -m rag_evals.data.golden`) builds three JSONL files from scifact qrels:

- `data/golden/retrieval.jsonl`. `{qid, query, gold_doc_ids}`. Used by notebooks 01–03 and the retrieval suite.
- `data/golden/filter_aware.jsonl`. Adds `filter_predicate`. 30% of rows have a deliberately corrupted predicate so notebook 04 has a non-zero false-exclusion rate to detect.
- `data/golden/generation.jsonl`. Small generation eval set (~50 rows); `gold_answer` is the first 200 chars of the most-relevant doc.

The corruption rate is hard-coded in `data/golden.py`. Tune it there if you want to stress-test the harness more aggressively.

## Substituting another corpus

The data layer hides behind two iterators in `src/rag_evals/data/scifact.py`: `documents()` and `test_queries()`. Replace those with iterators over your own corpus and the rest of the pipeline keeps working — `metadata.synthesize` is keyed by `doc_id`, not by anything scifact-specific.

If your source corpus already ships with metadata, drop the call to `synthesize` in `data/scifact.py` and pass the real fields through.
