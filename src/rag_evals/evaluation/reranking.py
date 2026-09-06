"""Compare rerankers on one materialized candidate pool per query."""

from collections.abc import Callable, Sequence
from dataclasses import asdict
from typing import Any

from rag_evals.evaluation.retrieval import evaluate_runs
from rag_evals.types import RetrievalHit


def compare_rerankers(
    rows: Sequence[dict[str, Any]],
    retrieve: Callable[..., list[RetrievalHit]],
    rerankers: dict[str, Callable[..., list[RetrievalHit]]],
    *,
    candidate_limit: int = 30,
    k: int = 10,
) -> dict[str, Any]:
    if candidate_limit < k or k <= 0:
        raise ValueError("Require candidate_limit >= k > 0")
    pools = {r["qid"]: retrieve(r["query"], limit=candidate_limit) for r in rows}
    gold = {r["qid"]: r["gold_doc_ids"] for r in rows}
    runs = {"baseline": {qid: [h.doc_id for h in hits] for qid, hits in pools.items()}}
    for name, rerank in rerankers.items():
        if name == "baseline":
            raise ValueError("baseline is a reserved name")
        runs[name] = {
            r["qid"]: [h.doc_id for h in rerank(r["query"], list(pools[r["qid"]]))] for r in rows
        }
    return {name: asdict(evaluate_runs(run, gold, k=k)) for name, run in runs.items()}
