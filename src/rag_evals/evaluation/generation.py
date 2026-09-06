"""Generation harness over explicit evidence and labeled QA fixtures.

Fixed-context evaluation isolates generation; it does not measure retrieval quality.
SciFact retrieval evaluation is a separate suite with real qrels.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from functools import partial
from pathlib import Path
from statistics import fmean

from rag_evals.config import ROOT
from rag_evals.evaluation.answer import abstains, deterministic_metrics
from rag_evals.evaluation.context import citation_support, coverage
from rag_evals.evaluation.faithfulness import faithfulness, llm_verify
from rag_evals.evaluation.llm_judge import pointwise
from rag_evals.evaluation.provenance import provenance
from rag_evals.evaluation.statistics import bootstrap_mean
from rag_evals.generation.llm import LLM
from rag_evals.generation.prompts import format_context
from rag_evals.generation.rag import run_rag
from rag_evals.types import RetrievalHit

FIXTURES = ROOT / "data" / "fixtures"


def load_cases(path: Path, split: str | None = None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len({r.get("qid", i) for i, r in enumerate(rows)}) != len(rows):
        raise ValueError("Duplicate query IDs")
    return [r for r in rows if split is None or r["split"] == split]


def validate_cases(rows: list[dict]) -> None:
    """Reject ambiguous QA labels before generating or spending API calls."""
    for row in rows:
        required = {
            "qid",
            "split",
            "query",
            "context",
            "reference_answers",
            "nuggets",
            "answerable",
            "fixture_answer",
            "gold_doc_ids",
        }
        if not required <= row.keys():
            raise ValueError("Missing QA fields")
        if type(row["answerable"]) is not bool or row["split"] not in {"calibration", "heldout"}:
            raise ValueError("Invalid QA split or answerability label")
        ids = [h["doc_id"] for h in row["context"]]
        if len(set(ids)) != len(ids) or not set(row["gold_doc_ids"]) <= set(ids):
            raise ValueError("Duplicate context IDs or unavailable gold evidence")
        if row["answerable"] and not (
            row["reference_answers"] and row["nuggets"] and row["gold_doc_ids"]
        ):
            raise ValueError("Answerable cases require references, nuggets and gold evidence")
        if any(not aliases or any(not a.strip() for a in aliases) for aliases in row["nuggets"]):
            raise ValueError("Empty nugget aliases")


def calibrate(judge: LLM) -> dict:
    rows = load_cases(FIXTURES / "calibration.jsonl")
    scored = []
    for row in rows:
        verdict = llm_verify([row["claim"]], row["context"], llm=judge)[0]
        scored.append({**row, "predicted": verdict.supported, "status": verdict.status})
    valid = [r for r in scored if r["predicted"] is not None]
    return {
        "n_attempted": len(rows),
        "n_invalid": len(rows) - len(valid),
        "accuracy_valid": fmean(r["predicted"] == r["supported"] for r in valid) if valid else None,
        "agreement_all_attempts": sum(r["predicted"] == r["supported"] for r in valid) / len(rows),
        "label_source": "hand-authored logical fixtures; not a human-panel study",
        "rows": scored,
    }


def run_generation(
    *,
    generator: LLM,
    judge: LLM,
    path: Path = FIXTURES / "generation.jsonl",
    split: str = "heldout",
    limit: int | None = None,
    replay: bool = False,
    sgr: bool = True,
) -> dict:
    all_rows = load_cases(path)
    validate_cases(all_rows)
    rows = [r for r in all_rows if r["split"] == split]
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        rows = rows[:limit]
    if not rows:
        raise ValueError("No generation cases selected")
    out = []
    for row in rows:
        item: dict = {"qid": row["qid"], "status": "error", "answerable": row["answerable"]}
        out.append(item)
        try:
            hits = [RetrievalHit(score=0.0, **h) for h in row["context"]]
            if replay:
                answer, duration = row["fixture_answer"], None
            else:
                rag = run_rag(
                    row["qid"], row["query"], partial(_fixed_context, hits), llm=generator, sgr=sgr
                )
                answer, duration = rag.answer, rag.latency_ms
            item.update(
                answer=answer,
                latency_ms=duration,
                status="ok",
                abstained=abstains(answer),
                deterministic=deterministic_metrics(answer, row),
            )
            if not replay:
                hits = rag.context
                ctx = format_context(hits)
                item["nugget_semantic_coverage"] = coverage(answer, row["nuggets"], judge)
                item["context_nugget_recall"] = coverage(ctx, row["nuggets"], judge)
                item["citation_set_support"] = citation_support(
                    answer, [{"doc_id": h.doc_id, "text": h.text} for h in hits], judge
                )
                item["faithfulness"] = asdict(faithfulness(answer, ctx, llm=judge))
                item["relevance"] = asdict(
                    pointwise(
                        row["query"],
                        answer,
                        context=ctx,
                        criterion="answer relevance to the question, including appropriate abstention",
                        llm=judge,
                    )
                )
                item["correctness"] = asdict(
                    pointwise(
                        row["query"],
                        answer,
                        context=ctx,
                        reference=json.dumps(
                            {"answers": row["reference_answers"], "answerable": row["answerable"]}
                        ),
                        criterion="correctness and completeness against reference and evidence; abstain if insufficient or conflicting",
                        llm=judge,
                    )
                )
        except Exception as exc:
            item.update(status="error", error_type=type(exc).__name__)
    valid = [r for r in out if r["status"] == "ok"]
    metrics = {}
    for key in deterministic_metrics(rows[0]["fixture_answer"], rows[0]):
        values = [r["deterministic"][key] for r in valid if r["deterministic"][key] is not None]
        metrics[key] = {
            "mean": fmean(values) if values else None,
            "n_scored": len(values),
            "ci95": bootstrap_mean(values) if values else None,
        }
    for key in (
        "faithfulness",
        "relevance",
        "correctness",
        "nugget_semantic_coverage",
        "context_nugget_recall",
        "citation_set_support",
    ):
        values = [r[key]["score"] for r in valid if key in r and r[key]["score"] is not None]
        metrics[key] = {
            "mean": fmean(values) if values else None,
            "n_scored": len(values),
            "n_invalid": sum(r.get(key, {}).get("status") == "invalid" for r in valid),
            "n_not_applicable": sum(
                r.get(key, {}).get("status") == "not_applicable" for r in valid
            ),
            "ci95": bootstrap_mean(values) if values else None,
        }
    return {
        "provenance": provenance(rows),
        "population": "fixed-context synthetic QA",
        "split": split,
        "replay": replay,
        "generation_mode": "sgr" if sgr else "text",
        "generator": generator.model,
        "judge": judge.model,
        "same_provider_judge": True,
        "is_mock": not replay and (generator.mode == "mock" or judge.mode == "mock"),
        "n_attempted": len(rows),
        "n_scored": len(valid),
        "n_failures": len(rows) - len(valid),
        "metrics": metrics,
        "rows": out,
        "calls": {"generator": generator.calls, "judge": judge.calls},
    }


def _fixed_context(hits: list[RetrievalHit], query: str) -> list[RetrievalHit]:
    return hits
