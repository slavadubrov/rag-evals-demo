"""Position-stratified eval to surface the lost-in-the-middle effect.

Take a query whose gold chunk is known. Pad the context with N distractor
chunks (drawn from the corpus excluding the gold). Place the gold at three
positions: 0, mid, last. Generate an answer per arrangement and score
correctness. The article cites Liu et al. (TACL 2023) on the U-shape.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rag_evals._mock_warning import is_mock, warn_mock_eval
from rag_evals.generation.llm import LLM
from rag_evals.generation.prompts import RAG_SYSTEM, rag_user_prompt
from rag_evals.types import RetrievalHit


@dataclass
class PositionRun:
    position: str  # 'first' | 'middle' | 'last'
    answer: str
    correct: bool


@dataclass
class PositionEvalResult:
    runs: list[PositionRun]
    accuracy_by_position: dict[str, float]


def _arrange(
    gold: RetrievalHit, distractors: Sequence[RetrievalHit], position: str
) -> list[RetrievalHit]:
    if position not in {"first", "middle", "last"}:
        raise ValueError("Unknown context position")
    n = len(distractors)
    if position == "first":
        return [gold, *distractors]
    if position == "last":
        return [*distractors, gold]
    mid = n // 2
    return [*distractors[:mid], gold, *distractors[mid:]]


def position_stratified_eval(
    query: str,
    gold_chunk: RetrievalHit,
    distractors: Sequence[RetrievalHit],
    *,
    is_correct,
    llm: LLM | None = None,
    positions: Sequence[str] = ("first", "middle", "last"),
) -> PositionEvalResult:
    llm = llm or LLM()
    if is_mock(llm):
        warn_mock_eval("lost_in_middle.position_stratified_eval")
    runs: list[PositionRun] = []
    by_pos: dict[str, list[bool]] = {p: [] for p in positions}
    for pos in positions:
        ctx = _arrange(gold_chunk, distractors, pos)
        answer = llm.ask(rag_user_prompt(query, ctx), system=RAG_SYSTEM)
        ok = bool(is_correct(answer))
        runs.append(PositionRun(position=pos, answer=answer, correct=ok))
        by_pos[pos].append(ok)
    accuracy = {p: (sum(v) / len(v) if v else 0.0) for p, v in by_pos.items()}
    return PositionEvalResult(runs=runs, accuracy_by_position=accuracy)
