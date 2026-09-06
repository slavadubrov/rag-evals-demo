"""Reference-based lexical diagnostics; semantic quality is evaluated separately."""

from __future__ import annotations

import re
from collections import Counter

from rag_evals.generation.rag import extract_citations


def normalize(text: str) -> str:
    text = re.sub(r"\[[^\]]*\]", "", text.lower())
    return " ".join(re.findall(r"\w+", text))


def abstains(answer: str) -> bool:
    return bool(re.search(r"\bi don['\u2019]t know\b|\binsufficient evidence\b", answer, re.I))


def token_f1(answer: str, reference: str) -> float:
    a, b = Counter(normalize(answer).split()), Counter(normalize(reference).split())
    overlap = sum((a & b).values())
    return 2 * overlap / (a.total() + b.total()) if a or b else 1.0


def deterministic_metrics(answer: str, row: dict) -> dict[str, float | None]:
    refs = row["reference_answers"]
    cited = set(extract_citations(answer))
    context_ids = {h["doc_id"] for h in row["context"]}
    gold = set(row["gold_doc_ids"])
    nuggets = row["nuggets"]
    normalized = f" {normalize(answer)} "
    return {
        "exact_match": float(any(normalize(answer) == normalize(r) for r in refs))
        if refs
        else None,
        "token_f1": max((token_f1(answer, r) for r in refs), default=None),
        "nugget_lexical_coverage": sum(
            any(f" {normalize(a)} " in normalized for a in aliases) for aliases in nuggets
        )
        / len(nuggets)
        if nuggets
        else None,
        "abstention_correct": float(abstains(answer) == (not row["answerable"])),
        "citation_id_validity": len(cited & context_ids) / len(cited) if cited else None,
        "citation_gold_recall": len(cited & gold) / len(gold) if gold else None,
    }
