"""Chunkers.

Recursive splitter is the article's general-purpose default. Structural
splitter exploits headings/paragraphs when present.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from itertools import pairwise

from rag_evals.types import Chunk, Document


def recursive_split(
    text: str,
    *,
    target_tokens: int = 256,
    overlap_tokens: int = 32,
    separators: tuple[str, ...] = ("\n\n", "\n", ". ", " "),
) -> list[str]:
    """Article default: 400-512 tokens recursive with ~10-20% overlap.

    Returns a list of chunk strings. Greedy: walks separators in priority
    order, splits the longest pieces, then merges adjacent fragments back
    up to ``target_tokens``.
    """
    if target_tokens <= 0 or not 0 <= overlap_tokens < target_tokens:
        raise ValueError("Require target_tokens > 0 and 0 <= overlap_tokens < target_tokens")
    if any(not sep for sep in separators):
        raise ValueError("Separators must be nonempty")
    # ponytail: character budget approximates tokens; use a tokenizer for hard model limits.
    target_chars, overlap_chars = target_tokens * 4, overlap_tokens * 4
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + target_chars, len(text))
        if end < len(text):
            for sep in separators:
                cut = text.rfind(sep, start + max(overlap_chars + 1, target_chars // 2), end)
                if cut >= 0:
                    end = cut + len(sep)
                    break
        if piece := text[start:end].strip():
            chunks.append(piece)
        if end == len(text):
            break
        start = end - overlap_chars
    return chunks


_HEADING_RE = re.compile(r"(?m)^(#{1,6}\s+.+|[A-Z][A-Z0-9 \-]{4,}$)")


def structural_split(text: str, target_tokens: int = 256) -> list[str]:
    """Split on Markdown headings and ALL-CAPS lines, then recurse if a
    section is larger than ``target_tokens``.
    """
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    boundaries = [m.start() for m in _HEADING_RE.finditer(text)]
    if not boundaries:
        return recursive_split(text, target_tokens=target_tokens, overlap_tokens=0)
    boundaries = sorted({0, *boundaries, len(text)})
    sections = [text[a:b].strip() for a, b in pairwise(boundaries)]
    out: list[str] = []
    for section in sections:
        if not section:
            continue
        if len(section) > 4 * target_tokens:
            out.extend(recursive_split(section, target_tokens=target_tokens, overlap_tokens=0))
        else:
            out.append(section)
    return out


def chunk_documents(
    docs: Iterable[Document],
    *,
    strategy: str = "recursive",
    target_tokens: int = 256,
    overlap_tokens: int = 32,
) -> list[Chunk]:
    if strategy not in {"recursive", "structural"}:
        raise ValueError(f"Unknown chunk strategy: {strategy}")
    out: list[Chunk] = []
    for doc in docs:
        text = (doc.title + "\n\n" + doc.text).strip() if doc.title else doc.text
        if strategy == "recursive":
            pieces = recursive_split(
                text, target_tokens=target_tokens, overlap_tokens=overlap_tokens
            )
        else:
            pieces = structural_split(text, target_tokens=target_tokens)
        for i, piece in enumerate(pieces):
            out.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}::{i}",
                    doc_id=doc.doc_id,
                    text=piece,
                    position=i,
                    metadata=doc.metadata,
                )
            )
    return out
