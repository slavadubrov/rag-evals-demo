"""Chunkers.

Recursive splitter is the article's general-purpose default. Structural
splitter exploits headings/paragraphs when present.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from itertools import pairwise

from rag_evals.types import Chunk, Document


def _approx_token_count(text: str) -> int:
    return max(1, len(text) // 4)  # 4 chars/token rule of thumb


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
    target_chars = target_tokens * 4
    overlap_chars = overlap_tokens * 4

    if _approx_token_count(text) <= target_tokens:
        return [text]

    # 1. split on the highest-priority separator that exists in the text
    for sep in separators:
        if sep in text:
            parts = text.split(sep)
            break
    else:
        parts = [text]

    # 2. greedily merge into chunks <= target
    chunks: list[str] = []
    buf: list[str] = []
    buf_chars = 0
    for part in parts:
        part_chars = len(part)
        if buf_chars + part_chars > target_chars and buf:
            chunks.append(sep.join(buf))
            tail_chars = 0
            tail: list[str] = []
            for piece in reversed(buf):
                tail.insert(0, piece)
                tail_chars += len(piece)
                if tail_chars >= overlap_chars:
                    break
            buf = tail[:]
            buf_chars = sum(len(p) for p in buf)
        buf.append(part)
        buf_chars += part_chars
    if buf:
        chunks.append(sep.join(buf))
    return [c.strip() for c in chunks if c.strip()]


_HEADING_RE = re.compile(r"(?m)^(#{1,6}\s+.+|[A-Z][A-Z0-9 \-]{4,}$)")


def structural_split(text: str, target_tokens: int = 256) -> list[str]:
    """Split on Markdown headings and ALL-CAPS lines, then recurse if a
    section is larger than ``target_tokens``.
    """
    boundaries = [m.start() for m in _HEADING_RE.finditer(text)]
    if not boundaries:
        return recursive_split(text, target_tokens=target_tokens)
    boundaries.append(len(text))
    sections = [text[a:b].strip() for a, b in pairwise(boundaries)]
    out: list[str] = []
    for section in sections:
        if not section:
            continue
        if _approx_token_count(section) > target_tokens:
            out.extend(recursive_split(section, target_tokens=target_tokens))
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
    splitter = recursive_split if strategy == "recursive" else structural_split
    out: list[Chunk] = []
    for doc in docs:
        text = (doc.title + "\n\n" + doc.text).strip() if doc.title else doc.text
        if strategy == "recursive":
            pieces = splitter(text, target_tokens=target_tokens, overlap_tokens=overlap_tokens)
        else:
            pieces = splitter(text, target_tokens=target_tokens)
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
