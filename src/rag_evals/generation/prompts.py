"""Prompt templates."""

from __future__ import annotations

from rag_evals.types import RetrievalHit

RAG_SYSTEM = """You answer questions strictly from the provided context.
- Cite sources by their bracketed id, e.g. [d3].
- If the context does not contain the answer, say "I don't know."
- Be concise: 1-3 sentences."""


def format_context(hits: list[RetrievalHit]) -> str:
    lines = []
    for h in hits:
        lines.append(f"[{h.doc_id}] {h.text or ''}")
    return "\n\n".join(lines)


def rag_user_prompt(query: str, hits: list[RetrievalHit]) -> str:
    return f"Context:\n{format_context(hits)}\n\nQuestion: {query}\n\nAnswer:"
