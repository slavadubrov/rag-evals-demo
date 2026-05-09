"""Surface 'this number was computed on a mock LLM' warnings.

The MockBackend (``rag_evals.generation.llm.MockBackend``) returns canned
fixture replays or a deterministic ``[mock:...]`` stub. Anything those
responses are fed into — faithfulness scoring, LLM-as-judge, position-
stratified accuracy, pairwise win-rates — is **not a real model evaluation**.

This module provides:

- ``warn_mock_eval(context)`` — print a loud, one-shot stderr banner the
  first time a given ``context`` (e.g. ``"faithfulness.llm_verify"``) is
  triggered with a mock LLM. Idempotent within a process.
- ``MOCK_BANNER_TEXT`` — the canonical multi-line warning string, embedded
  verbatim into report.md / report.json / notebook outputs whenever any of
  the numbers in those reports were computed against a mock LLM.

Numeric reports must include this banner whenever any of their rows were
mock-derived. Per-row markers (e.g. ``[MOCK]`` prefixed to a model name) are
the responsibility of the report renderer.
"""

from __future__ import annotations

import os
import sys

__all__ = [
    "MOCK_BANNER_MD",
    "MOCK_BANNER_TEXT",
    "is_mock",
    "reset_mock_warnings",
    "warn_mock_eval",
]

_warned: set[str] = set()

MOCK_BANNER_TEXT = (
    "MOCK DATA — these numbers come from MockBackend (deterministic stub or "
    "fixture replay). They are NOT real model evaluations. Set the matching "
    "provider API key (or RAG_EVALS_BACKEND=live) to compute real metrics."
)

MOCK_BANNER_MD = (
    "> ⚠️  **MOCK DATA — NOT A REAL EVALUATION** ⚠️\n"
    ">\n"
    "> One or more rows below were computed with `MockBackend` (no live LLM).\n"
    "> Mock responses are deterministic stubs / fixture replays. Faithfulness,\n"
    "> pairwise win-rates, latency, and any other LLM-derived numbers are\n"
    "> meaningless until the run is repeated with a live API key.\n"
    ">\n"
    "> Rows tagged `[MOCK]` were produced (in whole or in part) by a mock LLM.\n"
)


def is_mock(llm) -> bool:
    """True if ``llm`` is in mock mode. Accepts ``None`` (treated as not-mock)."""
    return getattr(llm, "mode", None) == "mock"


def warn_mock_eval(context: str) -> None:
    """Print a one-shot stderr banner the first time ``context`` triggers a
    mock-eval path. Subsequent calls with the same ``context`` are silent.

    Set ``RAG_EVALS_SUPPRESS_MOCK_WARNING=1`` to silence (e.g. inside the
    test suite where mock mode is the contract).
    """
    if os.getenv("RAG_EVALS_SUPPRESS_MOCK_WARNING") == "1":
        return
    if context in _warned:
        return
    _warned.add(context)
    bar = "!" * 78
    msg = f"\n{bar}\n!!! {MOCK_BANNER_TEXT}\n!!! triggered by: {context}\n{bar}\n"
    print(msg, file=sys.stderr, flush=True)


def reset_mock_warnings() -> None:
    """Clear the dedup set. Useful at the start of a fresh benchmark run so
    the banner re-prints in long-running processes."""
    _warned.clear()
