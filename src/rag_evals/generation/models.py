"""Supported LLM models, routed through LiteLLM.

Adding a model is one enum line. The judge notebook (07) explicitly uses
*different families* for generator and judge to demonstrate the article's
'never use a model to judge itself' rule.

Verified current as of May 2026:
- gpt-5-mini, gpt-5: OpenAI
- claude-haiku-4-5, claude-sonnet-4-6: Anthropic (Haiku 4.5 released Oct 2025)
- gemini-3-flash, gemini-3-flash-lite: Google (current Flash tier in 2026)
"""

from __future__ import annotations

from enum import StrEnum


class Model(StrEnum):
    GPT_5_MINI = "gpt-5-mini"
    GPT_5 = "gpt-5"
    CLAUDE_HAIKU_4_5 = "claude-haiku-4-5"
    CLAUDE_SONNET_4_6 = "claude-sonnet-4-6"
    GEMINI_3_FLASH = "gemini/gemini-3-flash"
    GEMINI_3_FLASH_LITE = "gemini/gemini-3-flash-lite"
    MOCK = "mock"


def required_env_var(model: Model) -> str | None:
    """Env var that must be set for ``model`` to run live.

    None means no env var is required (e.g. MOCK).
    """
    name = model.value
    if name.startswith("gpt-"):
        return "OPENAI_API_KEY"
    if name.startswith("claude-"):
        return "ANTHROPIC_API_KEY"
    if name.startswith("gemini/"):
        return "GEMINI_API_KEY"
    return None
