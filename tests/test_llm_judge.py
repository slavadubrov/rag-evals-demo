from __future__ import annotations

from rag_evals.evaluation.llm_judge import cross_family_judges
from rag_evals.generation.models import Model


def test_cross_family_excludes_same_family() -> None:
    judges = cross_family_judges(Model.GPT_5_MINI)
    assert Model.GPT_5_MINI not in judges
    assert Model.GPT_5 not in judges
    assert Model.CLAUDE_HAIKU_4_5 in judges


def test_cross_family_for_anthropic() -> None:
    judges = cross_family_judges(Model.CLAUDE_SONNET_4_6)
    assert all(not j.value.startswith("claude-") for j in judges)


def test_cross_family_for_gemini() -> None:
    judges = cross_family_judges(Model.GEMINI_3_FLASH)
    assert all(not j.value.startswith("gemini/") for j in judges)
