from __future__ import annotations

from rag_evals.evaluation.faithfulness import faithfulness, heuristic_verify


def test_heuristic_verify_supports_grounded_claims() -> None:
    context = "Mars has two moons, Phobos and Deimos. Curiosity landed on Mars in 2012."
    claims = ["Mars has two moons", "Curiosity landed on Mars"]
    verdicts = heuristic_verify(claims, context)
    assert all(v.supported for v in verdicts)


def test_heuristic_verify_rejects_unsupported_claims() -> None:
    context = "Mars has two moons, Phobos and Deimos."
    claims = ["Mars has thick atmosphere with oxygen"]
    verdicts = heuristic_verify(claims, context)
    assert not verdicts[0].supported


def test_faithfulness_score_uses_heuristic_when_requested() -> None:
    context = "The capital of France is Paris."
    answer = "Paris is the capital of France. Berlin is the capital of Spain."
    result = faithfulness(answer, context, use_heuristic=True)
    # First sentence supported, second is not.
    assert 0.0 < result.score < 1.0
