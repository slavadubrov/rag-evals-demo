import json

import pytest
from pydantic import ValidationError

from rag_evals.evaluation.faithfulness import faithfulness, llm_verify
from rag_evals.evaluation.filter_exclusion import rate_against_survivors
from rag_evals.evaluation.retrieval import evaluate_runs
from rag_evals.evaluation.runner import run_suite
from rag_evals.evaluation.schemas import Rating, Support
from rag_evals.generation.llm import LLM, InvalidResponse
from rag_evals.ingest.chunking import recursive_split, structural_split
from rag_evals.retrieval.hybrid_rrf import reciprocal_rank_fusion
from rag_evals.retrieval.iterative import retrieve_iteratively


def test_missing_output_and_document_duplicates():
    result = evaluate_runs({"a": ["d", "d"]}, {"a": ["d"], "b": ["e"]}, k=5)
    assert result.recall_at_k == 0.5
    assert (result.n_missing, result.n_attempted, result.n_scored) == (1, 1, 2)
    assert reciprocal_rank_fusion([["d", "d", "e"]]) == reciprocal_rank_fusion([["d", "e"]])


def test_chunk_bounds_and_preamble():
    text = "x" * 300
    chunks = recursive_split(text, target_tokens=7, overlap_tokens=0)
    assert "".join(chunks) == text and max(map(len, chunks)) <= 28
    assert structural_split("preamble\n# Heading\nbody")[0] == "preamble"
    assert max(map(len, structural_split("# Head\n" + text, target_tokens=7))) <= 28


def test_schema_rejects_invented_verdicts():
    for verdict in ["UNSUPPORTED", "NOT SUPPORTED", "SUPPORTED because yes"]:
        with pytest.raises(ValidationError):
            Support(evidence_quote="", explanation="", verdict=verdict)
    for score in [0, 6, True, "5"]:
        with pytest.raises(ValidationError):
            Rating(evidence_observation="", explanation="", score=score)
    llm = LLM(mode="mock")
    llm.ask = lambda *args, **kwargs: "not JSON"
    with pytest.raises(InvalidResponse):
        llm.structured("q", Support, system="s")


def test_no_claims_are_not_failure():
    result = faithfulness("I don't know.", "", use_heuristic=True)
    assert result.score is None and result.status == "not_applicable"


def test_support_requires_exact_evidence():
    class Judge:
        def structured(self, *args, **kwargs):
            return Support(evidence_quote="invented", explanation="", verdict="SUPPORTED")

    result = llm_verify(["claim"], "actual evidence", llm=Judge())[0]
    assert result.supported is None and result.status == "invalid"


def test_authorization_gold_and_override():
    row = {
        "qid": "q",
        "gold_doc_ids": ["private", "public"],
        "eligible_gold_doc_ids": ["public"],
        "authorization_predicate": {"tenant": "a"},
    }
    assert rate_against_survivors([row], lambda _: {"public"}).rate == 0
    row["eligible_gold_doc_ids"] = []
    assert rate_against_survivors([row], lambda _: set()).n_queries == 0
    row["eligible_gold_doc_ids"] = ["public"]
    row["filter_predicate"] = {"tenant": "b"}
    with pytest.raises(ValueError):
        rate_against_survivors([row], lambda _: set())


def test_offline_suite_and_invalid_cli_contract():
    result = run_suite(suite="offline")
    assert result["generation"]["replay"]
    assert result["generation"]["n_scored"] == 5
    assert all(g["pass"] for g in result["gates"])
    json.dumps(result, allow_nan=False)
    for kwargs in ({"suite": "typo"}, {"suite": "offline", "k": -1}):
        with pytest.raises(ValueError):
            run_suite(**kwargs)


def test_iterative_budgets_preserve_authorization():
    seen = []

    def retrieve(query, **kwargs):
        seen.append(kwargs["predicates"].copy())
        kwargs["predicates"]["tenant"] = "modified"
        return []

    result = retrieve_iteratively(
        ["a", "b", "c"], retrieve, predicates={"tenant": "original"}, max_calls=2
    )
    assert result.calls == 2 and result.stop_reason == "call_budget"
    assert seen == [{"tenant": "original"}] * 2


def test_rerankers_share_one_candidate_pool():
    from rag_evals.evaluation.reranking import compare_rerankers
    from rag_evals.types import RetrievalHit

    calls = []

    def retrieve(query, limit):
        calls.append(query)
        return [RetrievalHit("bad", 1.0), RetrievalHit("gold", 0.5)]

    result = compare_rerankers(
        [{"qid": "q", "query": "query", "gold_doc_ids": ["gold"]}],
        retrieve,
        {"reverse": lambda q, hits: hits[::-1]},
        k=1,
    )
    assert calls == ["query"]
    assert result["baseline"]["ndcg_at_k"] == 0
    assert result["reverse"]["ndcg_at_k"] == 1


def test_iterative_token_budget():
    result = retrieve_iteratively(["abc", "def"], lambda *a, **kw: [], max_tokens=3)
    assert result.calls == 1 and result.stop_reason == "token_budget"


def test_openai_adapter_request_and_failure_budget(monkeypatch):
    from types import SimpleNamespace

    import openai

    from rag_evals.config import settings

    captured = []

    class Client:
        def __init__(self, **kwargs):
            assert kwargs["max_retries"] == 0
            self.chat = SimpleNamespace(completions=self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def create(self, **kwargs):
            captured.append(kwargs)
            return SimpleNamespace(
                model="gpt-5.6-luna",
                usage=None,
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(
                            refusal=None,
                            content='{"evidence_quote":"text","explanation":"matches","verdict":"SUPPORTED"}',
                        ),
                    )
                ],
            )

    monkeypatch.setattr(openai, "OpenAI", Client)
    monkeypatch.setenv("OPENAI_API_KEY", "test-not-a-real-key")
    monkeypatch.setattr(settings, "llm_max_calls", 1)
    llm = LLM("gpt-5.6-luna", mode="live")
    assert llm.structured("q", Support, system="s").verdict == "SUPPORTED"
    assert captured[0]["model"] == "gpt-5.6-luna"
    assert "max_completion_tokens" in captured[0] and "max_tokens" not in captured[0]
    assert captured[0]["response_format"]["json_schema"]["strict"] is True
    with pytest.raises(RuntimeError, match="budget"):
        llm.ask("second call")


def test_invalid_completion_controls_fail_before_call():
    llm = LLM(mode="mock")
    with pytest.raises(ValueError, match="max_tokens"):
        llm.ask("q", max_tokens=0)
    with pytest.raises(ValueError, match="Temperature"):
        llm.ask("q", temperature=0.7)


def test_report_uses_actual_k():
    from dataclasses import asdict

    from rag_evals.evaluation.report import render_markdown

    report = render_markdown({"retrieval": asdict(evaluate_runs({"q": ["d"]}, {"q": ["d"]}, k=5))})
    assert "Recall@5" in report and "Recall@10" not in report
