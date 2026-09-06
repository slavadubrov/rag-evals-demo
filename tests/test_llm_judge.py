from rag_evals.evaluation.llm_judge import alternate_judges, measure_position_bias, pointwise
from rag_evals.evaluation.schemas import Preference
from rag_evals.generation.llm import LLM
from rag_evals.generation.models import Model


def test_alternate_models_are_explicitly_same_provider():
    assert alternate_judges(Model.GPT_5_6_LUNA) == [Model.GPT_5_6_TERRA, Model.GPT_6_ASTRA]


def test_unrecorded_mock_is_invalid():
    result = pointwise("q", "a", context="evidence", llm=LLM(mode="mock"))
    assert result.score is None and result.status == "invalid"


def test_swap_maps_candidate_identity():
    class Judge:
        def __init__(self, choices):
            self.choices = iter(choices)

        def structured(self, *args, **kwargs):
            return Preference(evidence_observation="", explanation="", winner=next(self.choices))

    consistent = measure_position_bias([("q", "good", "bad")], llm=Judge(["A", "B"]))
    assert consistent.swap_consistency == 1 and consistent.position_bias == 0
    biased = measure_position_bias([("q", "good", "bad")], llm=Judge(["A", "A"]))
    assert biased.swap_consistency == 0 and biased.position_bias == 0.5
