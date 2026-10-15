from __future__ import annotations

import pytest
import torch
from helpers import save_tiny_checkpoint

from exu import DecisionQuestion, DecisionRuntime, Option, StateTruncatedError


@pytest.fixture
def runtime(tmp_path):
    save_tiny_checkpoint(tmp_path / "checkpoint")
    return DecisionRuntime.load(tmp_path / "checkpoint", device="cpu")


def test_runtime_returns_a_distribution_and_two_confidences(runtime) -> None:
    question = DecisionQuestion.noul("Does this describe a failed payment?")

    decision = runtime.decide("the payment failed", question)

    assert decision.label in {"No", "Yes"}
    assert len(decision.probabilities) == 2
    assert sum(decision.probabilities) == pytest.approx(1.0)
    assert decision.confidence == pytest.approx(max(decision.probabilities))
    assert 0.0 <= decision.entropy_confidence <= 1.0
    assert decision.expected_level is None
    assert decision.kind == "noul"
    assert isinstance(decision.should_act, bool)


def test_runtime_reports_an_expected_level_for_ordinal_questions(runtime) -> None:
    question = DecisionQuestion.score("How urgent?", ["low", "normal", "high"])

    decision = runtime.decide("nothing is on fire", question)

    assert decision.expected_level is not None
    assert 0.0 <= decision.expected_level <= 2.0


def test_runtime_batches_distinct_questions(runtime) -> None:
    pairs = [
        ("payment failed", DecisionQuestion.noul("Is it fraud?")),
        (
            "cannot log in",
            DecisionQuestion.choice("Route", [Option("billing"), Option("support")]),
        ),
    ]

    decisions = runtime.decide_many(pairs)

    assert len(decisions) == 2
    assert decisions[0].kind == "noul"
    assert decisions[1].kind == "choice"


def test_runtime_rejects_an_empty_batch(runtime) -> None:
    with pytest.raises(ValueError, match="empty"):
        runtime.decide_many([])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_ordinal_question_on_cuda_does_not_mix_devices(tmp_path) -> None:
    """Regression: the level tensor was built on CPU, so `score` crashed on GPU.

    `DecisionRuntime.load` defaults to `device="auto"`, so this is the path a GPU
    user hits first. The rest of the suite runs on CPU, which is why it did not
    catch this, and CI has no GPU either: it only runs where one exists.
    """
    save_tiny_checkpoint(tmp_path / "checkpoint")
    runtime = DecisionRuntime.load(tmp_path / "checkpoint", device="cuda")
    question = DecisionQuestion.score("How urgent?", ["low", "normal", "high"])

    decision = runtime.decide("nothing is on fire", question)

    assert decision.expected_level is not None
    assert 0.0 <= decision.expected_level <= 2.0


LONG_STATE = " ".join(["payment failed again"] * 400)


def test_a_decision_says_how_much_of_its_state_was_read(runtime) -> None:
    question = DecisionQuestion.noul("Does this describe a failed payment?")

    short = runtime.decide("the payment failed", question).input_usage
    long = runtime.decide(LONG_STATE, question).input_usage

    assert not short.state_truncated
    assert short.state_tokens_used == short.state_tokens_original > 0
    assert long.state_truncated
    assert long.state_tokens_used < long.state_tokens_original
    assert long.option_count == 2


def test_strict_truncation_names_the_question_before_the_model_runs(tmp_path) -> None:
    save_tiny_checkpoint(tmp_path / "checkpoint")
    strict = DecisionRuntime.load(tmp_path / "checkpoint", device="cpu", truncation="error")
    question = DecisionQuestion.noul("Is it fraud?")

    def forbidden(**_inputs):
        raise AssertionError("the model ran on a refused batch")

    strict.model = forbidden
    with pytest.raises(StateTruncatedError) as raised:
        strict.decide_many([("payment failed", question), (LONG_STATE, question)])

    assert raised.value.index == 1
    assert raised.value.usage.state_truncated
    assert "index 1" in str(raised.value)


def test_strict_truncation_answers_like_the_default_when_nothing_is_cut(tmp_path) -> None:
    save_tiny_checkpoint(tmp_path / "checkpoint")
    allow = DecisionRuntime.load(tmp_path / "checkpoint", device="cpu")
    strict = DecisionRuntime.load(tmp_path / "checkpoint", device="cpu", truncation="error")
    pairs = [
        ("payment failed", DecisionQuestion.noul("Is it fraud?")),
        ("[MASK]", DecisionQuestion.noul("Is it fraud?")),
    ]

    assert strict.decide_many(pairs) == allow.decide_many(pairs)


def test_an_unknown_truncation_policy_is_refused(tmp_path) -> None:
    save_tiny_checkpoint(tmp_path / "checkpoint")

    with pytest.raises(ValueError, match="truncation"):
        DecisionRuntime.load(tmp_path / "checkpoint", device="cpu", truncation="shrink")
