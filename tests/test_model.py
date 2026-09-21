from __future__ import annotations

import math

import pytest
import torch
from helpers import tiny_encoder, tiny_model, tiny_tokenizer

from exu import DecisionQuestion, Option
from exu.model import ActionCosts, ExuConfig, ExuModel, masked_log_softmax, masked_softmax
from exu.sequence import SequenceBuilder


@pytest.fixture
def builder():
    return SequenceBuilder(tiny_tokenizer())


def _batch(builder, pairs):
    encoded = [builder.build(state, question) for state, question in pairs]
    raw = builder.pad(encoded)
    return {name: torch.tensor(value) for name, value in raw.items()}


def test_model_scores_only_real_options(builder) -> None:
    two = DecisionQuestion.noul("Is it fraud?")
    three = DecisionQuestion.choice("Route", [Option("A"), Option("B"), Option("C")])
    batch = _batch(builder, [("payment failed", two), ("payment failed", three)])
    model = tiny_model()

    output = model(**batch)

    assert output.probabilities.shape == (2, 3)
    assert output.probabilities[0, 2].item() == 0
    assert torch.allclose(output.probabilities.sum(dim=-1), torch.ones(2))
    assert output.action_logits.shape == (2, 2)


def test_temperature_override_sharpens_the_distribution(builder) -> None:
    question = DecisionQuestion.choice("Route", [Option("A"), Option("B"), Option("C")])
    batch = _batch(builder, [("state", question)])
    model = tiny_model()

    warm = model(**batch, temperature=4.0).probabilities
    cold = model(**batch, temperature=0.25).probabilities

    assert cold.max().item() > warm.max().item()


def test_per_row_temperatures_are_supported(builder) -> None:
    question = DecisionQuestion.choice("Route", [Option("A"), Option("B"), Option("C")])
    batch = _batch(builder, [("state", question), ("state", question)])
    model = tiny_model()

    output = model(**batch, temperature=torch.tensor([1.0, 3.0]))

    assert output.probabilities.shape == (2, 3)
    assert torch.allclose(output.probabilities.sum(dim=-1), torch.ones(2))


def test_masked_softmax_zeroes_padded_options() -> None:
    logits = torch.tensor([[1.0, 1.0, 1.0]])
    mask = torch.tensor([[True, True, False]])

    probabilities = masked_softmax(logits, mask)

    assert probabilities[0, 2].item() == 0
    assert probabilities[0, :2].sum().item() == pytest.approx(1.0)


def test_masked_softmax_is_stable_with_dtype_minimum_padding() -> None:
    # Padded logits look like this after the model's own masking. Dividing them
    # by a temperature below one used to overflow and produce NaN gradients.
    logits = torch.tensor([[1.0, 0.5, torch.finfo(torch.float32).min]])
    mask = torch.tensor([[True, True, False]])
    temperature = torch.tensor(0.5, requires_grad=True)

    probabilities = masked_softmax(logits, mask, temperature)
    probabilities[0, 0].backward()

    assert torch.isfinite(probabilities).all()
    assert probabilities[0, 2].item() == 0
    assert torch.isfinite(temperature.grad).all()


def test_masked_log_softmax_matches_the_log_of_the_probabilities() -> None:
    logits = torch.tensor([[1.0, -0.5, 3.0], [0.2, 0.1, 7.0]])
    mask = torch.tensor([[True, True, False], [True, True, True]])
    temperature = torch.tensor([0.5, 2.0])

    log_probabilities = masked_log_softmax(logits, mask, temperature)
    probabilities = masked_softmax(logits, mask, temperature)

    assert torch.allclose(log_probabilities[mask], probabilities[mask].log(), atol=1e-6)
    assert log_probabilities[0, 2].item() == 0


def test_masked_log_softmax_survives_what_softmax_underflows() -> None:
    logits = torch.tensor([[-5000.0, 0.0]], requires_grad=True)
    mask = torch.ones(1, 2, dtype=torch.bool)

    log_probabilities = masked_log_softmax(logits, mask)
    log_probabilities[0, 0].backward()

    assert masked_softmax(logits, mask)[0, 0].item() == 0
    assert log_probabilities[0, 0].item() == pytest.approx(-5000.0)
    assert torch.allclose(logits.grad, torch.tensor([[1.0, -1.0]]))


def test_masked_log_softmax_is_stable_with_dtype_minimum_padding() -> None:
    logits = torch.tensor([[1.0, 0.5, torch.finfo(torch.float32).min]])
    mask = torch.tensor([[True, True, False]])
    temperature = torch.tensor(0.5, requires_grad=True)

    log_probabilities = masked_log_softmax(logits, mask, temperature)
    log_probabilities[0, 0].backward()

    assert torch.isfinite(log_probabilities).all()
    assert torch.isfinite(temperature.grad).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_masked_log_softmax_computes_in_float32_whatever_arrives(dtype: torch.dtype) -> None:
    # Half precision cannot hold a confident miss. The upcast must not depend on
    # the caller remembering an autocast context.
    logits = torch.tensor([[0.0, 30.0, torch.finfo(dtype).min]], dtype=dtype)
    mask = torch.tensor([[True, True, False]])

    log_probabilities = masked_log_softmax(logits, mask)

    assert log_probabilities.dtype == torch.float32
    assert log_probabilities[0, 0].item() == pytest.approx(-30.0)


def test_masked_log_softmax_rejects_a_row_without_options() -> None:
    with pytest.raises(ValueError, match="valid option"):
        masked_log_softmax(torch.zeros(2, 2), torch.tensor([[True, True], [False, False]]))


def test_masked_log_softmax_does_not_hide_a_broken_logit() -> None:
    # A NaN or infinite logit is a bug upstream. It has to stay visible.
    logits = torch.tensor([[0.0, math.nan], [0.0, math.inf]])
    mask = torch.ones(2, 2, dtype=torch.bool)

    assert not torch.isfinite(masked_log_softmax(logits, mask)).all(dim=-1).any()


def test_action_cost_threshold_matches_the_business_rule() -> None:
    costs = ActionCosts(gain=1.0, loss=3.0, escalate=0.5)

    assert costs.threshold == pytest.approx(0.625)
    assert bool(costs.should_act(0.7)) is True
    assert bool(costs.should_act(0.5)) is False
    assert costs.should_act(torch.tensor([0.5, 0.7])).tolist() == [False, True]


def test_action_costs_reject_impossible_values() -> None:
    with pytest.raises(ValueError, match="gain"):
        ActionCosts(gain=0.0)
    with pytest.raises(ValueError, match="loss"):
        ActionCosts(loss=-1.0)


def test_config_round_trips_through_a_dict() -> None:
    config = ExuConfig(encoder_name="tiny", num_decision_layers=3, temperature=2.0)

    restored = ExuConfig.from_dict(config.to_dict())

    assert restored == config
    assert restored.action_costs.threshold == pytest.approx(0.625)


def test_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="num_decision_layers"):
        ExuConfig(num_decision_layers=0)
    with pytest.raises(ValueError, match="temperature"):
        ExuConfig(temperature=0.0)
    with pytest.raises(ValueError, match="dropout"):
        ExuConfig(dropout=1.0)


def test_model_rejects_a_hidden_size_without_room_for_heads() -> None:
    with pytest.raises(ValueError, match="hidden size"):
        ExuModel(tiny_encoder(hidden_size=65), ExuConfig(num_decision_layers=1))
