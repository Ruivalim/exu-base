from __future__ import annotations

import math

import pytest
import torch
from helpers import tiny_encoder, tiny_model, tiny_tokenizer

from exu import DecisionQuestion, Option
from exu.model import (
    ActionCosts,
    ExuConfig,
    ExuModel,
    centre_options,
    masked_log_softmax,
    masked_softmax,
    option_text_embeddings,
)
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


def _cls_model(hidden_size: int = 64) -> ExuModel:
    config = ExuConfig(
        encoder_name="tiny-bert", num_decision_layers=1, dropout=0.0, scorer="marker-cls"
    )
    return ExuModel(tiny_encoder(hidden_size=hidden_size), config)


def test_the_default_scorer_adds_no_weights() -> None:
    keys = tiny_model().state_dict()

    assert tiny_model().config.scorer == "marker"
    assert not any(key.startswith(("option_readout", "summary_norm")) for key in keys)
    assert any(key.startswith("option_readout") for key in _cls_model().state_dict())


def test_an_unknown_scorer_is_refused() -> None:
    with pytest.raises(ValueError, match="scorer"):
        ExuConfig(scorer="cls")


def test_the_scorer_survives_a_config_round_trip() -> None:
    assert ExuConfig.from_dict(ExuConfig(scorer="marker-cls").to_dict()).scorer == "marker-cls"
    # A checkpoint written before the option existed has no such key.
    assert ExuConfig.from_dict({"encoder_name": "x"}).scorer == "marker"


def test_option_text_embeddings_pool_exactly_the_tokens_of_each_option(builder) -> None:
    tokenizer = tiny_tokenizer()
    two = DecisionQuestion.noul("is it urgent", "no", "yes")
    three = DecisionQuestion.choice(
        "where should this ticket go",
        [
            Option("billing", "payment invoice or refund"),
            Option("support", "access or outage"),
            Option("security"),
        ],
    )
    pairs = [("payment failed twice today", two), ("the login page is down", three)]
    batch = _batch(builder, pairs)
    table = torch.randn(tokenizer.vocab_size, 8, generator=torch.Generator().manual_seed(0))

    pooled = option_text_embeddings(
        table[batch["input_ids"]], batch["input_ids"], batch["marker_positions"]
    )

    assert pooled.shape == (2, 3, 8)
    for row, (_state, question) in enumerate(pairs):
        for index, option in enumerate(question.options):
            ids = tokenizer.encode(option.render(), add_special_tokens=False)
            assert torch.allclose(pooled[row, index], table[ids].mean(dim=0), atol=1e-6)


def test_centring_removes_what_the_options_share() -> None:
    # Options of one question often share most of their words. Whatever is common
    # to all of them cancels in the softmax, so only the difference may get through.
    generator = torch.Generator().manual_seed(1)
    identity = torch.randn(2, 3, 8, generator=generator)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    shared = torch.randn(2, 1, 8, generator=generator) * 50

    plain = centre_options(identity, mask)
    shifted = centre_options(identity + shared, mask)

    assert torch.allclose(plain[mask], shifted[mask], atol=1e-4)
    assert torch.allclose(plain[mask].mean(dim=-1), torch.zeros(5), atol=1e-5)
    assert torch.isfinite(plain).all()


def test_centring_ignores_padded_options() -> None:
    identity = torch.randn(1, 3, 8, generator=torch.Generator().manual_seed(2))
    mask = torch.tensor([[True, True, False]])
    noisy = identity.clone()
    noisy[0, 2] = 1e6

    assert torch.allclose(
        centre_options(identity, mask)[mask], centre_options(noisy, mask)[mask], atol=1e-5
    )


def test_marker_cls_scores_mixed_option_counts(builder) -> None:
    two = DecisionQuestion.noul("Is it fraud?")
    three = DecisionQuestion.choice("Route", [Option("A"), Option("B"), Option("C")])
    batch = _batch(builder, [("payment failed", two), ("payment failed", three)])

    output = _cls_model().eval()(**batch)

    assert output.probabilities.shape == (2, 3)
    assert output.probabilities[0, 2].item() == 0
    assert torch.isfinite(output.probabilities).all()
    assert torch.allclose(output.probabilities.sum(dim=-1), torch.ones(2))


def test_marker_cls_trains_its_own_weights_and_the_word_embeddings(builder) -> None:
    three = DecisionQuestion.choice("Route", [Option("billing"), Option("support"), Option("risk")])
    batch = _batch(builder, [("payment failed", three)])
    model = _cls_model()

    model(**batch).logits[0, 1].backward()

    assert model.option_readout.weight.grad.abs().sum() > 0
    assert model.summary_norm.weight.grad.abs().sum() > 0
    assert model.encoder.get_input_embeddings().weight.grad.abs().sum() > 0


def test_marker_cls_learns_a_balanced_task_with_no_lexical_cue(builder) -> None:
    # Balanced labels, and options ("0" and "1") that share nothing with the state:
    # nothing here tells the options apart for a shared marker scorer, which is the
    # setting where it sat at the uniform guess on real data (balanced MultiNLI,
    # balanced BoolQ). The centred option identities break that symmetry.
    torch.manual_seed(0)
    question = DecisionQuestion.choice("which one", [Option("0"), Option("1")])
    states = ["payment refund invoice", "login password locked", "invoice charged twice"]
    states += ["account access blocked"]
    labels = torch.tensor([0, 1, 0, 1])
    batch = _batch(builder, [(state, question) for state in states])
    model = _cls_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(60):
        optimizer.zero_grad()
        logits = model(**batch).logits
        torch.nn.functional.cross_entropy(logits, labels).backward()
        optimizer.step()

    model.eval()
    assert model(**batch).logits.argmax(dim=-1).tolist() == labels.tolist()
