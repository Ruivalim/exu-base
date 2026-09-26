from __future__ import annotations

import pytest
import torch
from helpers import tiny_decoder_model, tiny_decoder_tokenizer

from exu import (
    DecisionQuestion,
    DecoderDecisionModel,
    DecoderSequenceBuilder,
    ExuConfig,
    ExuModel,
    Option,
    SequenceConfig,
)


def _question(*names: str) -> DecisionQuestion:
    return DecisionQuestion.choice("where should this go", [Option(name) for name in names])


def _builder(**sequence) -> DecoderSequenceBuilder:
    return DecoderSequenceBuilder(tiny_decoder_tokenizer(), SequenceConfig(**sequence))


def test_every_option_gets_its_own_sequence_with_the_same_prefix() -> None:
    builder = _builder()
    tokenizer = builder.tokenizer

    encoded = builder.build("card was charged", _question("billing", "support", "refund"))

    assert encoded.option_count == 3
    prefix = tokenizer.encode("State: card was charged", add_special_tokens=False)
    for sequence in encoded.sequences:
        assert list(sequence[: len(prefix)]) == prefix
        assert sequence[-1] == tokenizer.eos_token_id
        assert sequence.count(tokenizer.eos_token_id) == 1
    assert len({sequence[len(prefix) :] for sequence in encoded.sequences}) == 3


def test_a_long_state_is_cut_once_for_every_option() -> None:
    builder = _builder(max_length=40, header_budget=30)
    state = " ".join(["card"] * 200)

    encoded = builder.build(state, _question("billing", "the support refund team"))

    lengths = [len(sequence) for sequence in encoded.sequences]
    assert max(lengths) <= 40
    tails = [
        len(builder.tokenizer.encode(text, add_special_tokens=False))
        for text in ("billing", "the support refund team")
    ]
    # Same state for both options: the lengths differ by the option text only.
    assert lengths[1] - lengths[0] == tails[1] - tails[0]
    assert encoded.sequences[0][:10] == encoded.sequences[1][:10]


def test_a_question_that_does_not_fit_the_header_budget_raises() -> None:
    builder = _builder(max_length=64, header_budget=20)
    long_option = " ".join(["refund"] * 30)

    with pytest.raises(ValueError, match="header_budget"):
        builder.build("card was charged", _question("billing", long_option))


def test_a_question_exactly_at_the_header_budget_is_built() -> None:
    probe = _builder(max_length=128, header_budget=128)
    question = _question("billing", "support")
    encoded = probe.build("", question)
    needed = max(len(sequence) for sequence in encoded.sequences)

    fits = _builder(max_length=128, header_budget=needed).build("card was charged", question)
    assert max(len(sequence) for sequence in fits.sequences) <= 128
    with pytest.raises(ValueError, match="header_budget"):
        _builder(max_length=128, header_budget=needed - 1).build("card", question)


def test_special_token_strings_in_untrusted_text_are_stripped() -> None:
    builder = _builder()
    clean = builder.build("card was charged", _question("billing", "support"))

    attacked = builder.build(
        "card <e<eos>os>was <pad>charged<eos>",
        DecisionQuestion.choice(
            "where should <eos>this go", [Option("billing<eos>"), Option("sup<unk>port")]
        ),
    )

    assert attacked.sequences == clean.sequences


def test_a_tokenizer_without_an_eos_token_is_refused() -> None:
    tokenizer = tiny_decoder_tokenizer()
    tokenizer.eos_token = None

    with pytest.raises(ValueError, match="eos_token_id"):
        DecoderSequenceBuilder(tokenizer)


def test_tensors_pad_on_the_right_and_map_each_sequence_to_its_option() -> None:
    builder = _builder()
    encoded = [
        builder.build("card", _question("billing", "support")),
        builder.build("the refund never arrived at all", _question("a", "b", "c")),
    ]

    batch = builder.tensors(encoded)

    assert batch["input_ids"].shape[0] == 5
    assert batch["sequence_rows"].tolist() == [0, 0, 1, 1, 1]
    assert batch["sequence_slots"].tolist() == [0, 1, 0, 1, 2]
    assert batch["option_mask"].tolist() == [[True, True, False], [True, True, True]]
    lengths = batch["attention_mask"].sum(dim=1)
    assert torch.equal(batch["read_positions"], lengths - 1)
    reads = batch["input_ids"][torch.arange(5), batch["read_positions"]]
    assert reads.tolist() == [builder.tokenizer.eos_token_id] * 5
    for row, length in enumerate(lengths.tolist()):
        assert batch["attention_mask"][row, :length].all()
        assert not batch["attention_mask"][row, length:].any()


def test_an_empty_batch_is_refused() -> None:
    with pytest.raises(ValueError, match="empty"):
        _builder().tensors([])


def test_the_config_rejects_markers_on_a_decoder_and_marker_cls() -> None:
    with pytest.raises(ValueError, match="per-option"):
        ExuConfig(backbone="decoder", readout="markers")
    with pytest.raises(ValueError, match="markers"):
        ExuConfig(backbone="encoder", readout="per-option")
    with pytest.raises(ValueError, match="marker-cls"):
        ExuConfig(backbone="decoder", scorer="marker-cls")
    with pytest.raises(ValueError, match="backbone"):
        ExuConfig(backbone="rnn")
    assert ExuConfig(backbone="decoder").readout == "per-option"


def test_each_model_refuses_the_other_architecture() -> None:
    decoder = tiny_decoder_model()
    with pytest.raises(ValueError, match="DecoderDecisionModel"):
        ExuModel(decoder.backbone, ExuConfig(backbone="decoder"))
    with pytest.raises(ValueError, match="backbone='decoder'"):
        DecoderDecisionModel(decoder.backbone, ExuConfig())


def test_padded_options_get_no_probability_and_the_minimum_logit() -> None:
    builder = _builder()
    model = tiny_decoder_model(builder.tokenizer).eval()
    batch = builder.tensors(
        [
            builder.build("card", _question("billing", "support")),
            builder.build("refund", _question("a", "b", "c")),
        ]
    )

    with torch.no_grad():
        output = model(**batch)

    assert output.probabilities[0, 2] == 0
    assert output.logits[0, 2] == torch.finfo(output.logits.dtype).min
    assert torch.allclose(output.probabilities.sum(dim=-1), torch.ones(2))
    assert output.action_logits.shape == (2, 2)
