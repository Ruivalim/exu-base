"""The decoder's shared prefix: each distinct prefix runs once, with the K full runs' answers."""

from __future__ import annotations

import pytest
import torch
from helpers import smoke_examples, tiny_composite_decoder, tiny_decoder, tiny_decoder_tokenizer

from exu import (
    DecisionQuestion,
    DecoderDecisionModel,
    DecoderSequenceBuilder,
    EncodedOptions,
    ExuConfig,
    Option,
    SequenceConfig,
)

BACKBONES = {"qwen3": tiny_decoder, "qwen3.5": tiny_composite_decoder}


def _model(name: str) -> tuple[DecoderDecisionModel, DecoderSequenceBuilder]:
    torch.manual_seed(0)
    tokenizer = tiny_decoder_tokenizer()
    model = DecoderDecisionModel(
        BACKBONES[name](len(tokenizer)), ExuConfig(encoder_name="tiny", backbone="decoder")
    )
    builder = DecoderSequenceBuilder(tokenizer, SequenceConfig(max_length=128, header_budget=96))
    return model.eval(), builder


def _full(model: DecoderDecisionModel, batch):
    """The reference: the K full sequences, which a batch without prefix lengths runs."""
    with torch.no_grad():
        return model(**{key: value for key, value in batch.items() if key != "prefix_lengths"})


def _batch(builder: DecoderSequenceBuilder) -> dict[str, torch.Tensor]:
    pairs = [(example.state, example.question) for example in smoke_examples("test")]
    # A question with more options than the others, so some rows are padded, and a
    # longer state, so the prefixes of one batch differ in length.
    longer = DecisionQuestion.choice(
        "which of these many options fits the message best of all",
        [Option(f"option {index}", "refund payment invoice") for index in range(5)],
    )
    pairs.append((pairs[0][0] + " and much more text here", longer))
    return builder.tensors([builder.build(state, question) for state, question in pairs])


@pytest.mark.parametrize("name", sorted(BACKBONES))
def test_the_shared_prefix_answers_like_the_full_sequences(name) -> None:
    model, builder = _model(name)
    batch = _batch(builder)
    assert len(set(batch["prefix_lengths"].tolist())) > 1

    with torch.no_grad():
        shared = model(**batch)
    full = _full(model, batch)

    mask = batch["option_mask"]
    assert torch.allclose(shared.logits[mask], full.logits[mask], atol=1e-5)
    assert torch.allclose(shared.probabilities, full.probabilities, atol=1e-6)


@pytest.mark.parametrize("name", sorted(BACKBONES))
def test_a_state_cut_to_fit_answers_alike_both_ways(name) -> None:
    model, _ = _model(name)
    builder = DecoderSequenceBuilder(
        tiny_decoder_tokenizer(), SequenceConfig(max_length=40, header_budget=30)
    )
    question = DecisionQuestion.choice(
        "where should this go", [Option("billing"), Option("refund")]
    )
    batch = builder.tensors([builder.build(" ".join(["card"] * 200), question)])

    with torch.no_grad():
        shared = model(**batch).logits
    full = _full(model, batch).logits

    assert torch.allclose(shared, full, atol=1e-5)


def _backbone_calls(model: DecoderDecisionModel, batch) -> tuple[list[torch.Size], object]:
    shapes: list[torch.Size] = []
    hook = model.backbone.register_forward_pre_hook(
        lambda _module, _args, kwargs: shapes.append(kwargs["input_ids"].shape),
        with_kwargs=True,
    )
    with torch.no_grad():
        output = model(**batch)
    hook.remove()
    return shapes, output


def test_without_gradients_each_distinct_prefix_runs_once() -> None:
    model, builder = _model("qwen3")
    batch = _batch(builder)

    shapes, _ = _backbone_calls(model, batch)

    # Every state of this batch differs: one prefix run and one tail run each.
    questions = batch["option_mask"].size(0)
    assert len(shapes) == 2 * questions
    assert all(shape[0] == 1 for shape in shapes[0::2])
    assert all(shape[1] < batch["input_ids"].size(1) for shape in shapes[1::2])


@pytest.mark.parametrize("name", sorted(BACKBONES))
def test_questions_about_one_state_share_one_prefix_run(name) -> None:
    model, builder = _model(name)
    state = "card was charged twice for one invoice"
    questions = [
        DecisionQuestion.choice("where should this go", [Option("billing"), Option("support")]),
        DecisionQuestion.noul("is this a refund"),
        DecisionQuestion.score("how urgent is it", ["low", "medium", "high"]),
    ]
    batch = builder.tensors([builder.build(state, question) for question in questions])

    shapes, shared = _backbone_calls(model, batch)
    full = _full(model, batch)

    # One prefix run, then the 7 tails of the 3 questions in one call.
    assert [shape[0] for shape in shapes] == [1, 7]
    mask = batch["option_mask"]
    assert torch.allclose(shared.logits[mask], full.logits[mask], atol=1e-5)


def test_states_of_equal_length_do_not_share_a_prefix() -> None:
    model, builder = _model("qwen3")
    question = DecisionQuestion.choice(
        "where should this go", [Option("billing"), Option("support")]
    )
    encoded = [builder.build(state, question) for state in ("card refund", "card invoice")]
    assert encoded[0].prefix_length == encoded[1].prefix_length
    batch = builder.tensors(encoded)

    shapes, shared = _backbone_calls(model, batch)
    full = _full(model, batch)

    assert len(shapes) == 4
    assert torch.allclose(shared.logits, full.logits, atol=1e-5)
    assert not torch.allclose(shared.logits[0], shared.logits[1])


def test_a_batch_without_prefix_lengths_runs_the_full_sequences() -> None:
    model, builder = _model("qwen3")
    batch = _batch(builder)
    calls: list[int] = []
    hook = model.backbone.register_forward_pre_hook(lambda *_: calls.append(1))
    batch.pop("prefix_lengths")

    with torch.no_grad():
        model(**batch)
    hook.remove()

    assert len(calls) == 1


def test_with_gradients_the_full_sequences_run() -> None:
    model, builder = _model("qwen3")
    model.train()
    calls: list[int] = []
    hook = model.backbone.register_forward_pre_hook(lambda *_: calls.append(1))

    model(**_batch(builder))
    hook.remove()

    assert len(calls) == 1


def test_a_checkpointing_backbone_in_training_mode_runs_the_full_sequences() -> None:
    # Such a backbone drops the cache it is given: sharing would read tails alone.
    model, builder = _model("qwen3")
    model.train()
    model.backbone.gradient_checkpointing_enable({"use_reentrant": False})
    batch = _batch(builder)

    shapes, output = _backbone_calls(model, batch)
    full = _full(model, batch)

    assert len(shapes) == 1
    assert torch.equal(output.logits, full.logits)


@pytest.mark.parametrize(
    ("sequences", "prefix_length"),
    [
        (((1, 2, 3), (1, 9, 3)), 2),  # the second sequence does not share the prefix
        (((1, 2, 3), (1, 2)), 2),  # a sequence that is all prefix has nothing to read
        (((1, 2, 3), (1, 2, 4)), 0),  # no prefix at all
        ((), 1),  # no options
    ],
)
def test_encoded_options_refuse_a_prefix_the_sequences_do_not_share(
    sequences, prefix_length
) -> None:
    with pytest.raises(ValueError):
        EncodedOptions(sequences=sequences, question_type_id=0, prefix_length=prefix_length)


def test_the_builder_records_the_prefix_it_shares() -> None:
    builder = DecoderSequenceBuilder(tiny_decoder_tokenizer())
    question = DecisionQuestion.choice(
        "where should this go", [Option("billing"), Option("refund")]
    )

    encoded = builder.build("card was charged", question)

    prefix = builder.tokenizer.encode("State: card was charged", add_special_tokens=False)
    assert encoded.prefix_length == len(prefix)
