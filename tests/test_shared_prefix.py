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


def _layers(model: DecoderDecisionModel):
    return getattr(model.backbone, "language_model", model.backbone).layers


def _train_step(model: DecoderDecisionModel, batch) -> tuple[dict[str, torch.Tensor], dict]:
    """Gradients of a fixed weighting of the logits, and how the backbone was called."""
    model.zero_grad()
    calls = {"backbone": 0, "first_layer": []}
    hooks = [
        model.backbone.register_forward_pre_hook(
            lambda *_: calls.__setitem__("backbone", calls["backbone"] + 1)
        ),
        _layers(model)[0].register_forward_pre_hook(
            lambda _module, args: calls["first_layer"].append(args[0].shape[0])
        ),
    ]
    output = model(**batch)
    torch.manual_seed(1)
    weights = torch.randn(output.logits.shape)
    mask = batch["option_mask"]
    (output.logits[mask] * weights[mask]).sum().backward()
    for hook in hooks:
        hook.remove()
    grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    return grads, calls


def _assert_same_gradients(got: dict[str, torch.Tensor], want: dict[str, torch.Tensor]) -> None:
    assert got.keys() == want.keys()
    scale = max(grad.abs().max() for grad in want.values())
    for key, grad in want.items():
        assert torch.allclose(got[key], grad, atol=1e-5 * scale), key


@pytest.mark.parametrize("checkpointed", [False, True])
@pytest.mark.parametrize("name", sorted(BACKBONES))
def test_training_layer_by_layer_gives_the_full_gradients(name, checkpointed) -> None:
    model, builder = _model(name)
    model.train()
    if checkpointed:
        model.backbone.gradient_checkpointing_enable({"use_reentrant": False})
    batch = _batch(builder)
    full_batch = {key: value for key, value in batch.items() if key != "prefix_lengths"}

    shared, calls = _train_step(model, batch)
    full, _ = _train_step(model, full_batch)

    _assert_same_gradients(shared, full)
    # The backbone's own forward never runs: its layers do, prefixes then tails.
    assert calls["backbone"] == 0
    prefixes = len(set(batch["prefix_lengths"].tolist()))
    sequences = batch["input_ids"].size(0)
    forward = [prefixes, sequences]
    # A checkpointed step runs again in the backward pass; an unchecked one does not.
    assert calls["first_layer"] == (forward * 2 if checkpointed else forward)


@pytest.mark.parametrize("name", sorted(BACKBONES))
def test_contexts_train_layer_by_layer_with_the_full_gradients(name) -> None:
    # Two contexts over shared and distinct states: prefixes of several lengths,
    # where one context and one state group and the others do not.
    model, builder = _model(name)
    model.train()
    model.backbone.gradient_checkpointing_enable({"use_reentrant": False})
    question = DecisionQuestion.choice(
        "where should this go", [Option("billing"), Option("refund"), Option("support")]
    )
    items = [
        ("card was charged", "refund only within thirty days"),
        ("card refund", "refund only within thirty days"),
        ("card was charged", "always refund"),
        ("card was charged", "refund only within thirty days"),
    ]
    batch = builder.tensors([builder.build(state, question, context) for state, context in items])
    full_batch = {key: value for key, value in batch.items() if key != "prefix_lengths"}

    shared, calls = _train_step(model, batch)
    full, _ = _train_step(model, full_batch)

    _assert_same_gradients(shared, full)
    # Three distinct prefixes in the batch, each run once per layer pass.
    assert calls["first_layer"][0] == 3


@pytest.mark.parametrize("checkpointed", [False, True])
def test_no_layer_cache_outlives_its_step(checkpointed) -> None:
    # Each layer's cache holds its states copied once per tail. One kept alive by a
    # reference cycle waits for the garbage collector, and on a real base the
    # caches of every layer pile up in memory.
    import gc

    from transformers.cache_utils import CacheLayerMixin, LinearAttentionCacheLayerMixin

    layer_types = (CacheLayerMixin, LinearAttentionCacheLayerMixin)

    model, builder = _model("qwen3.5")
    model.train()
    if checkpointed:
        model.backbone.gradient_checkpointing_enable({"use_reentrant": False})
    batch = _batch(builder)
    gc.collect()
    gc.disable()
    try:
        _train_step(model, batch)
        cached = [
            obj for obj in gc.get_objects() if type(obj).__module__ == layer_types[0].__module__
        ]
        alive = [obj for obj in cached if isinstance(obj, layer_types)]
    finally:
        gc.enable()

    assert alive == []


def test_every_distinct_prefix_of_a_batch_runs_in_one_call_per_layer() -> None:
    model, builder = _model("qwen3.5")
    model.train()
    state = "card was charged twice for one invoice"
    questions = [
        DecisionQuestion.choice("where should this go", [Option("billing"), Option("support")]),
        DecisionQuestion.noul("is this a refund"),
    ]
    pairs = [(state, question) for question in questions] + [("card refund", questions[0])]
    batch = builder.tensors([builder.build(s, q) for s, q in pairs])
    full_batch = {key: value for key, value in batch.items() if key != "prefix_lengths"}

    shared, calls = _train_step(model, batch)
    full, _ = _train_step(model, full_batch)

    _assert_same_gradients(shared, full)
    # Two distinct states, so two prefixes in one call, then the 6 tails in another.
    assert calls["first_layer"] == [2, 6]


def test_a_checkpointing_backbone_in_training_mode_answers_like_the_full_sequences() -> None:
    # Such a backbone drops any cache it is given: the layer-by-layer path is the one
    # that still reads each tail on its prefix.
    model, builder = _model("qwen3")
    model.train()
    model.backbone.gradient_checkpointing_enable({"use_reentrant": False})
    batch = _batch(builder)

    shapes, output = _backbone_calls(model, batch)
    full = _full(model, batch)

    assert shapes == []
    mask = batch["option_mask"]
    assert torch.allclose(output.logits[mask], full.logits[mask], atol=1e-5)


def _tiny_llama(vocab_size: int):
    from transformers import LlamaConfig, LlamaModel

    return LlamaModel(
        LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=512,
        )
    )


def test_a_backbone_the_layer_path_does_not_know_trains_on_the_full_sequences() -> None:
    torch.manual_seed(0)
    tokenizer = tiny_decoder_tokenizer()
    model = DecoderDecisionModel(
        _tiny_llama(len(tokenizer)), ExuConfig(encoder_name="tiny", backbone="decoder")
    ).train()
    builder = DecoderSequenceBuilder(tokenizer, SequenceConfig(max_length=128, header_budget=96))
    batch = _batch(builder)

    _, calls = _train_step(model, batch)

    assert calls["backbone"] == 1
    assert calls["first_layer"] == [batch["input_ids"].size(0)]


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
