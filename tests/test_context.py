"""The optional context: read before the state, with a budget of its own."""

from __future__ import annotations

import json

import pytest
import torch
from helpers import (
    SMOKE,
    smoke_examples,
    tiny_decoder,
    tiny_decoder_model,
    tiny_decoder_tokenizer,
    tiny_encoder,
    tiny_model,
    tiny_tokenizer,
)

from exu import (
    DecisionQuestion,
    DecisionRuntime,
    DecoderSequenceBuilder,
    Option,
    TrainingExample,
    decide,
    load_checkpoint,
    train,
)
from exu.augmentation import permute_example
from exu.sequence import SequenceBuilder, SequenceConfig

QUESTION = DecisionQuestion.choice("where should this go", [Option("billing"), Option("refund")])
STATE = "card was charged twice"
CONTEXT = "refund only within thirty days"


def _encoder(**config) -> SequenceBuilder:
    return SequenceBuilder(tiny_tokenizer(), SequenceConfig(**config))


def _decoder(**config) -> DecoderSequenceBuilder:
    return DecoderSequenceBuilder(tiny_decoder_tokenizer(), SequenceConfig(**config))


def _ids(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


@pytest.mark.parametrize("make", [_encoder, _decoder])
def test_without_a_context_the_sequence_does_not_change(make) -> None:
    builder = make()

    assert builder.build(STATE, QUESTION) == builder.build(STATE, QUESTION, None)


def test_the_encoder_reads_the_context_between_the_options_and_the_state() -> None:
    builder = _encoder()
    sep = builder.tokenizer.sep_token_id
    plain = builder.build(STATE, QUESTION)

    encoded = builder.build(STATE, QUESTION, CONTEXT)

    context, state = _ids(builder.tokenizer, CONTEXT), _ids(builder.tokenizer, STATE)
    # The header and its markers are where they were; context and SEP follow them.
    header = len(plain.input_ids) - len(state) - 1
    assert encoded.marker_positions == plain.marker_positions
    assert list(encoded.input_ids[:header]) == list(plain.input_ids[:header])
    assert list(encoded.input_ids[header:]) == [*context, sep, *state, sep]


def test_the_decoder_reads_the_context_before_the_state_in_every_sequence() -> None:
    builder = _decoder()
    tokenizer = builder.tokenizer

    encoded = builder.build(STATE, QUESTION, CONTEXT)

    prefix = [
        *_ids(tokenizer, "Context: "),
        *_ids(tokenizer, CONTEXT),
        *_ids(tokenizer, "\nState: "),
        *_ids(tokenizer, STATE),
    ]
    assert encoded.prefix_length == len(prefix)
    for sequence in encoded.sequences:
        assert list(sequence[: len(prefix)]) == prefix


@pytest.mark.parametrize("make", [_encoder, _decoder])
def test_a_context_over_its_budget_is_refused_and_one_at_it_is_built(make) -> None:
    budget = len(_ids(make().tokenizer, CONTEXT))

    make(context_budget=budget).build(STATE, QUESTION, CONTEXT)
    with pytest.raises(ValueError, match="context_budget"):
        make(context_budget=budget - 1).build(STATE, QUESTION, CONTEXT)


@pytest.mark.parametrize("make", [_encoder, _decoder])
def test_a_long_state_is_cut_and_the_context_kept_whole(make) -> None:
    builder = make(max_length=64, header_budget=40)
    tokenizer = builder.tokenizer
    context = _ids(tokenizer, CONTEXT)

    encoded = builder.build(" ".join(["card"] * 300), QUESTION, CONTEXT)

    sequences = getattr(encoded, "sequences", None) or [encoded.input_ids]
    for sequence in sequences:
        assert len(sequence) <= 64
        text = list(sequence)
        assert any(text[i : i + len(context)] == context for i in range(len(text)))


@pytest.mark.parametrize("make", [_encoder, _decoder])
def test_a_context_and_question_that_leave_no_room_are_refused(make) -> None:
    builder = make(max_length=24, header_budget=20, context_budget=200)

    with pytest.raises(ValueError):
        builder.build(STATE, QUESTION, " ".join(["refund"] * 40))


@pytest.mark.parametrize(
    ("make", "token"),
    [(_encoder, "[MASK]"), (_decoder, "<eos>")],
)
def test_special_token_strings_in_the_context_are_stripped(make, token: str) -> None:
    builder = make()

    planted = builder.build(STATE, QUESTION, f"refund {token} only")
    clean = builder.build(STATE, QUESTION, "refund  only")

    assert planted == clean


@pytest.mark.parametrize(
    ("make", "context"),
    # Empty as given, or empty once the builder strips its special-token strings.
    [(_encoder, ""), (_encoder, "[MASK]"), (_decoder, ""), (_decoder, "<eos>")],
)
def test_an_empty_context_is_refused(make, context: str) -> None:
    with pytest.raises(ValueError, match="context cannot be empty"):
        make().build(STATE, QUESTION, context)


@pytest.mark.parametrize("name", ["encoder", "decoder"])
def test_the_model_answers_differently_with_a_context(name) -> None:
    torch.manual_seed(0)
    if name == "encoder":
        builder, model = _encoder(), tiny_model()
    else:
        tokenizer = tiny_decoder_tokenizer()
        builder, model = _decoder(), tiny_decoder_model(tokenizer)
    model.eval()

    with torch.no_grad():
        plain = model(**builder.tensors([builder.build(STATE, QUESTION)])).logits
        read = model(**builder.tensors([builder.build(STATE, QUESTION, CONTEXT)])).logits

    assert not torch.allclose(plain, read)


def test_one_context_over_many_states_shares_and_matches_the_full_sequences() -> None:
    torch.manual_seed(0)
    tokenizer = tiny_decoder_tokenizer()
    builder = _decoder()
    from exu import DecoderDecisionModel, ExuConfig

    model = DecoderDecisionModel(
        tiny_decoder(len(tokenizer)), ExuConfig(encoder_name="tiny", backbone="decoder")
    ).eval()
    encoded = [
        builder.build(state, QUESTION, CONTEXT)
        for state in ("card was charged", "card refund", "card was charged")
    ]
    batch = builder.tensors(encoded)

    with torch.no_grad():
        shared = model(**batch).logits
        full = model(**{k: v for k, v in batch.items() if k != "prefix_lengths"}).logits

    assert torch.allclose(shared, full, atol=1e-5)
    assert torch.allclose(shared[0], shared[2])


def test_a_record_context_round_trips_and_a_record_without_one_is_unchanged() -> None:
    record = {
        "id": "t-1",
        "state": STATE,
        "context": {"policy": "refund only within thirty days", "tier": 2},
        "question": {
            "kind": "noul",
            "instruction": "refund?",
            "options": [{"name": "no"}, {"name": "yes"}],
        },
        "target": [0.25, 0.75],
    }

    example = TrainingExample.from_record(record)

    assert example.context == '{"policy":"refund only within thirty days","tier":2}'
    assert example.to_record()["context"] == record["context"]
    bare = TrainingExample.from_record({k: v for k, v in record.items() if k != "context"})
    assert bare.context is None
    assert "context" not in bare.to_record()


@pytest.mark.parametrize("context", ["", "   ", [], 7])
def test_a_record_with_an_empty_or_malformed_context_is_refused(context) -> None:
    record = {
        "id": "t-1",
        "state": STATE,
        "context": context,
        "question": {
            "kind": "noul",
            "instruction": "refund?",
            "options": [{"name": "no"}, {"name": "yes"}],
        },
        "target": [0.25, 0.75],
    }

    with pytest.raises(ValueError, match="context"):
        TrainingExample.from_record(record)


def test_shuffling_the_options_keeps_the_context() -> None:
    example = smoke_examples("train")[0]
    with_context = TrainingExample(
        example.example_id, example.state, example.question, example.target, context=CONTEXT
    )

    permuted = permute_example(with_context, list(reversed(range(example.option_count))))

    assert permuted.context == CONTEXT


def test_the_context_budget_is_validated_and_survives_a_round_trip() -> None:
    with pytest.raises(ValueError, match="context_budget"):
        SequenceConfig(context_budget=-1)
    config = SequenceConfig(context_budget=77)

    assert SequenceConfig.from_dict(config.to_dict()).context_budget == 77
    # A checkpoint written before the field existed loads with the default.
    old = {key: value for key, value in config.to_dict().items() if key != "context_budget"}
    assert SequenceConfig.from_dict(old).context_budget == 256


def _context_dataset(tmp_path, context: str = CONTEXT):
    rows = [json.loads(line) for line in SMOKE.read_text(encoding="utf-8").splitlines() if line]
    path = tmp_path / "with-context.jsonl"
    path.write_text(
        "\n".join(json.dumps({**row, "context": context}) for row in rows) + "\n", encoding="utf-8"
    )
    return path


def _train(tmp_path, data, *extra: str) -> int:
    encoder = tmp_path / "tiny-encoder"
    tiny_tokenizer().save_pretrained(encoder)
    tiny_encoder().save_pretrained(encoder)
    return train.main(
        ["--train", str(data), "--train-split", "train",
         "--validation", str(data), "--validation-split", "validation",
         "--output", str(tmp_path / "out"), "--encoder", str(encoder),
         "--mode", "baseline", "--epochs", "1", "--batch-size", "4", "--device", "cpu", *extra]
    )  # fmt: skip


def test_exu_train_reads_contexts_and_the_checkpoint_keeps_the_budget(tmp_path) -> None:
    data = _context_dataset(tmp_path)

    assert _train(tmp_path, data, "--context-budget", "32") == 0

    assert load_checkpoint(tmp_path / "out").sequence_config.context_budget == 32


def test_exu_train_refuses_a_context_over_budget_before_training(tmp_path) -> None:
    data = _context_dataset(tmp_path, " ".join(["refund"] * 50))

    with pytest.raises(ValueError, match="record .*context_budget"):
        _train(tmp_path, data, "--context-budget", "8")
    assert not (tmp_path / "out").exists()


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    root = tmp_path_factory.mktemp("context")
    assert _train(root, _context_dataset(root)) == 0
    return root / "out"


def test_the_runtime_takes_a_context_in_every_entry_point(checkpoint) -> None:
    runtime = DecisionRuntime.load(checkpoint, device="cpu")

    alone = runtime.decide(STATE, QUESTION, CONTEXT)
    (many,) = runtime.decide_many([(STATE, QUESTION, CONTEXT)])
    (together,) = runtime.decide_all(STATE, [QUESTION], CONTEXT)
    plain = runtime.decide(STATE, QUESTION)

    assert alone.probabilities == pytest.approx(many.probabilities, abs=1e-6)
    assert alone.probabilities == pytest.approx(together.probabilities, abs=1e-6)
    assert alone.probabilities != pytest.approx(plain.probabilities, abs=1e-6)


def _decide(capsys, checkpoint, *arguments: str):
    code = decide.main(["--checkpoint", str(checkpoint), "--device", "cpu", *arguments])
    captured = capsys.readouterr()
    return code, [json.loads(line) for line in captured.out.splitlines() if line], captured.err


def test_exu_decide_reads_the_context_of_a_record_and_of_its_flags(
    capsys, checkpoint, tmp_path
) -> None:
    runtime = DecisionRuntime.load(checkpoint, device="cpu")
    expected = runtime.decide(STATE, QUESTION, CONTEXT)
    question = {"kind": "choice", "instruction": "where should this go",
                "options": [{"name": "billing"}, {"name": "refund"}]}  # fmt: skip
    source = tmp_path / "in.jsonl"
    source.write_text(
        json.dumps({"state": STATE, "context": CONTEXT, "question": question})
        + "\n"
        + json.dumps({"state": STATE, "context": CONTEXT, "questions": [question]})
        + "\n",
        encoding="utf-8",
    )
    context_file = tmp_path / "context.txt"
    context_file.write_text(CONTEXT, encoding="utf-8")
    flags = ["--state", STATE, "--instruction", "where should this go",
             "--option", "billing", "--option", "refund"]  # fmt: skip

    _code, records, _err = _decide(capsys, checkpoint, "--input", str(source))
    _code, (from_flag,), _err = _decide(capsys, checkpoint, *flags, "--context", CONTEXT)
    _code, (from_file,), _err = _decide(
        capsys, checkpoint, *flags, "--context-file", str(context_file)
    )

    for row in [*records, from_flag, from_file]:
        assert list(row["probabilities"].values()) == pytest.approx(
            list(expected.probabilities), abs=1e-5
        )


def test_exu_decide_refuses_two_contexts_and_names_an_empty_one(
    capsys, checkpoint, tmp_path
) -> None:
    with pytest.raises(SystemExit):
        decide.main(["--checkpoint", str(checkpoint), "--state", STATE,
                     "--instruction", "where", "--option", "a",
                     "--context", "x", "--context-file", str(tmp_path / "c.txt")])  # fmt: skip
    capsys.readouterr()
    source = tmp_path / "in.jsonl"
    record = {"state": STATE, "context": "", "question": {"kind": "noul", "instruction": "?",
              "options": [{"name": "no"}, {"name": "yes"}]}}  # fmt: skip
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")

    code, rows, err = _decide(capsys, checkpoint, "--input", str(source))

    assert code == 1
    assert "line 1" in err and "context" in err
    assert rows == []


class _SpyBuilder:
    """A real encoder builder that records the context of every sequence it builds."""

    def __init__(self) -> None:
        self.inner = _encoder()
        self.contexts: list[str | None] = []

    def build(self, state, question, context=None):
        self.contexts.append(context)
        return self.inner.build(state, question, context)

    def tensors(self, encoded, device=None):
        return self.inner.tensors(encoded, device)


def test_training_and_every_evaluation_pass_hand_the_context_to_the_builder() -> None:
    from exu import DecisionDataset, TemperatureMap
    from exu.evaluation import latency_benchmark, order_robustness

    examples = [
        TrainingExample(e.example_id, e.state, e.question, e.target, context=CONTEXT)
        for e in smoke_examples("test")
        if e.question.kind.value != "score"
    ]
    model = tiny_model().eval()

    for run in (
        lambda spy: DecisionDataset(examples, spy).collate(examples),
        lambda spy: order_robustness(model, spy, examples, TemperatureMap(), permutations=1),
        lambda spy: latency_benchmark(
            model, spy, examples, TemperatureMap(), questions_per_call=(2,), repeats=1
        ),
    ):
        spy = _SpyBuilder()
        run(spy)
        assert spy.contexts and set(spy.contexts) == {CONTEXT}
