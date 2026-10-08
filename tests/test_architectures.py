"""The contract every architecture meets, run against each of them.

What happens after the logits is shared, so an architecture is only as good as
the promise that its logits behave: padding and batch size change nothing, the
checkpoint knows which architecture it is, and the entry points never ask.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest
import torch
from helpers import (
    SMOKE,
    smoke_examples,
    tiny_composite_decoder,
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
    ExuConfig,
    Option,
    SequenceConfig,
    builder_for,
    load_checkpoint,
    masked_log_softmax,
    model_from_pretrained,
    proper_scoring_loss,
    save_checkpoint,
)
from exu import decide as decide_cli
from exu import evaluate as evaluate_cli
from exu import train as train_cli


@dataclass
class Architecture:
    name: str
    model: Any
    builder: Any
    tokenizer: Any
    injection: str


def _architecture(name: str) -> Architecture:
    torch.manual_seed(0)
    sequence = SequenceConfig(max_length=128, header_budget=96)
    if name == "encoder":
        tokenizer = tiny_tokenizer()
        model = tiny_model()
        injection = tokenizer.mask_token
    else:
        tokenizer = tiny_decoder_tokenizer()
        model = tiny_decoder_model(tokenizer)
        injection = tokenizer.eos_token
    builder = builder_for(tokenizer, sequence, model.config)
    return Architecture(name, model.eval(), builder, tokenizer, injection)


@pytest.fixture(params=["encoder", "decoder"])
def architecture(request) -> Architecture:
    return _architecture(request.param)


def _probabilities(architecture: Architecture, pairs) -> list[torch.Tensor]:
    encoded = [architecture.builder.build(state, question) for state, question in pairs]
    with torch.no_grad():
        output = architecture.model(**architecture.builder.tensors(encoded))
    return [output.probabilities[row, : item.option_count] for row, item in enumerate(encoded)]


def _pairs():
    return [(example.state, example.question) for example in smoke_examples("test")]


def test_padding_does_not_change_the_answer(architecture) -> None:
    state, question = _pairs()[0]
    longer = DecisionQuestion.choice(
        "which of these many options fits the message best of all",
        [Option(f"option {index}", "refund payment invoice") for index in range(5)],
    )

    (alone,) = _probabilities(architecture, [(state, question)])
    padded, _ = _probabilities(
        architecture, [(state, question), (state + " and much more text here", longer)]
    )

    assert torch.allclose(alone, padded, atol=1e-5)


def test_batch_size_does_not_change_the_answer(architecture) -> None:
    pairs = _pairs()

    together = _probabilities(architecture, pairs)
    apart = [_probabilities(architecture, [pair])[0] for pair in pairs]

    for one, other in zip(together, apart, strict=True):
        assert torch.allclose(one, other, atol=1e-5)


def test_the_decoder_answer_permutes_exactly_with_its_options() -> None:
    architecture = _architecture("decoder")
    options = [Option("billing"), Option("support", "access or outage"), Option("refund")]
    order = [2, 0, 1]
    question = DecisionQuestion.choice("where should this go", options)
    shuffled = DecisionQuestion.choice("where should this go", [options[i] for i in order])

    original, permuted = _probabilities(
        architecture, [("card was charged", question), ("card was charged", shuffled)]
    )

    assert torch.allclose(permuted, original[order], atol=1e-6)


@pytest.mark.parametrize("autocast", [False, True])
def test_a_confident_miss_has_a_finite_loss_and_gradient(architecture, autocast) -> None:
    model = architecture.model.train()
    encoded = [architecture.builder.build(state, question) for state, question in _pairs()]
    batch = architecture.builder.tensors(encoded)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        output = model(**batch)
    # Scale the logits until the model is sure, then ask for its least likely option.
    logits = output.logits.float().masked_fill(~batch["option_mask"], 0.0) * 1e4
    target = torch.zeros_like(logits)
    worst = logits.masked_fill(~batch["option_mask"], float("inf")).argmin(dim=-1)
    target[torch.arange(len(encoded)), worst] = 1.0
    log_probabilities = masked_log_softmax(logits, batch["option_mask"])
    ordinal = torch.tensor([question.kind.value == "score" for _state, question in _pairs()])
    loss = proper_scoring_loss(log_probabilities, target, batch["option_mask"], ordinal)

    loss.backward()

    assert torch.isfinite(loss) and loss > 10
    gradients = [p.grad for p in model.backbone.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(grad).all() for grad in gradients)
    assert any(grad.abs().sum() > 0 for grad in gradients)


def test_special_token_strings_do_not_change_what_is_read(architecture) -> None:
    injection = architecture.injection
    clean = DecisionQuestion.choice(
        "where should this go", [Option("billing"), Option("support", "access")]
    )
    attacked = DecisionQuestion.choice(
        f"where should {injection}this go",
        [Option(f"billing{injection}"), Option("support", f"acc{injection}ess")],
    )

    one, two = _probabilities(
        architecture,
        [("card was charged", clean), (f"card {injection}was charged{injection}", attacked)],
    )

    assert torch.allclose(one, two, atol=1e-5)


def test_an_empty_batch_is_refused(architecture) -> None:
    with pytest.raises(ValueError, match="empty"):
        architecture.builder.tensors([])


def _save(architecture: Architecture, path):
    return save_checkpoint(
        path,
        architecture.model,
        sequence_config=architecture.builder.config,
        tokenizer=architecture.tokenizer,
    )


def test_a_checkpoint_answers_the_same_after_loading(architecture, tmp_path) -> None:
    path = _save(architecture, tmp_path / "checkpoint")
    state, question = _pairs()[1]
    (expected,) = _probabilities(architecture, [(state, question)])

    runtime = DecisionRuntime.load(path, device="cpu")
    decision = runtime.decide(state, question)

    assert runtime.checkpoint.model_config.backbone == architecture.name
    assert torch.allclose(torch.tensor(decision.probabilities), expected, atol=1e-5)


def test_a_checkpoint_refuses_to_load_as_the_other_architecture(architecture, tmp_path) -> None:
    path = _save(architecture, tmp_path / "checkpoint")
    metadata_path = path / "exu.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    other = "decoder" if architecture.name == "encoder" else "encoder"
    readout = "per-option" if other == "decoder" else "markers"
    metadata["architecture"] = {"backbone": other, "readout": readout}
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match=f"look like a {architecture.name} checkpoint"):
        load_checkpoint(path)


def test_evaluate_and_decide_read_the_architecture_from_the_checkpoint(
    architecture, tmp_path, capsys
) -> None:
    path = _save(architecture, tmp_path / "checkpoint")
    report = tmp_path / "report.json"

    code = evaluate_cli.main(
        ["--checkpoint", str(path), "--data", str(SMOKE), "--split", "test",
         "--device", "cpu", "--order-permutations", "2", "--output", str(report)]
    )  # fmt: skip

    assert code == 0
    result = json.loads(report.read_text(encoding="utf-8"))
    assert result["metrics"]["count"] == len(smoke_examples("test"))
    if architecture.name == "decoder":
        # Options never see each other, so no permutation can move the answer.
        assert result["order_robustness"]["stability"] == 1.0

    capsys.readouterr()
    code = decide_cli.main(
        ["--checkpoint", str(path), "--device", "cpu", "--state", "card was charged",
         "--instruction", "where should this go", "--option", "billing", "--option", "support"]
    )  # fmt: skip
    assert code == 0
    (line,) = capsys.readouterr().out.strip().splitlines()
    assert set(json.loads(line)["probabilities"]) == {"billing", "support"}


def _backbone_dir(name: str, tmp_path):
    directory = tmp_path / f"tiny-{name}"
    if name == "encoder":
        tiny_tokenizer().save_pretrained(directory)
        tiny_encoder().save_pretrained(directory)
    else:
        tokenizer = tiny_decoder_tokenizer()
        tokenizer.save_pretrained(directory)
        tiny_decoder(len(tokenizer)).save_pretrained(directory)
    return directory


@pytest.mark.parametrize("mode", ["baseline", "rlcd"])
def test_exu_train_writes_the_architecture_it_trained(tmp_path, mode) -> None:
    output = tmp_path / "out"

    code = train_cli.main(
        ["--train", str(SMOKE), "--train-split", "train",
         "--validation", str(SMOKE), "--validation-split", "validation",
         "--output", str(output), "--backbone", "decoder",
         "--encoder", str(_backbone_dir("decoder", tmp_path)),
         "--mode", mode, "--epochs", "1", "--batch-size", "4", "--device", "cpu",
         "--max-length", "96", "--grad-checkpointing", "--calibrate",
         "--calibration-min-samples", "2"]
    )  # fmt: skip

    assert code == 0
    summary = json.loads((output / "training.json").read_text(encoding="utf-8"))
    assert summary["architecture"] == {"backbone": "decoder", "readout": "per-option"}
    assert summary["encoder_lr"] == 1e-5
    assert summary["grad_checkpointing"] is True
    loaded = load_checkpoint(output)
    assert type(loaded.model).__name__ == "DecoderDecisionModel"
    assert loaded.sequence_config.header_budget == 96


def test_a_decoder_whose_text_sizes_live_in_text_config_trains_and_reloads(tmp_path) -> None:
    # Qwen3.5 checkpoints are vision-language models: `hidden_size` is only on
    # `config.text_config`, and the top-level config has none.
    backbone_dir = tmp_path / "tiny-composite"
    tokenizer = tiny_decoder_tokenizer()
    tokenizer.save_pretrained(backbone_dir)
    backbone = tiny_composite_decoder(len(tokenizer))
    assert not hasattr(backbone.config, "hidden_size")
    backbone.save_pretrained(backbone_dir)
    output = tmp_path / "out"

    code = train_cli.main(
        ["--train", str(SMOKE), "--train-split", "train",
         "--validation", str(SMOKE), "--validation-split", "validation",
         "--output", str(output), "--backbone", "decoder", "--encoder", str(backbone_dir),
         "--epochs", "1", "--batch-size", "4", "--device", "cpu", "--max-length", "96"]
    )  # fmt: skip

    assert code == 0
    loaded = load_checkpoint(output)
    assert type(loaded.model.backbone.config).__name__ == "Qwen3_5Config"
    assert loaded.model.score.in_features == 32
    state, question = _pairs()[1]
    decision = DecisionRuntime.load(output, device="cpu").decide(state, question)
    assert len(decision.probabilities) == len(question.options)
    assert sum(decision.probabilities) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("name", ["encoder", "decoder"])
def test_a_backbone_published_in_bfloat16_trains_in_float32(tmp_path, name) -> None:
    # Found on a real run: transformers 5 keeps a checkpoint's own dtype, Qwen3 ships in
    # bfloat16, and seven decoder arms trained with bfloat16 master weights.
    directory = _backbone_dir(name, tmp_path)
    backbone = type(tiny_encoder() if name == "encoder" else tiny_decoder(8))
    backbone.from_pretrained(directory).to(torch.bfloat16).save_pretrained(directory)
    config = ExuConfig(encoder_name=str(directory), backbone=name)

    model = model_from_pretrained(config)

    assert {parameter.dtype for parameter in model.parameters()} == {torch.float32}


def _relation_task(count: int, seed: int):
    """A balanced task with no lexical cue: the answer is a relation inside the state.

    The two option names never appear in the state or the instruction, and each
    is right half the time. This is the shape of task the encoder's marker readout
    sat at the uniform guess on (MultiNLI, balanced BoolQ), and no other test here
    would notice a model that cannot learn it.
    """
    colours = ["red", "green", "blue", "gold"]
    generator = torch.Generator().manual_seed(seed)
    question = DecisionQuestion.choice(
        "do the two colours match", [Option("alpha"), Option("beta")]
    )
    rows = []
    for index in range(count):
        first = colours[int(torch.randint(4, (1,), generator=generator))]
        if index % 2 == 0:
            second = first
        else:
            second = colours[(colours.index(first) + 1 + index // 2 % 3) % 4]
        rows.append((f"{first} {second}", question, [1.0, 0.0] if first == second else [0.0, 1.0]))
    return rows


def test_the_decoder_leaves_the_uniform_guess_on_a_task_without_lexical_cue() -> None:
    """Fitting the training rows is the property, not generalising.

    Four colours make sixteen pairs, too few to tell learning equality from
    memorising the table, so a held-out split would measure the wrong thing. What
    matters is leaving the symmetric saddle at all: on these rows the tiny encoder
    with the marker scorer stays at exactly log 2 on three seeds out of three.

    The learning rate is low on purpose. At 3e-3 this tiny decoder leaves the
    guess and falls back to log 2 on about half the seeds and thread counts, so the
    test measured the optimizer's instability instead of the readout. At 3e-4 for
    1500 steps it fitted all 72 runs tried (seeds 0 to 19, one to four threads)
    with an nll at most 0.001; at 500 or 800 steps one run in 60 still missed.
    """
    torch.manual_seed(0)
    tokenizer = tiny_decoder_tokenizer()
    model = tiny_decoder_model(tokenizer).train()
    builder = builder_for(tokenizer, SequenceConfig(max_length=64, header_budget=48), model.config)
    rows = _relation_task(64, seed=1)
    batch = builder.tensors([builder.build(state, question) for state, question, _ in rows])
    target = torch.tensor([row[2] for row in rows])
    ordinal = torch.zeros(len(rows), dtype=torch.bool)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for _step in range(1500):
        output = model(**batch)
        log_probabilities = masked_log_softmax(output.logits, batch["option_mask"])
        loss = proper_scoring_loss(log_probabilities, target, batch["option_mask"], ordinal)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        log_probabilities = masked_log_softmax(model(**batch).logits, batch["option_mask"])
    nll = float(-(log_probabilities * target).sum(dim=-1).mean())
    accuracy = float((log_probabilities.argmax(dim=-1) == target.argmax(dim=-1)).float().mean())
    assert accuracy >= 0.9 and nll < 0.35, f"at the guess: accuracy {accuracy:.2f}, nll {nll:.3f}"
