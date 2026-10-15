"""Adapters: what they train, and that merging them changes no answer."""

from __future__ import annotations

import json

import pytest
import torch
from helpers import SMOKE, tiny_decoder, tiny_decoder_tokenizer, tiny_encoder, tiny_tokenizer
from test_architectures import _architecture, _pairs, _probabilities
from torch import nn

from exu import LoRALinear, apply_lora, load_checkpoint, merge_lora, save_checkpoint
from exu import train as train_cli


def _perturb(module: nn.Module) -> None:
    """Give every adapter a non-zero update, as training would."""
    torch.manual_seed(1)
    for child in module.modules():
        if isinstance(child, LoRALinear):
            nn.init.normal_(child.lora_b, std=0.05)


def test_a_fresh_adapter_answers_exactly_like_its_base() -> None:
    torch.manual_seed(0)
    base = nn.Linear(6, 4)
    inputs = torch.randn(3, 6)
    expected = base(inputs)

    assert torch.equal(LoRALinear(base, rank=2, alpha=2)(inputs), expected)


@pytest.mark.parametrize(("rank", "alpha"), [(0, 1.0), (2, 0.0), (2, -1.0)])
def test_an_invalid_rank_or_alpha_is_refused(rank, alpha) -> None:
    with pytest.raises(ValueError):
        LoRALinear(nn.Linear(4, 4), rank=rank, alpha=alpha)


def test_the_merged_layer_answers_like_the_adapter() -> None:
    torch.manual_seed(0)
    layer = LoRALinear(nn.Linear(6, 4), rank=2, alpha=8)
    nn.init.normal_(layer.lora_b)
    inputs = torch.randn(3, 6)

    merged = layer.merged()

    assert type(merged) is nn.Linear
    assert torch.allclose(merged(inputs), layer(inputs), atol=1e-5)


@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("default_dtype", [torch.float32, torch.float16])
def test_merge_stays_float32_under_autocast_and_a_low_default_dtype(
    autocast, default_dtype
) -> None:
    model = nn.Sequential(LoRALinear(nn.Linear(1, 1), rank=1, alpha=1))
    nn.init.zeros_(model[0].base.weight)
    nn.init.constant_(model[0].base.bias, 0.1)
    nn.init.constant_(model[0].lora_a, 300.0)
    nn.init.constant_(model[0].lora_b, 300.0)
    inputs = torch.tensor([[0.001]], dtype=torch.float32)
    expected = model(inputs).detach()
    previous_dtype = torch.get_default_dtype()

    try:
        torch.set_default_dtype(default_dtype)
        with torch.autocast(device_type="cpu", dtype=torch.float16, enabled=autocast):
            count = merge_lora(model)
    finally:
        torch.set_default_dtype(previous_dtype)

    assert count == 1
    assert type(model[0]) is nn.Linear
    assert model[0].weight.dtype == torch.float32
    assert model[0].bias.dtype == torch.float32
    torch.testing.assert_close(model[0].weight, torch.full((1, 1), 90000.0))
    torch.testing.assert_close(model(inputs), expected)
    assert merge_lora(model) == 0


def test_a_small_update_survives_the_merge_into_a_bfloat16_base() -> None:
    # Added in bfloat16, an update a thousand times smaller than the weight rounds away.
    torch.manual_seed(0)
    base = nn.Linear(8, 8, bias=False).to(torch.bfloat16)
    nn.init.constant_(base.weight, 1.0)
    layer = LoRALinear(base, rank=1, alpha=1)
    nn.init.constant_(layer.lora_a, 1e-2)
    nn.init.constant_(layer.lora_b, 1e-1)

    merged = layer.merged()

    assert merged.weight.dtype == torch.float32
    assert torch.allclose(merged.weight, torch.full((8, 8), 1.001), atol=1e-6)


def test_a_bfloat16_base_answers_like_a_float32_one_under_autocast() -> None:
    # Autocast runs the matmul in bfloat16 either way, so storing the frozen weight
    # in bfloat16 changes the memory, not the answer.
    torch.manual_seed(0)
    full = LoRALinear(nn.Linear(16, 8), rank=2, alpha=2)
    nn.init.normal_(full.lora_b)
    half = LoRALinear(nn.Linear(16, 8), rank=2, alpha=2, base_dtype=torch.bfloat16)
    half.load_state_dict(full.state_dict())
    inputs = torch.randn(4, 16)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        expected, got = full(inputs), half(inputs)

    assert half.base.weight.dtype == torch.bfloat16
    assert half.lora_a.dtype == torch.float32
    assert torch.equal(got, expected)


def test_a_bfloat16_base_still_answers_outside_autocast() -> None:
    # Evaluation runs without autocast: a float32 input must not hit a dtype error.
    torch.manual_seed(0)
    full = LoRALinear(nn.Linear(16, 8), rank=2, alpha=2)
    half = LoRALinear(nn.Linear(16, 8), rank=2, alpha=2, base_dtype=torch.bfloat16)
    half.load_state_dict(full.state_dict())
    inputs = torch.randn(4, 16)

    got = half(inputs)

    assert got.dtype == torch.float32
    assert torch.allclose(got, full(inputs), atol=5e-2)


def test_a_model_with_a_bfloat16_base_trains_and_merges_to_float32() -> None:
    architecture = _architecture("decoder")
    model = architecture.model.train()
    apply_lora(model.backbone, rank=2, alpha=2, base_dtype=torch.bfloat16)
    encoded = [architecture.builder.build(state, question) for state, question in _pairs()]
    batch = architecture.builder.tensors(encoded)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = model(**batch).logits.float()
    logits.masked_fill(~batch["option_mask"], 0.0).sum().backward()

    adapters = [m for m in model.backbone.modules() if isinstance(m, LoRALinear)]
    assert all(m.lora_b.grad is not None for m in adapters)
    assert any(m.lora_b.grad.abs().sum() > 0 for m in adapters)
    merge_lora(model.backbone)
    assert {p.dtype for p in model.parameters()} == {torch.float32}


def test_apply_freezes_the_backbone_and_trains_only_the_adapters() -> None:
    model = _architecture("decoder").model
    linear = sum(isinstance(child, nn.Linear) for child in model.backbone.modules())

    count = apply_lora(model.backbone, rank=2, alpha=2)

    assert count == linear
    trainable = {name for name, p in model.backbone.named_parameters() if p.requires_grad}
    assert trainable and all(name.endswith(("lora_a", "lora_b")) for name in trainable)
    # The head is not the backbone's: it stays trainable.
    assert all(p.requires_grad for p in model.score.parameters())


def test_apply_refuses_a_root_linear_before_freezing_or_creating_a_cycle() -> None:
    layer = nn.Linear(4, 4)

    with pytest.raises(ValueError, match="parent module"):
        apply_lora(layer, rank=2, alpha=2)

    assert not list(layer.children())
    assert all(parameter.requires_grad for parameter in layer.parameters())
    assert set(layer.state_dict()) == {"weight", "bias"}


def test_merge_refuses_a_root_adapter_without_registering_an_unused_child() -> None:
    layer = LoRALinear(nn.Linear(4, 4), rank=2, alpha=2)
    keys = set(layer.state_dict())
    trainable = {name: parameter.requires_grad for name, parameter in layer.named_parameters()}

    for _ in range(2):
        with pytest.raises(ValueError, match="parent module"):
            merge_lora(layer)

    assert list(layer.children()) == [layer.base]
    assert set(layer.state_dict()) == keys
    assert {name: p.requires_grad for name, p in layer.named_parameters()} == trainable


def test_a_backbone_without_linear_layers_is_refused() -> None:
    with pytest.raises(ValueError, match="no nn.Linear"):
        apply_lora(nn.Sequential(nn.LayerNorm(4)), rank=2, alpha=2)


def test_adapters_are_not_stacked_on_adapters() -> None:
    backbone = nn.Sequential(nn.Linear(4, 4))
    apply_lora(backbone, rank=2, alpha=2)

    with pytest.raises(ValueError, match="already"):
        apply_lora(backbone, rank=2, alpha=2)


@pytest.mark.parametrize("name", ["encoder", "decoder"])
def test_a_merged_checkpoint_answers_like_the_adapted_model(name, tmp_path) -> None:
    architecture = _architecture(name)
    keys = set(architecture.model.state_dict())
    apply_lora(architecture.model.backbone, rank=2, alpha=4)
    _perturb(architecture.model.backbone)
    adapted = _probabilities(architecture, _pairs())

    assert merge_lora(architecture.model.backbone) > 0
    merged = _probabilities(architecture, _pairs())
    save_checkpoint(tmp_path / "ckpt", architecture.model, tokenizer=architecture.tokenizer)
    architecture.model = load_checkpoint(tmp_path / "ckpt").model
    loaded = _probabilities(architecture, _pairs())

    # Same keys as a model that never had adapters, so the loader needs nothing new.
    assert set(architecture.model.state_dict()) == keys
    for before, after, reloaded in zip(adapted, merged, loaded, strict=True):
        assert torch.allclose(before, after, atol=1e-5)
        assert torch.allclose(before, reloaded, atol=1e-5)


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


def _train(tmp_path, name: str, *extra: str) -> int:
    return train_cli.main(
        ["--train", str(SMOKE), "--train-split", "train",
         "--validation", str(SMOKE), "--validation-split", "validation",
         "--output", str(tmp_path / "out"), "--backbone", name,
         "--encoder", str(_backbone_dir(name, tmp_path)),
         "--mode", "baseline", "--epochs", "1", "--batch-size", "4", "--device", "cpu",
         "--max-length", "96", *extra]
    )  # fmt: skip


@pytest.mark.parametrize("name", ["encoder", "decoder"])
def test_exu_train_with_adapters_changes_the_backbone_and_saves_it_merged(name, tmp_path) -> None:
    code = _train(tmp_path, name, "--lora-rank", "4", "--grad-checkpointing")

    assert code == 0
    summary = json.loads((tmp_path / "out" / "training.json").read_text(encoding="utf-8"))
    assert summary["lora"]["rank"] == 4 and summary["lora"]["alpha"] == 4.0
    assert summary["lora"]["layers"] > 0
    assert summary["encoder_lr"] == 2e-4
    trained = load_checkpoint(tmp_path / "out").model.backbone.state_dict()
    base = _base_backbone(name, tmp_path)
    assert set(trained) == set(base)
    changed = [key for key, value in base.items() if not torch.equal(value, trained[key])]
    # Only the linear layers move; embeddings and norms stay frozen.
    assert changed
    assert not any("norm" in key or "embed" in key for key in changed)


def _base_backbone(name: str, tmp_path) -> dict:
    from transformers import AutoModel

    return AutoModel.from_pretrained(tmp_path / f"tiny-{name}", dtype=torch.float32).state_dict()


def test_lora_alpha_without_a_rank_is_refused(tmp_path) -> None:
    with pytest.raises(SystemExit):
        _train(tmp_path, "decoder", "--lora-alpha", "8")


def test_an_explicit_learning_rate_wins_over_the_adapter_default(tmp_path) -> None:
    assert _train(tmp_path, "decoder", "--lora-rank", "2", "--encoder-lr", "5e-5") == 0

    summary = json.loads((tmp_path / "out" / "training.json").read_text(encoding="utf-8"))
    assert summary["encoder_lr"] == 5e-5
    assert summary["lora"]["alpha"] == 2.0


def test_a_run_without_adapters_records_none(tmp_path) -> None:
    assert _train(tmp_path, "decoder") == 0

    summary = json.loads((tmp_path / "out" / "training.json").read_text(encoding="utf-8"))
    assert summary["lora"] is None
    assert summary["encoder_lr"] == 1e-5
