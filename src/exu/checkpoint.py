"""Portable checkpoint: config, weights, tokenizer and temperatures in one folder.

Layout::

    checkpoint/
      exu.json            # format version, architecture, model, sequence, temperatures
      model.safetensors    # every weight, backbone included, one file
      encoder_config/      # the backbone's config only, so loading never downloads weights
      tokenizer/           # tokenizer files, so inference is offline

Loading validates before it loads: format version, required keys, a state dict
that matches the stored architecture exactly. A clear error beats a silent wrong
answer when someone points at the wrong folder. The ``architecture`` block picks
the model and the builder; format 1 had none and is always the encoder with
markers, so older checkpoints keep loading.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModel

from .architecture import DecisionModel, build_model
from .calibration import TemperatureMap
from .model import ExuConfig
from .sequence import SequenceConfig

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

FORMAT_VERSION = 2
# Format 1 is format 2 without the architecture block.
_READABLE_VERSIONS = (1, 2)
METADATA_FILE = "exu.json"
WEIGHTS_FILE = "model.safetensors"
ENCODER_DIR = "encoder_config"
TOKENIZER_DIR = "tokenizer"

# The attribute that holds the pretrained network, per backbone.
_BACKBONE_PREFIX = {"encoder": "encoder.", "decoder": "backbone."}


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """A loaded checkpoint plus everything needed to run it."""

    path: Path
    model: DecisionModel
    model_config: ExuConfig
    sequence_config: SequenceConfig
    temperature: TemperatureMap


def save_checkpoint(
    path: str | Path,
    model: DecisionModel,
    *,
    sequence_config: SequenceConfig | None = None,
    temperature: TemperatureMap | None = None,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> Path:
    """Write everything needed for offline inference. Refuses an existing path."""
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"checkpoint path already exists: {target}")
    target.mkdir(parents=True)
    try:
        model.backbone.config.save_pretrained(target / ENCODER_DIR)
        _write_json(
            target / METADATA_FILE,
            {
                "format_version": FORMAT_VERSION,
                "architecture": model.config.architecture(),
                "model": model.config.to_dict(),
                "sequence": (sequence_config or SequenceConfig()).to_dict(),
                "temperature": (temperature or TemperatureMap()).to_dict(),
            },
        )
        save_file(
            {name: tensor.detach().contiguous() for name, tensor in model.state_dict().items()},
            target / WEIGHTS_FILE,
        )
        if tokenizer is not None:
            tokenizer.save_pretrained(target / TOKENIZER_DIR)
    except Exception:
        shutil.rmtree(target)
        raise
    return target


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> Checkpoint:
    """Rebuild a model from a folder without touching the network."""
    source = Path(path)
    metadata = _read_metadata(source)
    model_config = ExuConfig.from_dict(
        _mapping(metadata["model"], "model"),
        _mapping(metadata.get("architecture", {}), "architecture"),
    )
    sequence_config = SequenceConfig.from_dict(_mapping(metadata.get("sequence", {}), "sequence"))
    temperature = TemperatureMap.from_dict(_mapping(metadata.get("temperature", {}), "temperature"))

    weights = source / WEIGHTS_FILE
    encoder_dir = source / ENCODER_DIR
    if not weights.is_file():
        raise FileNotFoundError(f"checkpoint is missing {WEIGHTS_FILE}: {source}")
    if not encoder_dir.is_dir():
        raise FileNotFoundError(f"checkpoint is missing {ENCODER_DIR}/: {source}")

    encoder_config = AutoConfig.from_pretrained(encoder_dir, local_files_only=True)
    model = build_model(AutoModel.from_config(encoder_config), model_config)
    state = load_file(weights, device="cpu")
    _validate_weight_keys(state, model.state_dict(), model_config.backbone)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return Checkpoint(source, model, model_config, sequence_config, temperature)


def load_tokenizer(path: str | Path) -> PreTrainedTokenizerBase:
    """Load the tokenizer stored beside a checkpoint."""
    from transformers import AutoTokenizer

    tokenizer_dir = Path(path) / TOKENIZER_DIR
    if not tokenizer_dir.is_dir():
        raise FileNotFoundError(f"checkpoint has no {TOKENIZER_DIR}/ folder: {path}")
    return AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)


def _read_metadata(source: Path) -> dict:
    metadata_path = source / METADATA_FILE
    if not metadata_path.is_file():
        raise FileNotFoundError(f"not a checkpoint, missing {METADATA_FILE}: {source}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"malformed checkpoint metadata: {metadata_path}")
    if metadata.get("format_version") not in _READABLE_VERSIONS:
        raise ValueError(
            f"unsupported checkpoint format {metadata.get('format_version')!r}; "
            f"expected one of {_READABLE_VERSIONS}"
        )
    if metadata["format_version"] == 1 and "architecture" in metadata:
        raise ValueError("format 1 checkpoints have no architecture block")
    if metadata["format_version"] == 2 and "architecture" not in metadata:
        raise ValueError("format 2 checkpoints require an architecture block")
    if not isinstance(metadata.get("model"), dict):
        raise ValueError("checkpoint metadata has no model config")
    return metadata


def _validate_weight_keys(state: dict, expected: dict, backbone: str) -> None:
    """Fail with an actionable message when the weights are not this architecture.

    Weights of one architecture never load as the other: the head and the
    backbone live under different names, so the strict key comparison fails. The
    message says which architecture the weights look like, because an edited
    ``architecture`` block is the likely cause.
    """
    if set(state) != set(expected):
        missing = sorted(set(expected) - set(state))[:3]
        extra = sorted(set(state) - set(expected))[:3]
        hint = ""
        for other, prefix in _BACKBONE_PREFIX.items():
            if other != backbone and any(key.startswith(prefix) for key in state):
                hint = f"; they look like a {other} checkpoint, the metadata says {backbone}"
        raise ValueError(
            f"weights do not match the model architecture{hint}; missing={missing}, extra={extra}"
        )
    if not any(key.startswith(_BACKBONE_PREFIX[backbone]) for key in state):
        raise ValueError("weights contain no backbone tensors")


def _mapping(value: object, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"checkpoint field {name!r} must be an object")
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
