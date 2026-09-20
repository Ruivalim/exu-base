"""Portable checkpoint: config, weights, tokenizer and temperatures in one folder.

Layout::

    checkpoint/
      rlcd.json            # format version, model config, sequence config, temperatures
      model.safetensors    # every weight, encoder included, one file
      encoder_config/      # architecture only, so loading never downloads weights
      tokenizer/           # tokenizer files, so inference is offline

Loading validates before it loads: format version, required keys, expected weight
prefixes, strict state dict. A clear error beats a silent wrong answer when
someone points at the wrong folder.
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

from .calibration import TemperatureMap
from .model import RLCDConfig, RLCDModel
from .sequence import SequenceConfig

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

FORMAT_VERSION = 1
METADATA_FILE = "rlcd.json"
WEIGHTS_FILE = "model.safetensors"
ENCODER_DIR = "encoder_config"
TOKENIZER_DIR = "tokenizer"

_REQUIRED_WEIGHT_PREFIXES = (
    "encoder.",
    "question_type_embedding.",
    "decision_layers.",
    "scorer.",
    "action_head.",
)


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """A loaded checkpoint plus everything needed to run it."""

    path: Path
    model: RLCDModel
    model_config: RLCDConfig
    sequence_config: SequenceConfig
    temperature: TemperatureMap


def save_checkpoint(
    path: str | Path,
    model: RLCDModel,
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
        model.encoder.config.save_pretrained(target / ENCODER_DIR)
        _write_json(
            target / METADATA_FILE,
            {
                "format_version": FORMAT_VERSION,
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
    model_config = RLCDConfig.from_dict(_mapping(metadata["model"], "model"))
    sequence_config = SequenceConfig.from_dict(_mapping(metadata.get("sequence", {}), "sequence"))
    temperature = TemperatureMap.from_dict(_mapping(metadata.get("temperature", {}), "temperature"))

    weights = source / WEIGHTS_FILE
    encoder_dir = source / ENCODER_DIR
    if not weights.is_file():
        raise FileNotFoundError(f"checkpoint is missing {WEIGHTS_FILE}: {source}")
    if not encoder_dir.is_dir():
        raise FileNotFoundError(f"checkpoint is missing {ENCODER_DIR}/: {source}")

    encoder_config = AutoConfig.from_pretrained(encoder_dir, local_files_only=True)
    model = RLCDModel(AutoModel.from_config(encoder_config), model_config)
    state = load_file(weights, device="cpu")
    _validate_weight_keys(state, model.state_dict())
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
    if metadata.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"unsupported checkpoint format {metadata.get('format_version')!r}; "
            f"expected {FORMAT_VERSION}"
        )
    if not isinstance(metadata.get("model"), dict):
        raise ValueError("checkpoint metadata has no model config")
    return metadata


def _validate_weight_keys(state: dict, expected: dict) -> None:
    """Fail with an actionable message when the weights are not an RLCD model."""
    if set(state) != set(expected):
        missing = sorted(set(expected) - set(state))[:3]
        extra = sorted(set(state) - set(expected))[:3]
        raise ValueError(
            f"weights do not match the model architecture; missing={missing}, extra={extra}"
        )
    if not any(key.startswith("encoder.") for key in state):
        raise ValueError("weights contain no encoder tensors")


def _mapping(value: object, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"checkpoint field {name!r} must be an object")
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
