from __future__ import annotations

import json

import pytest
import torch
from helpers import save_tiny_checkpoint, tiny_model, tiny_tokenizer

from rlcd import SequenceConfig, TemperatureMap
from rlcd.checkpoint import load_checkpoint, load_tokenizer, save_checkpoint


def test_round_trip_preserves_every_weight(tmp_path) -> None:
    model = tiny_model()
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(
        checkpoint,
        model,
        sequence_config=SequenceConfig(max_length=256, header_budget=128),
        temperature=TemperatureMap(default=1.0, by_type={"choice": 2.0}),
        tokenizer=tiny_tokenizer(),
    )

    loaded = load_checkpoint(checkpoint)

    for key, value in model.state_dict().items():
        assert torch.equal(value, loaded.model.state_dict()[key]), key
    assert loaded.sequence_config.max_length == 256
    assert loaded.temperature.temperature("choice", 2) == 2.0
    assert loaded.model.training is False


def test_checkpoint_refuses_an_existing_directory(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()

    with pytest.raises(FileExistsError):
        save_checkpoint(checkpoint, tiny_model())


def test_metadata_records_format_version_and_configs(tmp_path) -> None:
    checkpoint = save_tiny_checkpoint(tmp_path / "checkpoint")

    metadata = json.loads((checkpoint / "rlcd.json").read_text(encoding="utf-8"))

    assert metadata["format_version"] == 1
    assert metadata["model"]["encoder_name"] == "tiny-bert"
    assert metadata["model"]["action_costs"]["loss"] == 3.0
    assert "temperature" in metadata
    assert "sequence" in metadata


def test_load_rejects_an_unsupported_format_version(tmp_path) -> None:
    checkpoint = save_tiny_checkpoint(tmp_path / "checkpoint")
    metadata_path = checkpoint / "rlcd.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["format_version"] = 99
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported checkpoint format"):
        load_checkpoint(checkpoint)


def test_load_reports_a_missing_weights_file(tmp_path) -> None:
    checkpoint = save_tiny_checkpoint(tmp_path / "checkpoint")
    (checkpoint / "model.safetensors").unlink()

    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        load_checkpoint(checkpoint)


def test_load_rejects_weights_for_a_different_architecture(tmp_path) -> None:
    checkpoint = save_tiny_checkpoint(tmp_path / "checkpoint")
    metadata_path = checkpoint / "rlcd.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["model"]["num_decision_layers"] = 3
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="weights do not match"):
        load_checkpoint(checkpoint)


def test_load_rejects_a_directory_that_is_not_a_checkpoint(tmp_path) -> None:
    (tmp_path / "empty").mkdir()

    with pytest.raises(FileNotFoundError, match="rlcd.json"):
        load_checkpoint(tmp_path / "empty")


def test_tokenizer_round_trips_for_offline_inference(tmp_path) -> None:
    checkpoint = save_tiny_checkpoint(tmp_path / "checkpoint")

    tokenizer = load_tokenizer(checkpoint)

    assert tokenizer.mask_token_id == tiny_tokenizer().mask_token_id


def test_load_tokenizer_reports_a_missing_folder(tmp_path) -> None:
    model = tiny_model()
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model)

    with pytest.raises(FileNotFoundError, match="tokenizer"):
        load_tokenizer(checkpoint)
