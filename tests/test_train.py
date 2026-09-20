from __future__ import annotations

import json

from helpers import FIXTURES, SMOKE, tiny_encoder, tiny_tokenizer

from exu.checkpoint import load_checkpoint, load_tokenizer
from exu.train import build_parser, main


def _encoder_dir(tmp_path):
    directory = tmp_path / "tiny-encoder"
    tiny_tokenizer().save_pretrained(directory)
    tiny_encoder().save_pretrained(directory)
    return directory


def _common(tmp_path, encoder, output):
    return [
        "--train",
        str(SMOKE),
        "--train-split",
        "train",
        "--validation",
        str(SMOKE),
        "--validation-split",
        "validation",
        "--output",
        str(output),
        "--encoder",
        str(encoder),
        "--batch-size",
        "4",
        "--device",
        "cpu",
    ]


def test_parser_defaults_to_rlcd_and_no_calibration() -> None:
    parser = build_parser()

    assert parser.get_default("mode") == "rlcd"
    assert parser.get_default("calibrate") is False
    assert parser.get_default("encoder")


def test_baseline_mode_writes_a_reloadable_checkpoint(tmp_path) -> None:
    encoder = _encoder_dir(tmp_path)
    output = tmp_path / "baseline"

    code = main(_common(tmp_path, encoder, output) + ["--mode", "baseline", "--epochs", "1"])

    assert code == 0
    summary = json.loads((output / "training.json").read_text(encoding="utf-8"))
    assert summary["mode"] == "baseline"
    assert len(summary["train_epochs"]) == 1
    assert summary["validation"]["count"] == 4
    assert summary["policy"] is None
    restored = load_checkpoint(output)
    assert restored.model_config.encoder_name == str(encoder)
    assert load_tokenizer(output).mask_token_id == tiny_tokenizer().mask_token_id


def test_rlcd_mode_trains_calibrates_and_shuffles(tmp_path) -> None:
    encoder = _encoder_dir(tmp_path)
    output = tmp_path / "rlcd"

    code = main(
        _common(tmp_path, encoder, output)
        + [
            "--mode",
            "rlcd",
            "--epochs",
            "2",
            "--samples-per-question",
            "4",
            "--option-shuffle",
            "--calibrate",
            "--calibration",
            str(SMOKE),
            "--calibration-split",
            "calibration",
            "--calibration-min-samples",
            "2",
        ]
    )

    assert code == 0
    summary = json.loads((output / "training.json").read_text(encoding="utf-8"))
    assert summary["option_shuffle"] is True
    assert summary["policy"]["samples_per_question"] == 4
    assert all(epoch["mean_reward"] is not None for epoch in summary["train_epochs"])
    assert summary["calibration"]["source"].endswith("smoke.jsonl")
    assert summary["calibration"]["fits"]
    assert "validation_after" in summary["calibration"]
    restored = load_checkpoint(output)
    assert restored.temperature.default > 0


def test_train_refuses_an_existing_output(tmp_path) -> None:
    encoder = _encoder_dir(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()

    try:
        main(_common(tmp_path, encoder, output) + ["--epochs", "1"])
    except FileExistsError as error:
        assert "already exists" in str(error)
    else:  # pragma: no cover
        raise AssertionError("expected FileExistsError")


def test_train_validates_numeric_arguments() -> None:
    parser = build_parser()

    import pytest

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--train",
                str(FIXTURES / "x"),
                "--validation",
                str(FIXTURES / "y"),
                "--output",
                "o",
                "--epochs",
                "0",
            ]
        )
