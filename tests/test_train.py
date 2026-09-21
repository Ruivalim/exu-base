from __future__ import annotations

import json

import pytest
from helpers import FIXTURES, SMOKE, tiny_encoder, tiny_tokenizer

from exu.checkpoint import load_checkpoint, load_tokenizer
from exu.train import _group_scale, build_parser, main


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
    assert summary["confident_miss_threshold"] == 1e-4
    (epoch,) = summary["train_epochs"]
    assert 0.0 <= epoch["confident_miss_rate"] <= 1.0
    assert epoch["candidate_confident_miss_rate"] is None
    assert summary["reward"] == {
        "version": 1,
        "log_term": "log_softmax",
        "spherical_weight": 0.75,
        "rps_weight": 1.0,
    }
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
    assert "log_floor" not in summary["policy"]
    for epoch in summary["train_epochs"]:
        assert 0.0 <= epoch["confident_miss_rate"] <= 1.0
        assert 0.0 <= epoch["candidate_confident_miss_rate"] <= 1.0
    assert summary["reward"]["log_term"] == "log_softmax"
    assert summary["reward"]["spherical_weight"] == summary["policy"]["spherical_weight"]
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


def test_log_every_accepts_zero_and_rejects_negatives() -> None:
    """The help text promises that 0 disables logging, so 0 has to parse."""
    import pytest

    parser = build_parser()
    base = ["--train", "t", "--validation", "v", "--output", "o"]

    assert parser.parse_args(base + ["--log-every", "0"]).log_every == 0
    assert parser.parse_args(base + ["--log-every", "5"]).log_every == 5
    with pytest.raises(SystemExit):
        parser.parse_args(base + ["--log-every", "-1"])


def test_a_trailing_accumulation_group_keeps_its_weight() -> None:
    """Regression: the last group of an epoch was divided by the full grad_accum."""
    assert _group_scale(4, 10) == 0.25
    assert _group_scale(4, 4) == 0.25
    assert _group_scale(4, 1) == 1.0


def test_sigma_reaches_sigma_end_on_the_last_update(tmp_path) -> None:
    """Regression: the span was total_steps, so the anneal stopped one step short."""
    encoder = _encoder_dir(tmp_path)
    output = tmp_path / "anneal"

    code = main(
        _common(tmp_path, encoder, output)
        + [
            "--mode",
            "rlcd",
            "--epochs",
            "2",
            "--grad-accum",
            "3",
            "--samples-per-question",
            "2",
            "--sigma-start",
            "0.8",
            "--sigma-end",
            "0.2",
        ]
    )

    assert code == 0
    summary = json.loads((output / "training.json").read_text(encoding="utf-8"))
    assert summary["train_epochs"][-1]["sigma_end"] == pytest.approx(0.2)


def test_the_scorer_choice_is_recorded_and_reloads(tmp_path) -> None:
    encoder = _encoder_dir(tmp_path)
    output = tmp_path / "marker-cls"
    arguments = _common(tmp_path, encoder, output)

    code = main([*arguments, "--mode", "baseline", "--epochs", "1", "--scorer", "marker-cls"])

    assert code == 0
    summary = json.loads((output / "training.json").read_text(encoding="utf-8"))
    assert summary["scorer"] == "marker-cls"
    restored = load_checkpoint(output)
    assert restored.model_config.scorer == "marker-cls"
    assert any(key.startswith("option_readout") for key in restored.model.state_dict())


def test_the_default_scorer_is_the_marker_one(tmp_path) -> None:
    assert build_parser().get_default("scorer") == "marker"
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["--train", "a", "--validation", "b", "--output", "c", "--scorer", "x"]
        )
