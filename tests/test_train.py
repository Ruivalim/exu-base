from __future__ import annotations

import json

import pytest
from helpers import FIXTURES, SMOKE, tiny_encoder, tiny_tokenizer

import exu.train as train_module
from exu.checkpoint import load_checkpoint, load_tokenizer
from exu.train import _group_scale, build_parser, lr_multiplier, main, resolve_defaults


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
    assert parser.get_default("backbone") == "encoder"


def test_backbone_defaults_follow_the_backbone() -> None:
    required = ["--train", "t", "--validation", "v", "--output", "o"]

    encoder = resolve_defaults(build_parser().parse_args(required))
    decoder = resolve_defaults(build_parser().parse_args([*required, "--backbone", "decoder"]))

    assert encoder.encoder == "google-bert/bert-base-multilingual-cased"
    assert encoder.encoder_lr == 2.5e-5
    assert (encoder.lr_schedule, encoder.warmup_steps) == ("constant", 0)
    assert decoder.encoder == "Qwen/Qwen3-0.6B"
    assert decoder.encoder_lr == 1e-5
    # Section 14 of BENCHMARKS.md: the decoder wants the decaying schedule, the
    # encoder showed no gain from it.
    assert (decoder.lr_schedule, decoder.warmup_steps) == ("cosine", 10)


def test_explicit_flags_win_over_backbone_defaults() -> None:
    args = build_parser().parse_args(
        ["--train", "t", "--validation", "v", "--output", "o", "--backbone", "decoder",
         "--encoder", "some/model", "--encoder-lr", "3e-6", "--header-budget", "64",
         "--lr-schedule", "constant", "--warmup-steps", "0"]
    )  # fmt: skip

    resolve_defaults(args)

    assert (args.encoder, args.encoder_lr, args.header_budget) == ("some/model", 3e-6, 64)
    # An explicit zero is a choice, not a missing value.
    assert (args.lr_schedule, args.warmup_steps) == ("constant", 0)


def test_header_budget_never_exceeds_a_short_max_length() -> None:
    args = build_parser().parse_args(
        ["--train", "t", "--validation", "v", "--output", "o", "--max-length", "128"]
    )

    assert resolve_defaults(args).header_budget == 128


def test_marker_cls_with_a_decoder_is_a_usage_error(tmp_path) -> None:
    with pytest.raises(SystemExit):
        main(
            ["--train", "t", "--validation", "v", "--output", str(tmp_path / "out"),
             "--backbone", "decoder", "--scorer", "marker-cls"]
        )  # fmt: skip


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


def test_the_constant_schedule_leaves_the_learning_rate_alone() -> None:
    assert all(lr_multiplier(update, 10, 0, "constant") == 1.0 for update in range(10))


def test_warmup_ramps_to_the_full_rate_without_a_zero_step() -> None:
    ramp = [lr_multiplier(update, 20, 4, "constant") for update in range(6)]

    assert ramp == pytest.approx([0.2, 0.4, 0.6, 0.8, 1.0, 1.0])


def test_cosine_decays_monotonically_and_the_last_update_still_moves() -> None:
    factors = [lr_multiplier(update, 12, 2, "cosine") for update in range(12)]

    after_warmup = factors[2:]
    assert after_warmup[0] == 1.0
    assert all(
        later <= earlier for earlier, later in zip(after_warmup, after_warmup[1:], strict=False)
    )
    assert 0.0 < after_warmup[-1] < 0.05
    # Half way through the decay is half the rate.
    assert lr_multiplier(7, 12, 2, "cosine") == pytest.approx(0.5)


def test_cosine_with_a_single_update_runs_it_at_the_full_rate() -> None:
    assert lr_multiplier(0, 1, 0, "cosine") == 1.0


def test_an_unknown_schedule_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown lr schedule"):
        lr_multiplier(0, 10, 0, "linear")
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["--train", "t", "--validation", "v", "--output", "o", "--lr-schedule", "linear"]
        )


def test_warmup_that_covers_every_update_is_a_usage_error(tmp_path) -> None:
    encoder = _encoder_dir(tmp_path)
    output = tmp_path / "warm"
    # 12 training rows, batch 4: three updates per epoch, three in all.
    with pytest.raises(SystemExit):
        main(
            _common(tmp_path, encoder, output)
            + ["--mode", "baseline", "--epochs", "1", "--warmup-steps", "3"]
        )
    assert not output.exists()


def test_the_schedule_sets_the_rate_of_every_update_in_both_groups(tmp_path, monkeypatch) -> None:
    encoder = _encoder_dir(tmp_path)
    output = tmp_path / "cosine"
    seen: list[tuple[float, float]] = []
    real_step = train_module._optimizer_step

    def recording_step(model, optimizer, scheduler, max_grad_norm):
        seen.append(tuple(group["lr"] for group in optimizer.param_groups))
        real_step(model, optimizer, scheduler, max_grad_norm)

    monkeypatch.setattr(train_module, "_optimizer_step", recording_step)
    code = main(
        _common(tmp_path, encoder, output)
        + ["--mode", "baseline", "--epochs", "2", "--grad-accum", "2",
           "--encoder-lr", "1e-4", "--head-lr", "1e-3",
           "--lr-schedule", "cosine", "--warmup-steps", "1"]
    )  # fmt: skip

    assert code == 0
    # Three micro-batches per epoch at --grad-accum 2: two updates per epoch, the
    # second one a trailing group of one.
    total = 4
    assert len(seen) == total
    for update, (encoder_lr, head_lr) in enumerate(seen):
        factor = lr_multiplier(update, total, 1, "cosine")
        assert encoder_lr == pytest.approx(1e-4 * factor)
        assert head_lr == pytest.approx(1e-3 * factor)
    summary = json.loads((output / "training.json").read_text(encoding="utf-8"))
    assert summary["lr_schedule"] == "cosine"
    assert summary["warmup_steps"] == 1
