from __future__ import annotations

import json

import pytest
from helpers import SMOKE, tiny_encoder, tiny_tokenizer

from exu import evaluate, train


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    root = tmp_path_factory.mktemp("evaluate")
    encoder = root / "tiny-encoder"
    tiny_tokenizer().save_pretrained(encoder)
    tiny_encoder().save_pretrained(encoder)
    output = root / "baseline"
    arguments = ["--mode", "baseline", "--epochs", "1", "--batch-size", "4", "--device", "cpu"]
    arguments += ["--train", str(SMOKE), "--train-split", "train"]
    arguments += ["--validation", str(SMOKE), "--validation-split", "validation"]
    assert train.main([*arguments, "--output", str(output), "--encoder", str(encoder)]) == 0
    return output


def _report(capsys, checkpoint, *extra: str) -> dict:
    code = evaluate.main(["--checkpoint", str(checkpoint), "--device", "cpu", *extra])
    assert code == 0
    return json.loads(capsys.readouterr().out)


def test_the_prior_is_fitted_on_the_training_split_by_default(capsys, checkpoint) -> None:
    baselines = _report(capsys, checkpoint, "--data", str(SMOKE), "--split", "test")["baselines"]

    assert baselines["reference"]["split"] == "train"
    assert baselines["reference"]["rows"] == 12
    assert baselines["reference"]["unavailable"] is None
    assert (baselines["reference"]["seen_rows"], baselines["reference"]["unseen_rows"]) == (4, 0)
    assert baselines["prior"]["nll"] == pytest.approx(0.7230, abs=1e-4)
    assert baselines["prior_in_sample"]["nll"] == pytest.approx(0.3466, abs=1e-4)
    assert baselines["uniform"]["nll"] == pytest.approx(0.7945, abs=1e-4)
    assert "random" not in baselines


def test_majority_reports_accuracy_and_nothing_else(capsys, checkpoint) -> None:
    baselines = _report(capsys, checkpoint, "--data", str(SMOKE), "--split", "test")["baselines"]

    assert set(baselines["majority"]) == {"accuracy"}


@pytest.mark.parametrize("split", [("--split", "train"), ()])
def test_the_scored_rows_are_never_their_own_reference(capsys, checkpoint, split) -> None:
    baselines = _report(capsys, checkpoint, "--data", str(SMOKE), *split)["baselines"]

    assert baselines["prior"] is None
    assert baselines["majority"] is None
    assert "being scored" in baselines["reference"]["unavailable"]
    assert baselines["uniform"]["count"] > 0


def test_a_file_without_a_reference_split_says_so(capsys, checkpoint, tmp_path) -> None:
    only_test = tmp_path / "only-test.jsonl"
    lines = [line for line in SMOKE.read_text(encoding="utf-8").splitlines() if '"test"' in line]
    only_test.write_text("\n".join(lines) + "\n", encoding="utf-8")

    missing = _report(capsys, checkpoint, "--data", str(only_test), "--split", "test")
    supplied = _report(
        capsys, checkpoint, "--data", str(only_test), "--split", "test", "--reference", str(SMOKE)
    )

    assert missing["baselines"]["prior"] is None
    assert "no examples" in missing["baselines"]["reference"]["unavailable"]
    assert supplied["baselines"]["prior"]["nll"] == pytest.approx(0.7230, abs=1e-4)


def test_unseen_questions_are_counted_and_scored_apart(capsys, checkpoint, tmp_path) -> None:
    rows = [json.loads(line) for line in SMOKE.read_text(encoding="utf-8").splitlines() if line]
    for row in rows:
        if row["split"] == "test" and row["question"]["kind"] == "choice":
            row["question"]["instruction"] += " (never seen in training)"
    changed = tmp_path / "changed.jsonl"
    changed.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    baselines = _report(capsys, checkpoint, "--data", str(changed), "--split", "test")["baselines"]

    unseen = baselines["reference"]["unseen_rows"]
    assert unseen > 0
    assert baselines["reference"]["seen_rows"] + unseen == 4
    assert baselines["prior_unseen"]["count"] == unseen
    assert baselines["prior_seen"]["count"] == 4 - unseen
