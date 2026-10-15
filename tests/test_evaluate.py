from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from helpers import SMOKE, tiny_encoder, tiny_tokenizer

from exu import evaluate, train
from exu.data import load_jsonl


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


def _tagged(tmp_path) -> Path:
    """The smoke file with a policy group on three test rows and none on the fourth."""
    groups = {"test-1": "a", "test-2": "a", "test-3": "b"}
    rows = [json.loads(line) for line in SMOKE.read_text(encoding="utf-8").splitlines() if line]
    for row in rows:
        if row["id"] in groups:
            row["metadata"] = {"policy": groups[row["id"]]}
            row["case_id"] = "case-" + groups[row["id"]]
    path = tmp_path / "tagged.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_groups_recompose_the_rows_and_keep_the_missing_ones(capsys, checkpoint, tmp_path) -> None:
    data = _tagged(tmp_path)

    report = _report(
        capsys, checkpoint, "--data", str(data), "--split", "test", "--group-by", "metadata.policy"
    )

    grouped = report["by_group"]["metadata.policy"]
    assert grouped["rows"] == report["metrics"]["count"] == 4
    assert {key: value["rows"] for key, value in grouped["groups"].items()} == {"a": 2, "b": 1}
    assert grouped["missing"]["rows"] == 1
    assert grouped["missing"]["metrics"]["count"] == 1
    counts = [group["metrics"]["count"] for group in grouped["groups"].values()]
    assert sum(counts) + grouped["missing"]["rows"] == grouped["rows"]


def test_each_group_carries_its_own_baselines(capsys, checkpoint, tmp_path) -> None:
    data = _tagged(tmp_path)

    report = _report(
        capsys, checkpoint, "--data", str(data), "--split", "test", "--group-by", "metadata.policy"
    )

    group = report["by_group"]["metadata.policy"]["groups"]["a"]
    assert group["baselines"]["uniform"]["count"] == 2
    assert group["baselines"]["prior"]["count"] == 2
    assert set(group["baselines"]["majority"]) == {"accuracy"}


def test_grouping_without_a_reference_keeps_the_prior_unavailable(capsys, checkpoint) -> None:
    report = _report(capsys, checkpoint, "--data", str(SMOKE), "--group-by", "family")

    for group in report["by_group"]["family"]["groups"].values():
        assert group["baselines"]["prior"] is None
        assert group["baselines"]["majority"] is None


def test_a_field_nobody_has_is_one_missing_group(capsys, checkpoint) -> None:
    report = _report(
        capsys, checkpoint, "--data", str(SMOKE), "--split", "test", "--group-by", "metadata.none"
    )

    grouped = report["by_group"]["metadata.none"]
    assert grouped["groups"] == {}
    assert grouped["missing"]["rows"] == 4


@pytest.mark.parametrize("field", ["policy", "metadata.", "case_id", "Family"])
def test_an_unknown_group_field_is_an_error(capsys, checkpoint, field) -> None:
    with pytest.raises(SystemExit) as raised:
        evaluate.main(["--checkpoint", str(checkpoint), "--data", str(SMOKE), "--group-by", field])

    assert raised.value.code == 2
    assert "unknown group field" in capsys.readouterr().err


def test_the_case_count_says_when_it_does_not_know(capsys, checkpoint, tmp_path) -> None:
    plain = _report(capsys, checkpoint, "--data", str(SMOKE), "--split", "test")["cases"]
    tagged = _report(capsys, checkpoint, "--data", str(_tagged(tmp_path)), "--split", "test")

    assert plain == {"rows": 4, "rows_with_case_id": 0, "distinct_cases": None}
    assert tagged["cases"] == {"rows": 4, "rows_with_case_id": 3, "distinct_cases": 2}


def test_baseline_forecasts_do_not_read_the_scored_targets(tmp_path) -> None:
    data = _tagged(tmp_path)
    examples = load_jsonl(data, "test")
    flipped = [replace(example, target=example.target[::-1]) for example in examples]
    arguments = {"reference": data, "reference_split": "train", "data": data, "split": "test"}

    _, before = evaluate._baselines(examples, **arguments)
    _, after = evaluate._baselines(flipped, **arguments)

    assert set(before) == {"uniform", "prior", "majority"}
    for name, forecast in before.items():
        assert torch.equal(forecast, after[name]), name
