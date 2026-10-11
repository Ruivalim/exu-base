from __future__ import annotations

import io
import json

import pytest
from helpers import SMOKE, smoke_examples, tiny_encoder, tiny_tokenizer

from exu import DecisionRuntime, Option, decide, train


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    root = tmp_path_factory.mktemp("decide")
    encoder = root / "tiny-encoder"
    tiny_tokenizer().save_pretrained(encoder)
    tiny_encoder().save_pretrained(encoder)
    output = root / "model"
    arguments = ["--mode", "baseline", "--epochs", "1", "--batch-size", "4", "--device", "cpu"]
    arguments += ["--train", str(SMOKE), "--train-split", "train"]
    arguments += ["--validation", str(SMOKE), "--validation-split", "validation"]
    assert train.main([*arguments, "--output", str(output), "--encoder", str(encoder)]) == 0
    return output


def _run(capsys, checkpoint, *arguments: str) -> tuple[int, list[dict], str]:
    code = decide.main(["--checkpoint", str(checkpoint), "--device", "cpu", *arguments])
    captured = capsys.readouterr()
    rows = [json.loads(line) for line in captured.out.splitlines() if line.strip()]
    return code, rows, captured.err


def _test_rows() -> list[str]:
    lines = SMOKE.read_text(encoding="utf-8").splitlines()
    return [line for line in lines if line.strip() and json.loads(line)["split"] == "test"]


def test_one_choice_question_from_flags(capsys, checkpoint) -> None:
    code, rows, _err = _run(
        capsys,
        checkpoint,
        "--state",
        "I was charged twice for the same invoice.",
        "--instruction",
        "Where should this ticket go?",
        "--option",
        "billing=payment, invoice or refund",
        "--option",
        "support",
    )

    assert code == 0
    (row,) = rows
    assert row["kind"] == "choice"
    assert list(row["probabilities"]) == ["billing", "support"]
    assert sum(row["probabilities"].values()) == pytest.approx(1.0, abs=1e-5)
    assert row["label"] in row["probabilities"]
    assert row["confidence"] == pytest.approx(max(row["probabilities"].values()))
    assert row["expected_level"] is None
    assert isinstance(row["should_act"], bool)
    assert set(row["logits"]) == set(row["probabilities"])


def test_an_option_splits_at_the_first_equals_sign_only() -> None:
    assert decide.parse_option("billing=refund = money back") == Option(
        "billing", "refund = money back"
    )
    assert decide.parse_option("support") == Option("support")
    with pytest.raises(ValueError, match="name"):
        decide.parse_option("=no name")


def test_a_score_question_reports_its_expected_level(capsys, checkpoint) -> None:
    code, rows, _err = _run(
        capsys,
        checkpoint,
        "--state",
        "The answer was slow but correct.",
        "--kind",
        "score",
        "--instruction",
        "How good was the support?",
        "--level",
        "bad",
        "--level",
        "fine",
        "--level",
        "great",
    )

    assert code == 0
    assert rows[0]["kind"] == "score"
    assert 0.0 <= rows[0]["expected_level"] <= 2.0


def test_a_noul_question_defaults_to_no_and_yes(capsys, checkpoint) -> None:
    code, rows, _err = _run(
        capsys,
        checkpoint,
        "--state",
        "Refund requested.",
        "--kind",
        "noul",
        "--instruction",
        "Urgent?",
    )

    assert code == 0
    assert list(rows[0]["probabilities"]) == ["No", "Yes"]


def test_a_batch_keeps_order_ids_and_matches_the_python_api(capsys, checkpoint, tmp_path) -> None:
    # Dataset records work as they are: the target and the split are ignored.
    source = tmp_path / "questions.jsonl"
    source.write_text("\n\n".join(_test_rows()) + "\n", encoding="utf-8")
    examples = smoke_examples("test")

    code, rows, _err = _run(capsys, checkpoint, "--input", str(source), "--batch-size", "3")

    assert code == 0
    assert [row["id"] for row in rows] == [example.example_id for example in examples]
    runtime = DecisionRuntime.load(checkpoint, device="cpu")
    for row, example in zip(rows, examples, strict=True):
        expected = runtime.decide(example.state, example.question)
        assert row["label"] == expected.label
        assert list(row["probabilities"].values()) == pytest.approx(
            list(expected.probabilities), abs=1e-5
        )


def test_the_batch_size_does_not_change_an_answer(capsys, checkpoint, tmp_path) -> None:
    source = tmp_path / "questions.jsonl"
    source.write_text("\n".join(_test_rows()) + "\n", encoding="utf-8")

    _code, one_by_one, _err = _run(capsys, checkpoint, "--input", str(source), "--batch-size", "1")
    _code, together, _err = _run(capsys, checkpoint, "--input", str(source), "--batch-size", "16")

    for alone, batched in zip(one_by_one, together, strict=True):
        assert list(alone["probabilities"].values()) == pytest.approx(
            list(batched["probabilities"].values()), abs=1e-5
        )


def test_questions_can_arrive_on_stdin(capsys, checkpoint, monkeypatch) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(_test_rows()[0] + "\n"))

    code, rows, _err = _run(capsys, checkpoint, "--input", "-")

    assert code == 0
    assert len(rows) == 1


def test_output_goes_to_a_file_when_asked(capsys, checkpoint, tmp_path) -> None:
    source = tmp_path / "questions.jsonl"
    source.write_text("\n".join(_test_rows()) + "\n", encoding="utf-8")
    destination = tmp_path / "decisions.jsonl"

    code, rows, _err = _run(
        capsys, checkpoint, "--input", str(source), "--output", str(destination)
    )

    assert code == 0
    assert rows == []
    assert len(destination.read_text(encoding="utf-8").splitlines()) == len(_test_rows())


def test_a_record_without_an_id_is_still_answered(capsys, checkpoint, tmp_path) -> None:
    record = json.loads(_test_rows()[0])
    source = tmp_path / "bare.jsonl"
    source.write_text(
        json.dumps({"state": record["state"], "question": record["question"]}) + "\n",
        encoding="utf-8",
    )

    code, rows, _err = _run(capsys, checkpoint, "--input", str(source))

    assert code == 0
    assert "id" not in rows[0]


@pytest.mark.parametrize(
    ("second_line", "expected"),
    [
        ("{not json", "line 2"),
        ('{"state": "only a state"}', "line 2"),
        (
            '{"state": "", "question": {"kind": "noul", "instruction": "?", "options": []}}',
            "line 2",
        ),
        ('["a", "list"]', "line 2"),
    ],
)
def test_a_broken_line_is_named_and_nothing_is_half_answered(
    capsys, checkpoint, tmp_path, second_line: str, expected: str
) -> None:
    source = tmp_path / "broken.jsonl"
    source.write_text(_test_rows()[0] + "\n" + second_line + "\n", encoding="utf-8")

    code, rows, err = _run(capsys, checkpoint, "--input", str(source))

    assert code == 1
    assert expected in err
    assert rows == []


def test_an_empty_input_is_an_error(capsys, checkpoint, tmp_path) -> None:
    source = tmp_path / "empty.jsonl"
    source.write_text("\n\n", encoding="utf-8")

    code, _rows, err = _run(capsys, checkpoint, "--input", str(source))

    assert code == 1
    assert "no questions" in err


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--input", "x.jsonl", "--state", "both modes"],
        ["--state", "s", "--instruction", "pick", "--option", "only one"],
        [
            "--state",
            "s",
            "--kind",
            "score",
            "--instruction",
            "rate",
            "--option",
            "a",
            "--option",
            "b",
        ],
        [
            "--state",
            "s",
            "--kind",
            "choice",
            "--instruction",
            "pick",
            "--level",
            "a",
            "--level",
            "b",
        ],
        ["--state", "s", "--option", "a", "--option", "b"],
        ["--instruction", "pick", "--option", "a", "--option", "b"],
    ],
)
def test_a_malformed_command_line_is_refused_before_loading_anything(arguments) -> None:
    with pytest.raises(SystemExit) as error:
        decide.main(["--checkpoint", "does-not-exist", *arguments])

    assert error.value.code == 2


def test_a_missing_checkpoint_is_a_clean_error(capsys, tmp_path) -> None:
    code = decide.main(
        [
            "--checkpoint",
            str(tmp_path / "nope"),
            "--state",
            "s",
            "--kind",
            "noul",
            "--instruction",
            "?",
        ]
    )

    assert code == 1
    assert "error:" in capsys.readouterr().err


def _questions_file(tmp_path):
    source = tmp_path / "questions.jsonl"
    source.write_text("\n".join(_test_rows()) + "\n", encoding="utf-8")
    return source


def test_timing_is_off_unless_asked(capsys, checkpoint, tmp_path) -> None:
    code, rows, err = _run(capsys, checkpoint, "--input", str(_questions_file(tmp_path)))

    assert code == 0
    assert all("metrics" not in row for row in rows)
    assert err == ""


def test_metrics_time_each_batch_and_summarise_on_stderr(
    capsys, checkpoint, tmp_path, monkeypatch
) -> None:
    # A clock that advances half a second per reading makes every number exact:
    # one reading before and one after the load, then one pair per batch.
    ticks = iter(index * 0.5 for index in range(100))
    monkeypatch.setattr(decide.time, "perf_counter", lambda: next(ticks))

    code, rows, err = _run(
        capsys,
        checkpoint,
        "--input",
        str(_questions_file(tmp_path)),
        "--batch-size",
        "3",
        "--metrics",
    )

    assert code == 0
    assert [row["metrics"]["batch_size"] for row in rows] == [3, 3, 3, 1]
    assert all(row["metrics"]["batch_ms"] == pytest.approx(500.0) for row in rows)
    assert rows[0]["metrics"]["per_question_ms"] == pytest.approx(500.0 / 3)
    assert rows[3]["metrics"]["per_question_ms"] == pytest.approx(500.0)
    summary = json.loads(err)
    assert summary == {
        "questions": 4,
        "batches": 2,
        "load_ms": pytest.approx(500.0),
        "inference_ms": pytest.approx(1000.0),
        "first_batch_ms": pytest.approx(500.0),
        "questions_per_second": pytest.approx(4.0),
    }


def test_real_timings_are_positive(capsys, checkpoint, tmp_path) -> None:
    _code, rows, err = _run(
        capsys, checkpoint, "--input", str(_questions_file(tmp_path)), "--metrics"
    )

    assert all(row["metrics"]["batch_ms"] > 0 for row in rows)
    assert json.loads(err)["load_ms"] > 0


def test_top_only_keeps_the_winner_and_its_probability(capsys, checkpoint, tmp_path) -> None:
    source = _questions_file(tmp_path)

    _code, full, _err = _run(capsys, checkpoint, "--input", str(source))
    code, short, _err = _run(capsys, checkpoint, "--input", str(source), "--top-only")

    assert code == 0
    for whole, brief in zip(full, short, strict=True):
        assert set(brief) == {"id", "label", "confidence"}
        assert (brief["id"], brief["label"]) == (whole["id"], whole["label"])
        assert brief["confidence"] == pytest.approx(whole["confidence"])


def test_top_only_and_metrics_combine(capsys, checkpoint) -> None:
    code, rows, err = _run(
        capsys,
        checkpoint,
        "--state",
        "Refund requested.",
        "--kind",
        "noul",
        "--instruction",
        "Urgent?",
        "--top-only",
        "--metrics",
    )

    assert code == 0
    assert set(rows[0]) == {"label", "confidence", "metrics"}
    assert json.loads(err)["questions"] == 1


def _questions_record(identifier: str | None, ids: list[str | None]) -> dict:
    """A record with several questions about the state of the first test row."""
    rows = [json.loads(line) for line in _test_rows()]
    record: dict = {"state": rows[0]["state"], "questions": []}
    if identifier is not None:
        record["id"] = identifier
    for own, row in zip(ids, rows, strict=False):
        question = dict(row["question"])
        if own is not None:
            question["id"] = own
        record["questions"].append(question)
    return record


def test_a_record_of_questions_answers_each_like_separate_records(
    capsys, checkpoint, tmp_path
) -> None:
    record = _questions_record("case-1", ["route", None, "urgency"])
    source = tmp_path / "questions.jsonl"
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")

    code, rows, _err = _run(capsys, checkpoint, "--input", str(source))

    assert code == 0
    assert [row["id"] for row in rows] == ["case-1/route", "case-1/1", "case-1/urgency"]
    separate = tmp_path / "separate.jsonl"
    separate.write_text(
        "\n".join(
            json.dumps({"state": record["state"], "question": q}) for q in record["questions"]
        )
        + "\n",
        encoding="utf-8",
    )
    _code, expected, _err = _run(capsys, checkpoint, "--input", str(separate))
    for row, alone in zip(rows, expected, strict=True):
        assert list(row["probabilities"].values()) == pytest.approx(
            list(alone["probabilities"].values()), abs=1e-5
        )


def test_without_a_record_id_only_the_question_ids_name_the_answers(
    capsys, checkpoint, tmp_path
) -> None:
    source = tmp_path / "questions.jsonl"
    source.write_text(json.dumps(_questions_record(None, ["a", None])) + "\n", encoding="utf-8")

    code, rows, _err = _run(capsys, checkpoint, "--input", str(source))

    assert code == 0
    assert rows[0]["id"] == "a"
    assert "id" not in rows[1]


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (lambda record: record.update(question=record["questions"][0]), "not both"),
        (lambda record: record.update(questions=[]), "non-empty list"),
        (lambda record: record.update(questions={"kind": "noul"}), "non-empty list"),
        (lambda record: record["questions"][1].update(id="route"), "appears twice"),
        (lambda record: record["questions"][1].pop("instruction"), "question 1"),
        (lambda record: record["questions"].append("not a question"), "question 3"),
    ],
)
def test_a_broken_record_of_questions_is_named_and_nothing_is_answered(
    capsys, checkpoint, tmp_path, change, expected: str
) -> None:
    record = _questions_record("case-1", ["route", None, "urgency"])
    change(record)
    source = tmp_path / "broken.jsonl"
    source.write_text(_test_rows()[0] + "\n" + json.dumps(record) + "\n", encoding="utf-8")

    code, rows, err = _run(capsys, checkpoint, "--input", str(source))

    assert code == 1
    assert "line 2" in err
    assert expected in err
    assert rows == []


@pytest.mark.parametrize(
    ("sizes", "batch", "expected"),
    [
        ([2, 2, 2], 3, [2, 2, 2]),  # a record that would straddle two batches waits
        ([2, 2, 2], 4, [4, 2]),
        # A record larger than a batch is split, and its rest can share the next batch.
        ([1, 5, 1], 3, [1, 3, 3]),
        ([3], 3, [3]),
    ],
)
def test_batches_keep_a_record_together_when_it_fits(sizes, batch, expected) -> None:
    records = [
        [(f"{index}/{n}", "state", None) for n in range(size)] for index, size in enumerate(sizes)
    ]

    chunks = list(decide._chunks(records, batch))

    assert [len(chunk) for chunk in chunks] == expected
    flat = [question for record in records for question in record]
    assert [question for chunk in chunks for question in chunk] == flat


def test_decide_all_answers_like_decide_many(checkpoint) -> None:
    runtime = DecisionRuntime.load(checkpoint, device="cpu")
    examples = smoke_examples("test")[:3]
    state = examples[0].state
    questions = [example.question for example in examples]

    together = runtime.decide_all(state, questions)
    alone = [runtime.decide(state, question) for question in questions]

    for left, right in zip(together, alone, strict=True):
        assert left.probabilities == pytest.approx(right.probabilities, abs=1e-5)
    with pytest.raises(ValueError, match="empty"):
        runtime.decide_all(state, [])
