from __future__ import annotations

import json

import pytest
import torch
from helpers import FakeTokenizer

from exu.data import DecisionDataset, TrainingExample, load_jsonl, write_jsonl
from exu.sequence import SequenceBuilder


def record(kind: str = "noul") -> dict[str, object]:
    options = [{"name": "No"}, {"name": "Yes"}]
    target = [0.2, 0.8]
    if kind == "score":
        options = [{"name": "Level 0"}, {"name": "Level 1"}, {"name": "Level 2"}]
        target = [0.1, 0.7, 0.2]
    return {
        "id": f"test-{kind}",
        "state": {"text": "action required"},
        "question": {"kind": kind, "instruction": "classify", "options": options},
        "target": target,
        "family": "routing",
        "language": "en",
    }


def test_structured_state_serializes_without_ascii_escapes() -> None:
    example = TrainingExample.from_record(record())

    assert example.state == '{"text":"action required"}'


def test_example_rejects_target_with_wrong_option_count() -> None:
    invalid = record()
    invalid["target"] = [1.0]

    with pytest.raises(ValueError, match="target length"):
        TrainingExample.from_record(invalid)


def test_example_rejects_a_target_that_does_not_sum_to_one() -> None:
    invalid = record()
    invalid["target"] = [0.2, 0.2]

    with pytest.raises(ValueError, match="sum to one"):
        TrainingExample.from_record(invalid)


def test_example_reports_a_missing_field_by_name() -> None:
    invalid = record()
    del invalid["state"]

    with pytest.raises(ValueError, match="missing required field 'state'"):
        TrainingExample.from_record(invalid)


def test_record_round_trip_preserves_structure_and_family() -> None:
    example = TrainingExample.from_record(record("score"))

    restored = TrainingExample.from_record(example.to_record())

    assert restored.state == '{"text":"action required"}'
    assert restored.family == "routing"
    assert restored.target == example.target
    assert [option.name for option in restored.question.options] == [
        "Level 0",
        "Level 1",
        "Level 2",
    ]


def test_a_string_state_that_looks_like_json_comes_back_structured() -> None:
    """The record format stores the state as a string, so that type is not kept.

    Documented rather than worked around: the text the model reads is identical
    either way, and the JSON guess is what makes a structured state come back
    whole.
    """
    example = TrainingExample.from_record(record() | {"state": '{"x":1}'})

    written = example.to_record()
    restored = TrainingExample.from_record(written)

    assert written["state"] == {"x": 1}
    assert restored.state == '{"x":1}'


def test_jsonl_loader_filters_an_explicit_split(tmp_path) -> None:
    train = record()
    train["split"] = "train"
    validation = record("score")
    validation["split"] = "validation"
    dataset = tmp_path / "split.jsonl"
    write_jsonl(
        path := dataset,
        [TrainingExample.from_record(train), TrainingExample.from_record(validation)],
    )

    examples = load_jsonl(path, split="validation")

    assert len(examples) == 1
    assert examples[0].question.kind.value == "score"
    assert examples[0].split == "validation"


def test_jsonl_loader_identifies_a_bad_line(tmp_path) -> None:
    dataset = tmp_path / "bad.jsonl"
    dataset.write_text(json.dumps(record()) + "\nnot json\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 2"):
        load_jsonl(dataset)


def test_jsonl_loader_reports_an_empty_selection(tmp_path) -> None:
    dataset = tmp_path / "empty.jsonl"
    write_jsonl(dataset, [TrainingExample.from_record(record())])

    with pytest.raises(ValueError, match="no examples"):
        load_jsonl(dataset, split="test")


def test_collator_pads_targets_and_marks_ordinal_rows() -> None:
    dataset = DecisionDataset(
        [TrainingExample.from_record(record()), TrainingExample.from_record(record("score"))],
        SequenceBuilder(FakeTokenizer()),
    )

    batch = dataset.collate([dataset[0], dataset[1]])

    assert batch.targets.shape == (2, 3)
    assert batch.targets[0, 2].item() == 0
    assert batch.ordinal_mask.tolist() == [False, True]
    assert batch.option_counts.tolist() == [2, 3]
    assert batch.question_kinds == ("noul", "score")
    assert batch.question_families == ("routing", "routing")
    assert set(batch.model_inputs()) == {
        "input_ids",
        "attention_mask",
        "marker_positions",
        "option_mask",
        "question_type_ids",
    }


def test_dataset_applies_a_transform_per_access() -> None:
    calls: list[str] = []

    def transform(example: TrainingExample) -> TrainingExample:
        calls.append(example.example_id)
        return example

    dataset = DecisionDataset(
        [TrainingExample.from_record(record())],
        SequenceBuilder(FakeTokenizer()),
        transform=transform,
    )

    _ = dataset[0]
    _ = dataset[0]

    assert calls == ["test-noul", "test-noul"]


def test_dataset_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        DecisionDataset([], SequenceBuilder(FakeTokenizer()))


def test_a_record_without_provenance_loads_as_before() -> None:
    example = TrainingExample.from_record(record())

    assert example.case_id is None
    assert dict(example.metadata) == {}
    assert "case_id" not in example.to_record()
    assert "metadata" not in example.to_record()


def test_case_id_and_metadata_survive_a_round_trip(tmp_path) -> None:
    raw = record() | {"case_id": "case-7", "metadata": {"policy_family_id": "refunds"}}
    path = tmp_path / "provenance.jsonl"

    write_jsonl(path, [TrainingExample.from_record(raw)])
    (restored,) = load_jsonl(path)

    assert restored.case_id == "case-7"
    assert dict(restored.metadata) == {"policy_family_id": "refunds"}


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({"rank": 3}, "must be a string"),
        ({"tags": ["a"]}, "must be a string"),
        ({"": "x"}, "non-empty strings"),
        (["policy"], "must be an object"),
        ("policy", "must be an object"),
    ],
)
def test_metadata_outside_the_string_map_contract_is_refused(metadata, message) -> None:
    with pytest.raises(ValueError, match=message):
        TrainingExample.from_record(record() | {"metadata": metadata})


def test_an_empty_case_id_is_refused() -> None:
    with pytest.raises(ValueError, match="case_id"):
        TrainingExample.from_record(record() | {"case_id": " "})


def test_metadata_cannot_be_changed_behind_the_frozen_example() -> None:
    source = {"policy_family_id": "refunds"}
    example = TrainingExample.from_record(record() | {"metadata": source})

    source["policy_family_id"] = "changed"

    assert example.metadata["policy_family_id"] == "refunds"
    with pytest.raises(TypeError):
        example.metadata["policy_family_id"] = "changed"  # type: ignore[index]


def test_provenance_never_reaches_the_model_input() -> None:
    builder = SequenceBuilder(FakeTokenizer())
    plain = TrainingExample.from_record(record())
    tagged = TrainingExample.from_record(
        record() | {"case_id": "case-7", "metadata": {"policy": "action required"}}
    )

    batches = [
        DecisionDataset([example], builder).collate([example]) for example in (plain, tagged)
    ]

    for name, tensor in batches[0].inputs.items():
        assert torch.equal(tensor, batches[1].inputs[name]), name
