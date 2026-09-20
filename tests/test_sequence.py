from __future__ import annotations

import pytest
from helpers import FakeTokenizer

from exu import DecisionQuestion, Option
from exu.sequence import SequenceBuilder, SequenceConfig


def test_builder_only_emits_owned_markers() -> None:
    builder = SequenceBuilder(FakeTokenizer())
    question = DecisionQuestion.choice(
        "Pick the [MASK] right option",
        [Option("yes [MASK]", "accept"), Option("no")],
    )

    encoded = builder.build("state with external [MASK] text", question)

    assert len(encoded.marker_positions) == 2
    assert [encoded.input_ids[position] for position in encoded.marker_positions] == [103, 103]
    assert encoded.input_ids.count(103) == 2


def test_builder_truncates_state_after_the_header() -> None:
    builder = SequenceBuilder(FakeTokenizer(), SequenceConfig(max_length=32, header_budget=24))
    question = DecisionQuestion.choice("Pick the best answer", [Option("A"), Option("B")])

    encoded = builder.build(" ".join(f"state{index}" for index in range(100)), question)

    assert len(encoded.input_ids) == 32
    assert encoded.input_ids[-1] == FakeTokenizer.sep_token_id


def test_builder_rejects_options_that_cannot_fit_the_header() -> None:
    builder = SequenceBuilder(FakeTokenizer(), SequenceConfig(max_length=32, header_budget=16))
    question = DecisionQuestion.choice("Pick", [Option("A"), Option("B"), Option("C")])

    with pytest.raises(ValueError, match="too many options"):
        builder.build("state", question)


def test_pad_marks_nonexistent_options_invalid() -> None:
    builder = SequenceBuilder(FakeTokenizer())
    two = builder.build("x", DecisionQuestion.choice("x", [Option("A"), Option("B")]))
    three = builder.build(
        "x", DecisionQuestion.choice("x", [Option("A"), Option("B"), Option("C")])
    )

    batch = builder.pad([two, three])

    assert batch["option_mask"] == [[1, 1, 0], [1, 1, 1]]


def test_pad_rejects_an_empty_batch() -> None:
    with pytest.raises(ValueError, match="empty batch"):
        SequenceBuilder(FakeTokenizer()).pad([])


def test_config_requires_all_three_type_prefixes() -> None:
    with pytest.raises(ValueError, match="type_prefixes"):
        SequenceConfig(type_prefixes={"choice": "a", "score": "b"})


def test_config_round_trips_through_a_dict() -> None:
    config = SequenceConfig(
        max_length=256,
        header_budget=128,
        type_prefixes={"choice": "t: c", "score": "t: s", "noul": "t: n"},
    )

    restored = SequenceConfig.from_dict(config.to_dict())

    assert restored == config
    assert restored.type_id("noul") == 2


def test_builder_requires_mask_and_pad_tokens() -> None:
    class Incomplete(FakeTokenizer):
        mask_token_id = None

    with pytest.raises(ValueError, match="mask_token_id"):
        SequenceBuilder(Incomplete())
