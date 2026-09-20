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


def test_instruction_survives_short_options() -> None:
    """Regression: the allocator reserved each option's ceiling, not its length.

    Four one-token options used to pin the instruction to its 8-token floor while
    the rest of the header sat unused, which silently truncated the question.
    """
    tokenizer = FakeTokenizer()
    builder = SequenceBuilder(tokenizer)
    instruction = " ".join(f"word{index}" for index in range(30))
    question = DecisionQuestion.choice(instruction, [Option(name) for name in ("a", "b", "c", "d")])

    encoded = builder.build("state", question)

    assert tokenizer.encode(instruction)[-1] in encoded.input_ids


def test_a_short_option_does_not_reserve_its_ceiling() -> None:
    builder = SequenceBuilder(FakeTokenizer())

    short_limits, short_instruction = builder._allocate_header([1, 1], 1)
    full_limits, full_instruction = builder._allocate_header([47, 47], 1)

    assert short_limits == [1, 1]
    assert short_instruction > full_instruction


def test_a_long_option_is_capped_at_max_option_tokens() -> None:
    builder = SequenceBuilder(FakeTokenizer(), SequenceConfig(max_option_tokens=8))

    limits, _instruction = builder._allocate_header([50, 50], 1)

    assert max(limits) == 7  # the ceiling counts the marker


def test_the_allocator_rejects_an_empty_option_list() -> None:
    builder = SequenceBuilder(FakeTokenizer())

    with pytest.raises(ValueError, match="at least one option"):
        builder._allocate_header([], 1)


def test_interleaved_mask_text_cannot_inject_a_marker() -> None:
    """Regression: one replace pass left ``[MASK]`` behind in ``[MA[MASK]SK]``."""
    builder = SequenceBuilder(FakeTokenizer())
    question = DecisionQuestion.choice("Pick", [Option("A"), Option("B")])

    encoded = builder.build("crafted [MA[MASK]SK] text", question)

    assert encoded.input_ids.count(FakeTokenizer.mask_token_id) == 2


def test_interleaved_mask_text_in_the_instruction_is_stripped_too() -> None:
    builder = SequenceBuilder(FakeTokenizer())
    question = DecisionQuestion.choice("Pick [MA[MASK]SK] this", [Option("A"), Option("B")])

    encoded = builder.build("state", question)

    assert encoded.input_ids.count(FakeTokenizer.mask_token_id) == 2


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
