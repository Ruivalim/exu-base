from __future__ import annotations

import random

import pytest

from exu import DecisionQuestion, Option
from exu.augmentation import (
    OptionShuffler,
    permute_example,
    permute_target,
    random_order,
)
from exu.data import TrainingExample


def example() -> TrainingExample:
    question = DecisionQuestion.choice("Pick", [Option("A"), Option("B"), Option("C")])
    return TrainingExample("x", "state", question, (0.1, 0.2, 0.7), family="f")


def test_random_order_is_a_permutation() -> None:
    generator = random.Random(1)
    order = random_order(5, generator)

    assert sorted(order) == [0, 1, 2, 3, 4]


def test_permute_target_follows_the_order() -> None:
    assert permute_target((0.1, 0.2, 0.7), [2, 0, 1]) == (0.7, 0.1, 0.2)


def test_permute_example_keeps_option_and_target_aligned() -> None:
    permuted = permute_example(example(), [1, 2, 0])

    assert [option.name for option in permuted.question.options] == ["B", "C", "A"]
    assert permuted.target == (0.2, 0.7, 0.1)
    assert permuted.family == "f"
    assert permuted.target.index(0.2) == [
        option.name for option in permuted.question.options
    ].index("B")


def test_permute_rejects_an_invalid_order() -> None:
    with pytest.raises(ValueError, match="permutation"):
        permute_target((0.5, 0.5), [0, 0])
    with pytest.raises(ValueError, match="order length"):
        permute_example(example(), [0, 1])


def test_shuffler_is_deterministic_for_a_seed() -> None:
    first = OptionShuffler(7)(example())
    second = OptionShuffler(7)(example())

    assert [option.name for option in first.question.options] == [
        option.name for option in second.question.options
    ]
    assert sorted(option.name for option in first.question.options) == ["A", "B", "C"]
