from __future__ import annotations

import random

import pytest

from exu import DecisionQuestion, Option
from exu.augmentation import (
    OptionShuffler,
    permute_example,
    permute_target,
    random_order,
    shuffle_example,
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


def test_score_questions_are_never_shuffled() -> None:
    """Regression: permuting the levels of an ordinal question broke the RPS term.

    The levels are the meaning of the option order, so a shuffle keeps the target
    aligned while destroying the distance the ordinal reward reads. The symptom
    was a far miss scoring better than a near one.
    """
    question = DecisionQuestion.score("How urgent?", ["low", "normal", "high"])
    ordinal = TrainingExample("x", "state", question, (0.0, 0.1, 0.9), family="urgency")

    for seed in range(5):
        shuffled = shuffle_example(ordinal, random.Random(seed))
        assert shuffled is ordinal
        assert [option.name for option in shuffled.question.options] == [
            "Level 0",
            "Level 1",
            "Level 2",
        ]


def test_noul_and_choice_questions_are_still_shuffled() -> None:
    question = DecisionQuestion.choice("Pick", [Option("A"), Option("B"), Option("C")])
    example_ = TrainingExample("x", "state", question, (0.1, 0.2, 0.7), family="f")

    orders = {
        tuple(
            option.name
            for option in shuffle_example(example_, random.Random(seed)).question.options
        )
        for seed in range(8)
    }

    assert len(orders) > 1
