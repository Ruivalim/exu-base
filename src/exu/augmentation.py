"""Augmentations that stop the model learning shortcuts.

The most important one is option shuffling. Without it the model learns
position: in Laya the winning option changed in 15% to 23% of
cases when the option order was permuted. Shuffling every pass makes position
uninformative, and evaluating on permutations measures whether it worked.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from .data import TrainingExample
from .types import DecisionQuestion


def random_order(size: int, generator: random.Random) -> list[int]:
    """A random permutation of ``range(size)``."""
    if size < 1:
        raise ValueError("size must be positive")
    order = list(range(size))
    generator.shuffle(order)
    return order


def permute_target(target: Sequence[float], order: Sequence[int]) -> tuple[float, ...]:
    """Reorder a target so position ``i`` matches ``order[i]`` of the options."""
    if sorted(order) != list(range(len(target))):
        raise ValueError("order must be a permutation of the target indices")
    return tuple(target[index] for index in order)


def permute_example(example: TrainingExample, order: Sequence[int]) -> TrainingExample:
    """Reorder options and target together, keeping the mapping consistent."""
    if len(order) != example.option_count:
        raise ValueError("order length must equal option count")
    options = tuple(example.question.options[index] for index in order)
    question = DecisionQuestion(example.question.kind, example.question.instruction, options)
    return TrainingExample(
        example_id=example.example_id,
        state=example.state,
        question=question,
        target=permute_target(example.target, order),
        split=example.split,
        family=example.family,
        language=example.language,
    )


def shuffle_example(example: TrainingExample, generator: random.Random) -> TrainingExample:
    """Return the example with its options permuted."""
    return permute_example(example, random_order(example.option_count, generator))


class OptionShuffler:
    """Callable augmentation: shuffle a question's options on every pass."""

    def __init__(self, seed: int | None = None) -> None:
        self.generator = random.Random(seed)

    def __call__(self, example: TrainingExample) -> TrainingExample:
        return shuffle_example(example, self.generator)
