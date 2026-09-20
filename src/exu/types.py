"""The typed-decision contract.

A decision is a *state* plus a *question*, and the answer is a probability
distribution over the question's explicit options. There is no generated text:
the model scores options and softmaxes, so the output is already a distribution.

The three primitives are one mechanism wearing three hats:

- ``choice`` picks one explicit option among N.
- ``score`` places the state on an ordinal rubric. Its options are generated
  from the level descriptions.
- ``noul`` is a boolean question, a ``choice`` with two fixed options.

New tasks do not require a new output layer: the options are written at request
time and read by the model as text.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum


class DecisionType(StrEnum):
    """Supported decision primitives."""

    CHOICE = "choice"
    SCORE = "score"
    NOUL = "noul"


@dataclass(frozen=True, slots=True)
class Option:
    """One candidate outcome, with optional criterion text.

    ``render`` is what the model actually reads. The name is the label returned
    to the caller, the description is the criterion that distinguishes it from
    the other options.
    """

    name: str
    description: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("option name cannot be empty")
        if self.description is not None and not self.description.strip():
            object.__setattr__(self, "description", None)

    def render(self) -> str:
        return self.name if self.description is None else f"{self.name}: {self.description}"


@dataclass(frozen=True, slots=True)
class DecisionQuestion:
    """Question answered by a distribution over explicit options."""

    kind: DecisionType
    instruction: str
    options: tuple[Option, ...]

    def __post_init__(self) -> None:
        if not self.instruction.strip():
            raise ValueError("instruction cannot be empty")
        if not 2 <= len(self.options) <= 255:
            raise ValueError("a question must have between 2 and 255 options")
        if len({option.name for option in self.options}) != len(self.options):
            raise ValueError("option names must be unique")
        if self.kind is DecisionType.NOUL and len(self.options) != 2:
            raise ValueError("noul questions always have exactly two options")

    @classmethod
    def choice(cls, instruction: str, options: Iterable[Option]) -> DecisionQuestion:
        return cls(DecisionType.CHOICE, instruction, tuple(options))

    @classmethod
    def score(
        cls,
        instruction: str,
        levels: Iterable[str],
        level_prefix: str = "Level",
    ) -> DecisionQuestion:
        """Build an ordinal question. ``level_prefix`` sets the rendering language."""
        options = tuple(
            Option(f"{level_prefix} {index}", level) for index, level in enumerate(levels)
        )
        return cls(DecisionType.SCORE, instruction, options)

    @classmethod
    def noul(
        cls,
        instruction: str,
        false_option: str = "No",
        true_option: str = "Yes",
    ) -> DecisionQuestion:
        return cls(
            DecisionType.NOUL,
            instruction,
            (Option(false_option), Option(true_option)),
        )

    @property
    def option_count(self) -> int:
        return len(self.options)
