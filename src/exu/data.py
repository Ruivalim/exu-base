"""Validated JSONL examples and torch batches.

One JSONL record is one fully specified typed decision::

    {
      "id": "uuid or short key",
      "state": "text" | {"any": "json"} | ["any", "json"],
      "question": {
        "kind": "choice" | "score" | "noul",
        "instruction": "what to decide",
        "options": [{"name": "...", "description": "..."}]
      },
      "target": [0.2, 0.8],
      "split": "train" | "validation" | "test" | "calibration",
      "family": "routing",
      "language": "en"
    }

The target is a distribution, not a label. Soft targets let several annotators
or a teacher model express disagreement, which is exactly the signal a
calibrated model should learn from.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .sequence import SequenceBuilder
from .types import DecisionQuestion, DecisionType, Option


@dataclass(frozen=True, slots=True)
class TrainingExample:
    """One fully specified typed-decision training record."""

    example_id: str
    state: str
    question: DecisionQuestion
    target: tuple[float, ...]
    split: str | None = None
    family: str | None = None
    language: str | None = None

    def __post_init__(self) -> None:
        if not self.example_id.strip():
            raise ValueError("example id cannot be empty")
        if len(self.target) != len(self.question.options):
            raise ValueError("target length must equal option count")
        if any(not math.isfinite(value) or value < 0 for value in self.target):
            raise ValueError("target must contain finite, non-negative probabilities")
        if not math.isclose(sum(self.target), 1.0, abs_tol=1e-6):
            raise ValueError("target probabilities must sum to one")
        if self.split is not None and not self.split.strip():
            raise ValueError("split cannot be empty")
        if self.family is not None and not self.family.strip():
            raise ValueError("family cannot be empty")

    @property
    def option_count(self) -> int:
        return len(self.question.options)

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> TrainingExample:
        try:
            raw_question = _mapping(record["question"], "question")
            raw_options = _sequence(raw_question["options"], "question.options")
            options = tuple(
                Option(
                    _string(_mapping(item, "question option")["name"], "question option name"),
                    _optional_string(_mapping(item, "question option").get("description")),
                )
                for item in raw_options
            )
            question = DecisionQuestion(
                DecisionType(_string(raw_question["kind"], "question kind")),
                _string(raw_question["instruction"], "question instruction"),
                options,
            )
            raw_target = _sequence(record["target"], "target")
            target = tuple(float(value) for value in raw_target)
            return cls(
                _string(record["id"], "id"),
                _serialize_state(record["state"]),
                question,
                target,
                _optional_string(record.get("split")),
                _optional_string(record.get("family")),
                _optional_string(record.get("language")),
            )
        except KeyError as error:
            raise ValueError(f"record missing required field {error.args[0]!r}") from error
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid training record: {error}") from error

    def to_record(self) -> dict[str, Any]:
        return {
            "id": self.example_id,
            "state": _parse_state(self.state),
            "question": {
                "kind": self.question.kind.value,
                "instruction": self.question.instruction,
                "options": [
                    {"name": option.name, "description": option.description}
                    for option in self.question.options
                ],
            },
            "target": list(self.target),
            "split": self.split,
            "family": self.family,
            "language": self.language,
        }


@dataclass(frozen=True, slots=True)
class TrainingBatch:
    """Tensor batch for a model forward pass and a proper-scoring objective."""

    input_ids: Tensor
    attention_mask: Tensor
    marker_positions: Tensor
    option_mask: Tensor
    question_type_ids: Tensor
    targets: Tensor
    ordinal_mask: Tensor
    question_kinds: tuple[str, ...]
    question_families: tuple[str | None, ...] = ()

    def model_inputs(self) -> dict[str, Tensor]:
        return {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "marker_positions": self.marker_positions,
            "option_mask": self.option_mask,
            "question_type_ids": self.question_type_ids,
        }

    @property
    def option_counts(self) -> Tensor:
        return self.option_mask.sum(dim=-1)

    def to(self, device: torch.device | str) -> TrainingBatch:
        return TrainingBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            marker_positions=self.marker_positions.to(device),
            option_mask=self.option_mask.to(device),
            question_type_ids=self.question_type_ids.to(device),
            targets=self.targets.to(device),
            ordinal_mask=self.ordinal_mask.to(device),
            question_kinds=self.question_kinds,
            question_families=self.question_families,
        )


class DecisionDataset(Dataset[TrainingExample]):
    """In-memory examples with a collator backed by :class:`SequenceBuilder`."""

    def __init__(
        self,
        examples: Sequence[TrainingExample],
        builder: SequenceBuilder,
        transform: Callable[[TrainingExample], TrainingExample] | None = None,
    ) -> None:
        if not examples:
            raise ValueError("dataset cannot be empty")
        self.examples = tuple(examples)
        self.builder = builder
        self.transform = transform

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> TrainingExample:
        example = self.examples[index]
        return self.transform(example) if self.transform is not None else example

    def collate(self, examples: Sequence[TrainingExample]) -> TrainingBatch:
        if not examples:
            raise ValueError("cannot collate an empty batch")
        encoded = [self.builder.build(example.state, example.question) for example in examples]
        raw_batch = self.builder.pad(encoded)
        option_count = max(example.option_count for example in examples)
        targets = torch.zeros((len(examples), option_count), dtype=torch.float32)
        for index, example in enumerate(examples):
            targets[index, : example.option_count] = torch.tensor(
                example.target, dtype=torch.float32
            )
        return TrainingBatch(
            input_ids=torch.tensor(raw_batch["input_ids"], dtype=torch.long),
            attention_mask=torch.tensor(raw_batch["attention_mask"], dtype=torch.long),
            marker_positions=torch.tensor(raw_batch["marker_positions"], dtype=torch.long),
            option_mask=torch.tensor(raw_batch["option_mask"], dtype=torch.bool),
            question_type_ids=torch.tensor(raw_batch["question_type_ids"], dtype=torch.long),
            targets=targets,
            ordinal_mask=torch.tensor(
                [example.question.kind is DecisionType.SCORE for example in examples],
                dtype=torch.bool,
            ),
            question_kinds=tuple(example.question.kind.value for example in examples),
            question_families=tuple(example.family for example in examples),
        )


def load_jsonl(path: str | Path, split: str | None = None) -> list[TrainingExample]:
    """Read UTF-8 JSONL records, optionally selecting a declared split."""
    if split is not None and not split.strip():
        raise ValueError("split cannot be empty")
    examples: list[TrainingExample] = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = _mapping(json.loads(line), "record")
                example = TrainingExample.from_record(record)
                if split is None or example.split == split:
                    examples.append(example)
            except (json.JSONDecodeError, ValueError, TypeError) as error:
                raise ValueError(f"invalid JSONL line {line_number}: {error}") from error
    if not examples:
        suffix = f" for split {split!r}" if split is not None else ""
        raise ValueError(f"dataset contains no examples{suffix}")
    return examples


def write_jsonl(path: str | Path, examples: Sequence[TrainingExample]) -> None:
    """Write examples back out, preserving every declared field."""
    stream = "\n".join(
        json.dumps(example.to_record(), ensure_ascii=False, sort_keys=True) for example in examples
    )
    Path(path).write_text(stream + "\n", encoding="utf-8")


def _serialize_state(value: Any) -> str:
    if isinstance(value, str):
        if not value.strip():
            raise ValueError("state cannot be empty")
        return value
    if isinstance(value, (Mapping, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    raise TypeError("state must be a string, object, or list")


def _parse_state(value: str) -> Any:
    """Recover the original structure so a round trip is lossless."""
    stripped = value.lstrip()
    if stripped.startswith(("{", "[")):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be an array")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def _optional_string(value: Any) -> str | None:
    if value is not None and not isinstance(value, str):
        raise TypeError("optional text field must be a string or null")
    return value
