"""Turn a state and a question into a single encoder sequence.

The layout, one sequence per question::

    [CLS] <type prefix> <instruction> [SEP]
    [MASK] <option 0> [SEP]
    [MASK] <option 1> [SEP]
    ...
    <state, truncated from the right> [SEP]

The ``[MASK]`` before each option is a *marker*. Its hidden position is where
the decision head reads that option's score, and a bidirectional encoder lets
that position see the option, the competing options, the instruction and the
state at once.

Two properties matter and are tested:

- The token budget is explicit. Instructions and options share ``header_budget``
  and each option has a ceiling; options cede tokens before the instruction does.
  The state gets the remainder. Runaway option lists fail loudly instead of
  silently producing a degenerate question.
- Only builder-owned mask tokens are markers. Literal mask strings are stripped
  from instruction, option and state text first, so untrusted input cannot
  inject fake markers.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from .types import DecisionQuestion

DEFAULT_TYPE_PREFIXES: Mapping[str, str] = {
    "choice": "type: choice",
    "score": "type: ordinal scale",
    "noul": "type: yes or no",
}

_TYPE_ID: Mapping[str, int] = {"choice": 0, "score": 1, "noul": 2}


class MaskTokenizer(Protocol):
    """Minimum Hugging Face tokenizer surface needed by :class:`SequenceBuilder`."""

    cls_token_id: int | None
    sep_token_id: int | None
    mask_token_id: int | None
    pad_token_id: int | None
    mask_token: str | None

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...


@dataclass(frozen=True, slots=True)
class SequenceConfig:
    """Token budget and prompt scaffolding for sequence construction."""

    max_length: int = 512
    header_budget: int = 192
    max_option_tokens: int = 48
    min_instruction_tokens: int = 8
    min_option_text_tokens: int = 3
    type_prefixes: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_TYPE_PREFIXES))

    def __post_init__(self) -> None:
        if self.max_length < 16:
            raise ValueError("max_length must be at least 16")
        if not 8 <= self.header_budget <= self.max_length:
            raise ValueError("header_budget must be between 8 and max_length")
        if self.max_option_tokens < self.min_option_text_tokens + 1:
            raise ValueError("max_option_tokens must leave room for marker and option text")
        if set(self.type_prefixes) != {"choice", "score", "noul"}:
            raise ValueError("type_prefixes must define exactly choice, score and noul")
        if any(not prefix.strip() for prefix in self.type_prefixes.values()):
            raise ValueError("type prefixes cannot be blank")

    def to_dict(self) -> dict[str, object]:
        return {
            "max_length": self.max_length,
            "header_budget": self.header_budget,
            "max_option_tokens": self.max_option_tokens,
            "min_instruction_tokens": self.min_instruction_tokens,
            "min_option_text_tokens": self.min_option_text_tokens,
            "type_prefixes": dict(self.type_prefixes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> SequenceConfig:
        return cls(
            max_length=int(data.get("max_length", 512)),
            header_budget=int(data.get("header_budget", 192)),
            max_option_tokens=int(data.get("max_option_tokens", 48)),
            min_instruction_tokens=int(data.get("min_instruction_tokens", 8)),
            min_option_text_tokens=int(data.get("min_option_text_tokens", 3)),
            type_prefixes=dict(data.get("type_prefixes", DEFAULT_TYPE_PREFIXES)),  # type: ignore[arg-type]
        )

    def type_id(self, kind: str) -> int:
        return _TYPE_ID[kind]


@dataclass(frozen=True, slots=True)
class EncodedQuestion:
    """One question encoded as a single encoder sequence."""

    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    marker_positions: tuple[int, ...]
    question_type_id: int


class SequenceBuilder:
    """Build one encoder sequence per state/question pair."""

    def __init__(self, tokenizer: MaskTokenizer, config: SequenceConfig | None = None) -> None:
        self.tokenizer = tokenizer
        self.config = config or SequenceConfig()
        self._validate_tokenizer()

    def build(self, state: str, question: DecisionQuestion) -> EncodedQuestion:
        clean_state = self._clean(state)
        prefix_ids = self._encode(self.config.type_prefixes[question.kind.value])
        instruction_ids = self._encode(self._clean(question.instruction))
        option_ids = [self._encode(self._clean(option.render())) for option in question.options]

        option_limits, instruction_limit = self._allocate_header(
            len(question.options), len(prefix_ids)
        )
        header = [self._required(self.tokenizer.cls_token_id), *prefix_ids]
        header.extend(instruction_ids[:instruction_limit])
        header.append(self._required(self.tokenizer.sep_token_id))

        marker_positions: list[int] = []
        for ids, limit in zip(option_ids, option_limits, strict=True):
            marker_positions.append(len(header))
            header.append(self._required(self.tokenizer.mask_token_id))
            header.extend(ids[:limit])
            header.append(self._required(self.tokenizer.sep_token_id))

        state_limit = self.config.max_length - len(header) - 1
        if state_limit < 0:
            raise ValueError("header exceeds max_length")
        input_ids = (
            *header,
            *self._encode(clean_state)[:state_limit],
            self._required(self.tokenizer.sep_token_id),
        )
        return EncodedQuestion(
            input_ids=input_ids,
            attention_mask=(1,) * len(input_ids),
            marker_positions=tuple(marker_positions),
            question_type_id=self.config.type_id(question.kind.value),
        )

    def pad(
        self, encoded: list[EncodedQuestion] | tuple[EncodedQuestion, ...]
    ) -> dict[str, list[list[int]] | list[int]]:
        """Pad a batch of distinct questions for :meth:`rlcd.model.RLCDModel.forward`."""
        if not encoded:
            raise ValueError("cannot pad an empty batch")
        length = max(len(item.input_ids) for item in encoded)
        options = max(len(item.marker_positions) for item in encoded)
        pad_id = self._required(self.tokenizer.pad_token_id)
        return {
            "input_ids": [
                list(item.input_ids) + [pad_id] * (length - len(item.input_ids)) for item in encoded
            ],
            "attention_mask": [
                list(item.attention_mask) + [0] * (length - len(item.attention_mask))
                for item in encoded
            ],
            "marker_positions": [
                list(item.marker_positions) + [0] * (options - len(item.marker_positions))
                for item in encoded
            ],
            "option_mask": [
                [1] * len(item.marker_positions) + [0] * (options - len(item.marker_positions))
                for item in encoded
            ],
            "question_type_ids": [item.question_type_id for item in encoded],
        }

    def _allocate_header(self, option_count: int, prefix_length: int) -> tuple[list[int], int]:
        separator_count = option_count + 2  # after instruction, every option, final state
        fixed = 1 + prefix_length + separator_count + option_count  # CLS + separators + markers
        available = self.config.header_budget - fixed
        minimum = (
            self.config.min_instruction_tokens + option_count * self.config.min_option_text_tokens
        )
        if available < minimum:
            raise ValueError("too many options for configured header_budget")

        option_cap = self.config.max_option_tokens - 1
        option_total = min(
            option_count * option_cap, available - self.config.min_instruction_tokens
        )
        base, remainder = divmod(option_total, option_count)
        option_limits = [base + int(index < remainder) for index in range(option_count)]
        instruction_limit = available - sum(option_limits)
        return option_limits, instruction_limit

    def _clean(self, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("state and question text must be strings")
        token = self.tokenizer.mask_token
        return value.replace(token, "") if token else value

    def _encode(self, value: str) -> list[int]:
        return self.tokenizer.encode(value, add_special_tokens=False)

    def _validate_tokenizer(self) -> None:
        for name in ("cls_token_id", "sep_token_id", "mask_token_id", "pad_token_id"):
            if getattr(self.tokenizer, name) is None:
                raise ValueError(f"tokenizer must provide {name}")

    @staticmethod
    def _required(value: int | None) -> int:
        if value is None:
            raise ValueError("required tokenizer token id is missing")
        return value
