"""Inference runtime: load a checkpoint once, then decide.

One forward pass per question batch, no generation, no parser. The result is a
distribution and a label, plus two *different* confidence scales that are easy to
confuse:

- ``confidence`` is the maximum calibrated probability, the quantity the ECE was
  measured on. Put business thresholds on this one.
- ``entropy_confidence`` is one minus normalized entropy. It is smaller for the
  same distribution (0.85 top probability over four options is about 0.58 here),
  so a threshold copied from the other scale will behave differently.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from .calibration import TemperatureMap
from .checkpoint import Checkpoint, load_checkpoint, load_tokenizer
from .devices import resolve_device
from .sequence import SequenceBuilder
from .types import DecisionQuestion, DecisionType

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True, slots=True)
class Decision:
    """One answered question."""

    kind: str
    option_names: tuple[str, ...]
    probabilities: tuple[float, ...]
    logits: tuple[float, ...]
    confidence: float
    entropy_confidence: float
    expected_level: float | None
    label: str
    should_act: bool


class DecisionRuntime:
    """A loaded checkpoint plus its tokenizer and sequence builder."""

    def __init__(
        self,
        checkpoint: Checkpoint,
        tokenizer: PreTrainedTokenizerBase,
        device: torch.device,
    ) -> None:
        self.checkpoint = checkpoint
        self.model = checkpoint.model
        self.tokenizer = tokenizer
        self.device = device
        self.builder = SequenceBuilder(tokenizer, checkpoint.sequence_config)
        self.temperature: TemperatureMap = checkpoint.temperature

    @classmethod
    def load(cls, path: str | Path, device: str = "auto") -> DecisionRuntime:
        resolved = resolve_device(device)
        checkpoint = load_checkpoint(path, device=resolved)
        tokenizer = load_tokenizer(path)
        return cls(checkpoint, tokenizer, resolved)

    def decide(self, state: str, question: DecisionQuestion) -> Decision:
        return self.decide_many([(state, question)])[0]

    def decide_many(self, pairs: list[tuple[str, DecisionQuestion]]) -> list[Decision]:
        if not pairs:
            raise ValueError("pairs cannot be empty")
        encoded = [self.builder.build(state, question) for state, question in pairs]
        raw = self.builder.pad(encoded)
        inputs = {
            "input_ids": torch.tensor(raw["input_ids"], dtype=torch.long, device=self.device),
            "attention_mask": torch.tensor(
                raw["attention_mask"], dtype=torch.long, device=self.device
            ),
            "marker_positions": torch.tensor(
                raw["marker_positions"], dtype=torch.long, device=self.device
            ),
            "option_mask": torch.tensor(raw["option_mask"], dtype=torch.bool, device=self.device),
            "question_type_ids": torch.tensor(
                raw["question_type_ids"], dtype=torch.long, device=self.device
            ),
        }
        temperatures = torch.tensor(
            [
                self.temperature.temperature(question.kind.value, len(question.options))
                for _state, question in pairs
            ],
            dtype=torch.float32,
            device=self.device,
        )
        with torch.inference_mode():
            output = self.model(**inputs, temperature=temperatures)
        return [
            self._decision(question, output.logits[row], output.probabilities[row])
            for row, (_state, question) in enumerate(pairs)
        ]

    def _decision(
        self, question: DecisionQuestion, logits: Tensor, probabilities: Tensor
    ) -> Decision:
        count = len(question.options)
        probs = probabilities[:count]
        scores = logits[:count]
        winner = int(torch.argmax(probs).item())
        confidence = float(probs[winner].item())
        normalized = _normalized_entropy(probs)
        expected_level: float | None = None
        if question.kind is DecisionType.SCORE:
            levels = torch.arange(count, dtype=probs.dtype)
            expected_level = float((probs * levels).sum().item())
        return Decision(
            kind=question.kind.value,
            option_names=tuple(option.name for option in question.options),
            probabilities=tuple(float(value) for value in probs.tolist()),
            logits=tuple(float(value) for value in scores.tolist()),
            confidence=confidence,
            entropy_confidence=1.0 - normalized,
            expected_level=expected_level,
            label=question.options[winner].name,
            should_act=bool(self.model.config.action_costs.should_act(confidence)),
        )


def _normalized_entropy(probabilities: Tensor) -> float:
    if probabilities.numel() < 2:
        return 0.0
    safe = probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny)
    entropy = float((-(safe * safe.log()).sum()).item())
    denominator = float(torch.tensor(probabilities.numel(), dtype=torch.float32).log().item())
    return min(1.0, max(0.0, entropy / denominator))
