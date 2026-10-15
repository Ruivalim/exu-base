"""Inference runtime: load a checkpoint once, then decide.

One forward pass per question batch, no generation, no parser. The result is a
distribution and a label, plus two *different* confidence scales that are easy to
confuse:

- ``confidence`` is the maximum calibrated probability, the quantity the ECE was
  measured on. Put business thresholds on this one.
- ``entropy_confidence`` is one minus normalized entropy. It is smaller for the
  same distribution (0.85 top probability over four options is about 0.58 here),
  so a threshold copied from the other scale will behave differently.

Every decision also carries ``input_usage``: how many tokens of the state the model
read, out of how many it was given. A state longer than the sequence is cut from
the right, which is the default (``truncation="allow"``). With
``truncation="error"`` the runtime refuses instead, before the forward pass, and
names the question: an answer about the first half of a trace is an answer about
a different trace.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import Tensor

from .architecture import builder_for
from .calibration import TemperatureMap
from .checkpoint import Checkpoint, load_checkpoint, load_tokenizer
from .devices import resolve_device
from .sequence import InputUsage
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
    input_usage: InputUsage | None = None


Truncation = Literal["allow", "error"]


class StateTruncatedError(ValueError):
    """A state did not fit and the runtime was told not to cut it.

    ``index`` is the question's position in the call, ``usage`` what was counted.
    """

    def __init__(self, index: int, usage: InputUsage) -> None:
        self.index = index
        self.usage = usage
        self.detail = (
            f"the state takes {usage.state_tokens_original} tokens and only "
            f"{usage.state_tokens_used} fit; truncation='error' refuses to cut it"
        )
        super().__init__(f"question at index {index}: {self.detail}")


class DecisionRuntime:
    """A loaded checkpoint plus its tokenizer and sequence builder."""

    def __init__(
        self,
        checkpoint: Checkpoint,
        tokenizer: PreTrainedTokenizerBase,
        device: torch.device,
        *,
        truncation: Truncation = "allow",
    ) -> None:
        if truncation not in ("allow", "error"):
            raise ValueError("truncation must be 'allow' or 'error'")
        self.truncation = truncation
        self.checkpoint = checkpoint
        self.model = checkpoint.model
        self.tokenizer = tokenizer
        self.device = device
        self.builder = builder_for(tokenizer, checkpoint.sequence_config, checkpoint.model_config)
        self.temperature: TemperatureMap = checkpoint.temperature

    @classmethod
    def load(
        cls, path: str | Path, device: str = "auto", *, truncation: Truncation = "allow"
    ) -> DecisionRuntime:
        resolved = resolve_device(device)
        checkpoint = load_checkpoint(path, device=resolved)
        tokenizer = load_tokenizer(path)
        return cls(checkpoint, tokenizer, resolved, truncation=truncation)

    def decide(
        self, state: str, question: DecisionQuestion, context: str | None = None
    ) -> Decision:
        return self.decide_many([(state, question, context)])[0]

    def decide_all(
        self, state: str, questions: Sequence[DecisionQuestion], context: str | None = None
    ) -> list[Decision]:
        """Answer several questions about one state, in one forward pass.

        A decoder checkpoint reads the context and the state once for all of them.
        """
        if not questions:
            raise ValueError("questions cannot be empty")
        return self.decide_many([(state, question, context) for question in questions])

    def decide_many(
        self,
        pairs: Sequence[tuple[str, DecisionQuestion] | tuple[str, DecisionQuestion, str | None]],
    ) -> list[Decision]:
        """Answer ``(state, question)`` or ``(state, question, context)`` items, in one pass."""
        encoded = self.encode(pairs)
        inputs = self.builder.tensors(encoded, self.device)
        temperatures = torch.tensor(
            [
                self.temperature.temperature(question.kind.value, len(question.options))
                for _state, question, *_context in pairs
            ],
            dtype=torch.float32,
            device=self.device,
        )
        with torch.inference_mode():
            output = self.model(**inputs, temperature=temperatures)
        return [
            self._decision(question, output.logits[row], output.probabilities[row], item.usage)
            for row, ((_state, question, *_context), item) in enumerate(
                zip(pairs, encoded, strict=True)
            )
        ]

    def encode(
        self,
        pairs: Sequence[tuple[str, DecisionQuestion] | tuple[str, DecisionQuestion, str | None]],
    ) -> list[Any]:
        """Build every item and apply the truncation policy, without running the model.

        Raises :class:`StateTruncatedError` for the first item whose state was cut
        when ``truncation="error"``. A caller can run this over a whole input before
        answering any of it.
        """
        if not pairs:
            raise ValueError("pairs cannot be empty")
        encoded = [self.builder.build(*item) for item in pairs]
        if self.truncation == "error":
            for index, item in enumerate(encoded):
                if item.usage is None:
                    raise ValueError("this builder reports no input usage to check")
                if item.usage.state_truncated:
                    raise StateTruncatedError(index, item.usage)
        return encoded

    def _decision(
        self,
        question: DecisionQuestion,
        logits: Tensor,
        probabilities: Tensor,
        usage: InputUsage | None,
    ) -> Decision:
        count = len(question.options)
        probs = probabilities[:count]
        scores = logits[:count]
        winner = int(torch.argmax(probs).item())
        confidence = float(probs[winner].item())
        normalized = _normalized_entropy(probs)
        expected_level: float | None = None
        if question.kind is DecisionType.SCORE:
            levels = torch.arange(count, dtype=probs.dtype, device=probs.device)
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
            input_usage=usage,
        )


def _normalized_entropy(probabilities: Tensor) -> float:
    if probabilities.numel() < 2:
        return 0.0
    safe = probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny)
    entropy = float((-(safe * safe.log()).sum()).item())
    denominator = float(torch.tensor(probabilities.numel(), dtype=torch.float32).log().item())
    return min(1.0, max(0.0, entropy / denominator))
