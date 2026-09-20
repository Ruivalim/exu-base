"""Shared evaluation: run the model once, then score under any temperature map.

Every metric on a split comes from one pass over the logits. Temperatures are
applied afterwards, so the same collected logits answer "before calibration" and
"after calibration" without a second forward pass.
"""

from __future__ import annotations

import random
import time
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .augmentation import permute_example, random_order
from .calibration import TemperatureMap
from .data import TrainingBatch, TrainingExample
from .metrics import DecisionMetrics, classification_metrics
from .model import RLCDModel, masked_softmax
from .sequence import EncodedQuestion, SequenceBuilder


@dataclass(frozen=True, slots=True)
class Collected:
    """Logits and labels collected in a single evaluation pass."""

    logits: Tensor
    target: Tensor
    option_mask: Tensor
    question_kinds: tuple[str, ...]
    option_counts: Tensor
    families: tuple[str | None, ...]

    def __len__(self) -> int:
        return self.logits.size(0)

    def temperatures(self, temperature: TemperatureMap) -> Tensor:
        values = [
            temperature.temperature(kind, int(count))
            for kind, count in zip(self.question_kinds, self.option_counts.tolist(), strict=True)
        ]
        return torch.tensor(values, dtype=torch.float32)

    def ordinal_mask(self) -> Tensor:
        return torch.tensor([kind == "score" for kind in self.question_kinds], dtype=torch.bool)

    def probabilities(self, temperature: TemperatureMap) -> Tensor:
        return masked_softmax(self.logits, self.option_mask, self.temperatures(temperature))


def collect(
    model: RLCDModel, loader: DataLoader[TrainingBatch], device: torch.device | str
) -> Collected:
    """Run the model over a loader and keep every logit."""
    model.eval()
    logits: list[Tensor] = []
    targets: list[Tensor] = []
    masks: list[Tensor] = []
    kinds: list[str] = []
    counts: list[int] = []
    families: list[str | None] = []
    with torch.inference_mode():
        for batch in loader:
            moved = batch.to(device)
            output = model(**moved.model_inputs())
            logits.append(output.logits.float().cpu())
            targets.append(moved.targets.float().cpu())
            masks.append(moved.option_mask.cpu())
            kinds.extend(moved.question_kinds)
            counts.extend(int(value) for value in moved.option_counts.tolist())
            families.extend(moved.question_families or (None,) * len(moved.question_kinds))
    if not logits:
        raise ValueError("loader produced no batches")
    width = max(block.size(1) for block in logits)
    return Collected(
        logits=torch.cat([_pad_rows(block, width) for block in logits]),
        target=torch.cat([_pad_rows(block, width) for block in targets]),
        option_mask=torch.cat([_pad_rows(block, width) for block in masks]),
        question_kinds=tuple(kinds),
        option_counts=torch.tensor(counts, dtype=torch.long),
        families=tuple(families),
    )


def _pad_rows(block: Tensor, width: int) -> Tensor:
    if block.size(1) == width:
        return block
    return torch.nn.functional.pad(block, (0, width - block.size(1)))


def metrics_for(collected: Collected, temperature: TemperatureMap) -> DecisionMetrics:
    return classification_metrics(
        collected.probabilities(temperature),
        collected.target,
        collected.option_mask,
        ordinal_mask=collected.ordinal_mask(),
    )


def grouped_metrics(
    collected: Collected, temperature: TemperatureMap, group: str = "kind"
) -> dict[str, DecisionMetrics]:
    """Metrics per question kind or per task family, skipping empty groups."""
    if group == "kind":
        keys: Sequence[str | None] = collected.question_kinds
    elif group == "family":
        keys = collected.families
    else:
        raise ValueError("group must be 'kind' or 'family'")
    if any(key is None for key in keys):
        return {}
    result: dict[str, DecisionMetrics] = {}
    for key in sorted({str(value) for value in keys}):
        index = torch.tensor([str(value) == key for value in keys], dtype=torch.bool)
        if not bool(index.any()):
            continue
        probabilities = collected.probabilities(temperature)
        result[key] = classification_metrics(
            probabilities[index],
            collected.target[index],
            collected.option_mask[index],
            ordinal_mask=collected.ordinal_mask()[index],
        )
    return result


def order_robustness(
    model: RLCDModel,
    builder: SequenceBuilder,
    examples: Sequence[TrainingExample],
    temperature: TemperatureMap,
    *,
    permutations: int = 4,
    seed: int = 17,
    device: torch.device | str = "cpu",
    batch_size: int = 32,
) -> dict[str, float]:
    """How often the answer changes when the option order is permuted.

    Laya changed its answer in 15% to 23% of cases, which is the
    signature of a model that learned position instead of criteria. Report
    ``stability``: the fraction of permutations agreeing with the unpermuted
    answer. It should be close to 1.
    """
    if permutations < 1:
        raise ValueError("permutations must be positive")
    if not examples:
        raise ValueError("examples cannot be empty")

    base = _forward_probabilities(
        model,
        builder,
        [builder.build(example.state, example.question) for example in examples],
        [
            temperature.temperature(example.question.kind.value, example.option_count)
            for example in examples
        ],
        device,
        batch_size,
    )
    winners = [int(row.argmax().item()) for row in base]

    generator = random.Random(seed)
    encoded: list[EncodedQuestion] = []
    metadata: list[tuple[int, list[int]]] = []
    temperatures: list[float] = []
    for index, example in enumerate(examples):
        for _ in range(permutations):
            order = random_order(example.option_count, generator)
            permuted = permute_example(example, order)
            encoded.append(builder.build(permuted.state, permuted.question))
            metadata.append((index, order))
            temperatures.append(
                temperature.temperature(permuted.question.kind.value, permuted.option_count)
            )

    permuted_probabilities = _forward_probabilities(
        model, builder, encoded, temperatures, device, batch_size
    )
    agreement = 0
    winner_mass = 0.0
    for row, (index, order) in enumerate(metadata):
        mapped = order[int(permuted_probabilities[row].argmax().item())]
        agreement += int(mapped == winners[index])
        winner_mass += float(permuted_probabilities[row][order.index(winners[index])].item())
    total = len(metadata)
    return {
        "examples": len(examples),
        "permutations": permutations,
        "stability": agreement / total,
        "mean_winner_probability": winner_mass / total,
    }


def latency_benchmark(
    model: RLCDModel,
    builder: SequenceBuilder,
    examples: Sequence[TrainingExample],
    temperature: TemperatureMap,
    *,
    questions_per_call: Sequence[int] = (1, 5, 10, 50),
    repeats: int = 5,
    device: torch.device | str = "cpu",
) -> dict[str, dict[str, float]]:
    """p50 and p95 latency for a batched call of N questions on the same state."""
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if not examples:
        raise ValueError("examples cannot be empty")
    resolved = torch.device(device)
    report: dict[str, dict[str, float]] = {}
    for count in questions_per_call:
        if count < 1:
            raise ValueError("questions_per_call entries must be positive")
        chosen = [examples[index % len(examples)] for index in range(count)]
        state = chosen[0].state
        encoded = [builder.build(state, example.question) for example in chosen]
        temperatures = [
            temperature.temperature(example.question.kind.value, example.option_count)
            for example in chosen
        ]
        _forward_probabilities(model, builder, encoded, temperatures, resolved, count)
        timings: list[float] = []
        for _ in range(repeats):
            _synchronize(resolved)
            start = time.perf_counter()
            _forward_probabilities(model, builder, encoded, temperatures, resolved, count)
            _synchronize(resolved)
            timings.append((time.perf_counter() - start) * 1000.0)
        timings.sort()
        report[str(count)] = {
            "p50_ms": timings[len(timings) // 2],
            "p95_ms": timings[min(len(timings) - 1, int(round(0.95 * (len(timings) - 1))))],
        }
    return report


def _forward_probabilities(
    model: RLCDModel,
    builder: SequenceBuilder,
    encoded: Sequence[EncodedQuestion],
    temperatures: Sequence[float],
    device: torch.device | str,
    batch_size: int,
) -> list[Tensor]:
    if len(encoded) != len(temperatures):
        raise ValueError("encoded and temperatures must have the same length")
    model.eval()
    rows: list[Tensor] = []
    for start in range(0, len(encoded), batch_size):
        chunk = list(encoded[start : start + batch_size])
        raw = builder.pad(chunk)
        inputs = {
            "input_ids": torch.tensor(raw["input_ids"], dtype=torch.long, device=device),
            "attention_mask": torch.tensor(raw["attention_mask"], dtype=torch.long, device=device),
            "marker_positions": torch.tensor(
                raw["marker_positions"], dtype=torch.long, device=device
            ),
            "option_mask": torch.tensor(raw["option_mask"], dtype=torch.bool, device=device),
            "question_type_ids": torch.tensor(
                raw["question_type_ids"], dtype=torch.long, device=device
            ),
        }
        local_temperature = torch.tensor(
            list(temperatures[start : start + batch_size]), dtype=torch.float32, device=device
        )
        with torch.inference_mode():
            output = model(**inputs, temperature=local_temperature)
        for offset, item in enumerate(chunk):
            rows.append(output.probabilities[offset, : len(item.marker_positions)].cpu())
    return rows


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
