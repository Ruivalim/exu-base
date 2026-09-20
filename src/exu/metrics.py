"""Evaluation metrics and baselines for calibrated decisions.

Accuracy alone hides the failure mode that matters here. A model can be right
most of the time and still be untrustworthy if its confidence is uncalibrated,
so every report carries NLL, Brier, expected calibration error, and, for ordinal
questions, the distance from the true level.

Two baselines are mandatory before claiming progress: the uniform forecast over
the valid options and the per-question class prior. `evaluate` adds the majority
class as a third one. In Laya, the base multilingual checkpoints scored *below*
the majority-class baseline on unseen task families, which nobody would have
noticed without that comparison.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from .data import TrainingExample
from .scoring import validate_distributions


@dataclass(frozen=True, slots=True)
class DecisionMetrics:
    """Decision quality and calibration measured on one split."""

    count: int
    nll: float
    brier: float
    accuracy: float
    soft_accuracy: float
    ece: float
    rps: float | None = None
    ordinal_mae: float | None = None


def classification_metrics(
    probabilities: Tensor,
    target: Tensor,
    option_mask: Tensor,
    *,
    ordinal_mask: Tensor | None = None,
    ece_bins: int = 15,
) -> DecisionMetrics:
    """Compute NLL, Brier, accuracy, soft accuracy, ECE and ordinal scores."""
    validate_distributions(probabilities, target, option_mask)
    if ece_bins < 1:
        raise ValueError("ece_bins must be positive")

    eps = torch.finfo(probabilities.dtype).tiny
    nll = -(target * probabilities.clamp_min(eps).log()).sum(dim=-1).mean()
    brier = ((probabilities - target).square() * option_mask).sum(dim=-1).mean()
    soft_accuracy = (probabilities * target).sum(dim=-1).mean()
    prediction = probabilities.argmax(dim=-1)
    outcome = target.argmax(dim=-1)
    correct = prediction.eq(outcome)
    confidence = probabilities.max(dim=-1).values
    ece = expected_calibration_error(confidence, correct, ece_bins)

    rps: float | None = None
    ordinal_mae: float | None = None
    if ordinal_mask is not None and bool(ordinal_mask.any()):
        if ordinal_mask.shape != probabilities.shape[:-1]:
            raise ValueError("ordinal_mask must have one value per row")
        cumulative_error = (probabilities.cumsum(-1) - target.cumsum(-1)).square()
        valid_count = option_mask.sum(dim=-1).sub(1).clamp_min(1)
        per_row = (cumulative_error * option_mask).sum(dim=-1) / valid_count
        rps = float(per_row.masked_select(ordinal_mask).mean().item())
        levels = torch.arange(probabilities.size(-1), dtype=probabilities.dtype)
        expected = (probabilities * levels).sum(dim=-1)
        wanted = (target * levels).sum(dim=-1)
        ordinal_mae = float((expected - wanted).abs().masked_select(ordinal_mask).mean().item())

    return DecisionMetrics(
        count=probabilities.shape[0],
        nll=float(nll.item()),
        brier=float(brier.item()),
        accuracy=float(correct.float().mean().item()),
        soft_accuracy=float(soft_accuracy.item()),
        ece=float(ece.item()),
        rps=rps,
        ordinal_mae=ordinal_mae,
    )


def expected_calibration_error(confidence: Tensor, correct: Tensor, bins: int = 15) -> Tensor:
    """Gap between confidence and accuracy, weighted over equal-width bins."""
    if confidence.shape != correct.shape:
        raise ValueError("confidence and correct must have identical shape")
    if bins < 1:
        raise ValueError("bins must be positive")
    error = torch.zeros((), dtype=confidence.dtype, device=confidence.device)
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        in_bin = (confidence >= lower) & (
            confidence <= upper if index == bins - 1 else confidence < upper
        )
        if in_bin.any():
            error += (
                in_bin.float().mean()
                * (confidence[in_bin].mean() - correct[in_bin].float().mean()).abs()
            )
    return error


def selective_coverage(
    probabilities: Tensor,
    target: Tensor,
    option_mask: Tensor,
    fractions: Sequence[float] = (0.5, 0.8),
) -> dict[str, float]:
    """Accuracy when answering only the most confident fraction of questions.

    If this curve does not rise as coverage falls, confidence cannot be used to
    route work to a human.
    """
    validate_distributions(probabilities, target, option_mask)
    if not fractions:
        raise ValueError("fractions cannot be empty")
    if any(not 0 < fraction <= 1 for fraction in fractions):
        raise ValueError("fractions must be in (0, 1]")
    correct = probabilities.argmax(dim=-1).eq(target.argmax(dim=-1))
    confidence = probabilities.max(dim=-1).values
    order = torch.argsort(confidence, descending=True)
    total = correct.numel()
    coverage = {"all": float(correct.float().mean().item())}
    for fraction in sorted(fractions):
        keep = max(1, int(round(total * fraction)))
        selected = correct[order[:keep]]
        coverage[f"top_{fraction:g}"] = float(selected.float().mean().item())
    return coverage


def stack_examples(
    examples: Sequence[TrainingExample],
) -> tuple[Tensor, Tensor, tuple[str, ...], Tensor]:
    """Turn examples into padded target, mask, kinds and option counts."""
    if not examples:
        raise ValueError("examples cannot be empty")
    width = max(example.option_count for example in examples)
    targets = torch.zeros((len(examples), width), dtype=torch.float32)
    mask = torch.zeros((len(examples), width), dtype=torch.bool)
    kinds: list[str] = []
    for index, example in enumerate(examples):
        targets[index, : example.option_count] = torch.tensor(example.target, dtype=torch.float32)
        mask[index, : example.option_count] = True
        kinds.append(example.question.kind.value)
    return targets, mask, tuple(kinds), mask.sum(dim=-1)


def prior_baseline(examples: Sequence[TrainingExample]) -> Tensor:
    """Per-question mean target: the strongest trivial distribution baseline."""
    targets, _mask, _kinds, _counts = stack_examples(examples)
    priors: dict[tuple[str, str, tuple[str, ...]], Tensor] = {}
    rows: list[tuple[tuple[str, str, tuple[str, ...]], int]] = []
    for index, example in enumerate(examples):
        key = _question_key(example)
        rows.append((key, index))
        priors.setdefault(key, torch.zeros_like(targets[0]))
        priors[key] += targets[index]
    result = torch.zeros_like(targets)
    for key, index in rows:
        prior = priors[key]
        result[index] = prior / prior.sum().clamp_min(torch.finfo(prior.dtype).tiny)
    return result


def majority_class_baseline(examples: Sequence[TrainingExample]) -> Tensor:
    """One-hot pick of the most frequent outcome per question."""
    prior = prior_baseline(examples)
    winner = prior.argmax(dim=-1, keepdim=True)
    result = torch.zeros_like(prior)
    result.scatter_(1, winner, 1.0)
    return result


def random_baseline(examples: Sequence[TrainingExample]) -> Tensor:
    """The guess-a-label forecast: uniform ``1/K`` over each question's options.

    Deterministic, and the weakest thing worth reporting. It carries no
    information, which is what makes it a useful floor: any model that cannot
    beat it on NLL has learned nothing. Its ECE is not zero, because a 15-bin
    ECE reads a maximum probability of ``1/K`` as slight miscalibration; see
    `docs/evaluation.md` for what that does and does not mean.
    """
    targets, mask, _kinds, _counts = stack_examples(examples)
    result = mask.to(targets.dtype)
    return result / result.sum(dim=-1, keepdim=True).clamp_min(1.0)


def _question_key(example: TrainingExample) -> tuple[str, str, tuple[str, ...]]:
    return (
        example.question.kind.value,
        example.question.instruction,
        tuple(option.render() for option in example.question.options),
    )
