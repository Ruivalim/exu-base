from __future__ import annotations

import pytest
import torch
from helpers import smoke_examples

from exu.metrics import (
    classification_metrics,
    expected_calibration_error,
    majority_class_baseline,
    prior_baseline,
    random_baseline,
    selective_coverage,
    stack_examples,
)


def test_perfect_predictions_have_zero_calibration_error() -> None:
    probabilities = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    mask = torch.ones(2, 2, dtype=torch.bool)

    metrics = classification_metrics(probabilities, probabilities.clone(), mask)

    assert metrics.accuracy == 1.0
    assert metrics.ece == 0.0
    assert metrics.nll == 0.0
    assert metrics.brier == 0.0
    assert metrics.soft_accuracy == 1.0


def test_ece_measures_the_confidence_accuracy_gap() -> None:
    probabilities = torch.tensor([[0.8, 0.2], [0.8, 0.2]])
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    mask = torch.ones(2, 2, dtype=torch.bool)

    metrics = classification_metrics(probabilities, target, mask, ece_bins=2)

    assert metrics.accuracy == 0.5
    assert metrics.ece == pytest.approx(0.3)


def test_metrics_reject_invalid_prediction_mass() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        classification_metrics(
            torch.tensor([[0.7, 0.2]]),
            torch.tensor([[1.0, 0.0]]),
            torch.ones(1, 2, dtype=torch.bool),
        )


def test_ordinal_metrics_report_rps_and_mean_absolute_error() -> None:
    probabilities = torch.tensor([[0.0, 1.0, 0.0]])
    target = torch.tensor([[0.0, 0.0, 1.0]])
    mask = torch.ones(1, 3, dtype=torch.bool)

    metrics = classification_metrics(probabilities, target, mask, ordinal_mask=torch.tensor([True]))

    assert metrics.rps is not None and metrics.rps > 0
    assert metrics.ordinal_mae == pytest.approx(1.0)


def test_non_ordinal_metrics_skip_ordinal_scores() -> None:
    probabilities = torch.tensor([[0.5, 0.5]])
    mask = torch.ones(1, 2, dtype=torch.bool)

    metrics = classification_metrics(probabilities, probabilities.clone(), mask)

    assert metrics.rps is None
    assert metrics.ordinal_mae is None


def test_selective_coverage_improves_when_confidence_is_informative() -> None:
    probabilities = torch.tensor([[0.9, 0.1], [0.8, 0.2], [0.6, 0.4], [0.55, 0.45]])
    target = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    mask = torch.ones(4, 2, dtype=torch.bool)

    coverage = selective_coverage(probabilities, target, mask, fractions=(0.5,))

    assert coverage["all"] == 0.5
    assert coverage["top_0.5"] == 1.0


def test_selective_coverage_validates_the_fraction() -> None:
    with pytest.raises(ValueError, match="fractions"):
        selective_coverage(
            torch.tensor([[0.5, 0.5]]),
            torch.tensor([[1.0, 0.0]]),
            torch.ones(1, 2, dtype=torch.bool),
            fractions=(0.0,),
        )


def test_stack_examples_pads_to_the_widest_question() -> None:
    examples = smoke_examples()
    targets, mask, kinds, counts = stack_examples(examples)

    assert targets.shape[0] == len(examples)
    assert targets.shape[1] == max(counts).item()
    assert mask.sum(dim=-1).tolist() == counts.tolist()
    assert set(kinds) == {"choice", "score", "noul"}
    assert torch.allclose(targets.sum(dim=-1), torch.ones(len(examples)))


def test_prior_baseline_reproduces_the_per_question_mean() -> None:
    examples = smoke_examples()
    targets, _mask, _kinds, _counts = stack_examples(examples)

    prior = prior_baseline(examples)

    assert torch.allclose(prior.sum(dim=-1), torch.ones(len(examples)))
    for index, example in enumerate(examples):
        if example.question.kind.value == "noul" and len(example.question.options) == 2:
            assert prior[index, :2].sum().item() == pytest.approx(1.0)
    assert not torch.allclose(prior, targets)


def test_random_baseline_is_uniform_over_the_valid_options() -> None:
    examples = smoke_examples()
    _targets, mask, _kinds, _counts = stack_examples(examples)

    random = random_baseline(examples)

    counts = mask.sum(dim=-1, keepdim=True)
    assert torch.allclose(random, mask.to(random.dtype) / counts)
    assert torch.allclose(random.sum(dim=-1), torch.ones(len(examples)))


def test_uniform_baseline_log_score_is_log_k() -> None:
    """The floor is exactly `log K`: no information, so nothing better than chance."""
    examples = smoke_examples()
    targets, mask, _kinds, _counts = stack_examples(examples)

    metrics = classification_metrics(random_baseline(examples), targets, mask)

    expected = masks_to_log_k(mask).mean()
    assert metrics.nll == pytest.approx(float(expected), abs=1e-5)


def masks_to_log_k(mask: torch.Tensor) -> torch.Tensor:
    return mask.sum(dim=-1).to(torch.float64).log()


def test_majority_baseline_is_a_valid_distribution() -> None:
    examples = smoke_examples()
    majority = majority_class_baseline(examples)

    assert torch.allclose(majority.sum(dim=-1), torch.ones(len(examples)))


def test_expected_calibration_error_validates_shapes() -> None:
    with pytest.raises(ValueError, match="shape"):
        expected_calibration_error(torch.tensor([0.5]), torch.tensor([0.5, 0.5]))
