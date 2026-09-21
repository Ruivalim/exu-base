from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from helpers import smoke_examples

from exu import DecisionQuestion, Option
from exu.data import TrainingExample
from exu.metrics import (
    classification_metrics,
    expected_calibration_error,
    fit_reference_prior,
    prior_in_sample_baseline,
    selective_coverage,
    stack_examples,
    uniform_baseline,
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


def _row(question: DecisionQuestion, target: tuple[float, ...], name: str = "x") -> TrainingExample:
    return TrainingExample(name, "state", question, target, family="f")


AB = DecisionQuestion.choice("pick", [Option("a"), Option("b")])
BA = DecisionQuestion.choice("pick", [Option("b"), Option("a")])


def test_uniform_baseline_is_flat_over_the_valid_options() -> None:
    examples = smoke_examples()
    _targets, mask, _kinds, _counts = stack_examples(examples)

    uniform = uniform_baseline(examples)

    counts = mask.sum(dim=-1, keepdim=True)
    assert torch.allclose(uniform, mask.to(uniform.dtype) / counts)
    assert torch.allclose(uniform.sum(dim=-1), torch.ones(len(examples)))


def test_uniform_baseline_log_score_is_log_k() -> None:
    """The floor is exactly `log K`: no information, so nothing better than chance."""
    examples = smoke_examples()
    targets, mask, _kinds, _counts = stack_examples(examples)

    metrics = classification_metrics(uniform_baseline(examples), targets, mask)

    expected = masks_to_log_k(mask).mean()
    assert metrics.nll == pytest.approx(float(expected), abs=1e-5)


def masks_to_log_k(mask: torch.Tensor) -> torch.Tensor:
    return mask.sum(dim=-1).to(torch.float64).log()


def test_the_reference_prior_never_reads_the_labels_it_scores() -> None:
    # The rule for anything reported as a bar: mutate the evaluation targets and
    # the predictions must not move. The in-sample prior fails it by construction.
    prior = fit_reference_prior(smoke_examples("train"))
    evaluation = smoke_examples("test")
    mutated = [replace(item, target=tuple(reversed(item.target))) for item in evaluation]

    honest, _seen = prior.predict(evaluation)
    after, _seen = prior.predict(mutated)

    assert torch.equal(honest, after)
    assert torch.equal(prior.majority(evaluation), prior.majority(mutated))
    assert not torch.equal(prior_in_sample_baseline(evaluation), prior_in_sample_baseline(mutated))


def test_a_singleton_question_does_not_get_its_own_target() -> None:
    reference = [_row(AB, (1.0, 0.0)), _row(AB, (1.0, 0.0)), _row(AB, (0.0, 1.0))]
    singleton = [_row(AB, (0.0, 1.0))]

    predictions, seen = fit_reference_prior(reference).predict(singleton)

    # (counts + 1/K) / (n + 1) = ([2, 1] + 0.5) / 4
    assert torch.allclose(predictions, torch.tensor([[0.625, 0.375]]))
    assert seen.tolist() == [True]
    assert torch.equal(prior_in_sample_baseline(singleton), torch.tensor([[0.0, 1.0]]))


def test_an_unseen_question_falls_back_to_uniform_and_is_counted() -> None:
    other = DecisionQuestion.choice("route", [Option("x"), Option("y"), Option("z")])
    prior = fit_reference_prior([_row(AB, (1.0, 0.0))])

    predictions, seen = prior.predict([_row(other, (0.0, 0.0, 1.0)), _row(AB, (1.0, 0.0))])

    assert torch.allclose(predictions[0], torch.full((3,), 1 / 3))
    assert seen.tolist() == [False, True]


def test_an_outcome_the_reference_never_saw_keeps_a_finite_nll() -> None:
    # Without smoothing this row costs -log(float32 tiny) = 87.3 nats on its own.
    reference = [_row(AB, (1.0, 0.0))] * 5
    evaluation = [_row(AB, (0.0, 1.0))]

    predictions, _seen = fit_reference_prior(reference).predict(evaluation)
    targets, mask, _kinds, _counts = stack_examples(evaluation)
    metrics = classification_metrics(predictions, targets, mask)

    assert predictions[0, 1].item() == pytest.approx(0.5 / 6)
    assert metrics.nll < 3


def test_soft_targets_are_counted_as_mass_not_as_votes() -> None:
    reference = [_row(AB, (0.25, 0.75)), _row(AB, (0.75, 0.25))]

    predictions, _seen = fit_reference_prior(reference).predict([_row(AB, (1.0, 0.0))])

    assert torch.allclose(predictions, torch.tensor([[0.5, 0.5]]))


def test_a_choice_question_matches_its_reference_in_any_option_order() -> None:
    prior = fit_reference_prior([_row(AB, (1.0, 0.0))] * 3)

    predictions, seen = prior.predict([_row(BA, (0.0, 1.0))])

    # The mass follows option "a", which the permuted question shows second.
    assert seen.tolist() == [True]
    assert torch.allclose(predictions, torch.tensor([[0.125, 0.875]]))


def test_an_ordinal_question_is_not_reordered() -> None:
    rising = DecisionQuestion.score("rate", ["bad", "fine", "good"])
    falling = DecisionQuestion.score("rate", ["good", "fine", "bad"])
    prior = fit_reference_prior([_row(rising, (1.0, 0.0, 0.0))])

    _predictions, seen = prior.predict([_row(falling, (1.0, 0.0, 0.0))])

    assert seen.tolist() == [False]


def test_mixed_option_counts_stay_valid_distributions() -> None:
    wide = DecisionQuestion.choice("route", [Option("x"), Option("y"), Option("z")])
    reference = [_row(AB, (1.0, 0.0)), _row(wide, (0.0, 0.0, 1.0))]
    evaluation = [_row(AB, (0.0, 1.0)), _row(wide, (1.0, 0.0, 0.0))]

    predictions, _seen = fit_reference_prior(reference).predict(evaluation)

    assert predictions.shape == (2, 3)
    assert predictions[0, 2].item() == 0
    assert torch.allclose(predictions.sum(dim=-1), torch.ones(2))


def test_majority_breaks_a_tie_on_option_identity_not_on_position() -> None:
    prior = fit_reference_prior([_row(AB, (1.0, 0.0)), _row(AB, (0.0, 1.0))])

    shown_first = prior.majority([_row(AB, (1.0, 0.0))])
    shown_second = prior.majority([_row(BA, (1.0, 0.0))])

    assert shown_first.tolist() == [[1.0, 0.0]]
    assert shown_second.tolist() == [[0.0, 1.0]]


def test_majority_is_the_argmax_of_the_same_prior() -> None:
    prior = fit_reference_prior(smoke_examples("train"))
    evaluation = smoke_examples("test")

    predictions, _seen = prior.predict(evaluation)
    majority = prior.majority(evaluation)

    assert torch.allclose(majority.sum(dim=-1), torch.ones(len(evaluation)))
    assert torch.equal(majority.argmax(dim=-1), predictions.argmax(dim=-1))


def test_a_reference_cannot_be_empty() -> None:
    with pytest.raises(ValueError, match="reference"):
        fit_reference_prior([])


def test_the_fixture_numbers_of_the_decision_record_hold() -> None:
    evaluation = smoke_examples("test")
    targets, mask, _kinds, _counts = stack_examples(evaluation)
    predictions, seen = fit_reference_prior(smoke_examples("train")).predict(evaluation)

    fitted = classification_metrics(predictions, targets, mask)
    in_sample = classification_metrics(prior_in_sample_baseline(evaluation), targets, mask)

    assert seen.all()
    assert (fitted.nll, fitted.accuracy) == (pytest.approx(0.7230, abs=1e-4), 0.25)
    assert (in_sample.nll, in_sample.accuracy) == (pytest.approx(0.3466, abs=1e-4), 0.75)


def test_expected_calibration_error_validates_shapes() -> None:
    with pytest.raises(ValueError, match="shape"):
        expected_calibration_error(torch.tensor([0.5]), torch.tensor([0.5, 0.5]))
