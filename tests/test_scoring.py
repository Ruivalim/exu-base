from __future__ import annotations

import math

import pytest
import torch

from exu.model import masked_log_softmax
from exu.scoring import (
    composite_score,
    log_score,
    proper_scoring_loss,
    ranked_probability_score,
    spherical_score,
    validate_distributions,
)


def test_log_score_has_no_floor() -> None:
    # softmax([-200, 0]) underflows to exactly [0, 1] in float32. In log space the
    # confident miss keeps its real price instead of a clamped one.
    logits = torch.tensor([[-200.0, 0.0]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    target = torch.tensor([[1.0, 0.0]])

    score = log_score(masked_log_softmax(logits, mask), target)

    assert float(score.item()) == pytest.approx(-200.0)


def test_log_score_reads_zero_mass_on_a_zero_probability_as_zero() -> None:
    log_probabilities = torch.tensor([[-math.inf, 0.0]])

    ignored = log_score(log_probabilities, torch.tensor([[0.0, 1.0]]))
    punished = log_score(log_probabilities, torch.tensor([[1.0, 0.0]]))

    assert float(ignored.item()) == 0.0
    assert float(punished.item()) == -math.inf


def test_spherical_score_is_one_for_a_perfect_answer() -> None:
    probabilities = torch.tensor([[0.0, 1.0]])

    assert float(spherical_score(probabilities, probabilities).item()) == pytest.approx(1.0)


def test_ranked_probability_score_prefers_a_near_miss() -> None:
    mask = torch.ones(1, 3)
    near = torch.tensor([[0.0, 1.0, 0.0]])
    far = torch.tensor([[1.0, 0.0, 0.0]])
    target = torch.tensor([[0.0, 0.0, 1.0]])

    assert float(ranked_probability_score(near, target, mask)) < float(
        ranked_probability_score(far, target, mask)
    )


def test_composite_score_only_applies_rps_to_ordinal_rows() -> None:
    logits = torch.tensor([[0.0, 0.0, 9.0], [0.0, 0.0, 9.0]])
    target = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    mask = torch.ones(2, 3, dtype=torch.bool)
    ordinal = torch.tensor([False, True])
    log_probabilities = masked_log_softmax(logits, mask)

    without = composite_score(log_probabilities, target, mask)
    with_ordinal = composite_score(log_probabilities, target, mask, ordinal)

    assert torch.allclose(with_ordinal[0], without[0])
    assert with_ordinal[1] < without[1]


@pytest.mark.parametrize("rare", [1e-9, 1e-5, 2e-4, 0.3])
def test_the_truth_beats_every_report_even_for_a_rare_outcome(rare: float) -> None:
    # The old 1e-4 floor made reporting zero on a rare outcome score higher than
    # the truth for any component below about 2.7e-4. One honest-versus-wrong pair
    # cannot see that, a grid that reaches below the old floor can.
    target = torch.tensor([[rare, 1.0 - rare]], dtype=torch.float64)
    grid = torch.logspace(-12, math.log10(0.5), 2001, dtype=torch.float64)
    reports = torch.stack([grid, 1.0 - grid], dim=-1)
    mask = torch.ones_like(reports, dtype=torch.bool)

    scores = composite_score(reports.log(), target.expand_as(reports), mask)
    honest = composite_score(target.log(), target, mask[:1])

    assert torch.all(scores <= honest + 1e-15)
    assert scores[0] < honest, "deleting the rare outcome must not pay"


@pytest.mark.parametrize("ordinal", [False, True])
def test_the_truth_beats_random_reports_with_four_options(ordinal: bool) -> None:
    generator = torch.Generator().manual_seed(0)
    mask = torch.ones(5001, 4, dtype=torch.bool)
    ordinal_mask = torch.full((5001,), ordinal)

    for trial in range(20):
        target = torch.rand(4, generator=generator, dtype=torch.float64)
        if trial % 2:
            target[0] *= 1e-5  # a component far below the old floor
        target = target / target.sum()
        logits = torch.randn(5000, 4, generator=generator, dtype=torch.float64) * 8
        reports = torch.cat([torch.log_softmax(logits, dim=-1), target.log().unsqueeze(0)])

        scores = composite_score(reports, target.expand_as(reports), mask, ordinal_mask)

        assert scores[:-1].max() <= scores[-1] + 1e-12


@pytest.mark.parametrize("gap", [9.3, 12.0, 50.0, 5000.0])
def test_the_loss_keeps_learning_from_a_confident_miss(gap: float) -> None:
    # With the floor this gradient was 6.9e-5 at a gap of 9.3 and fell from there:
    # a row the model got confidently wrong stopped teaching.
    logits = torch.tensor([[0.0, gap]], requires_grad=True)
    mask = torch.ones(1, 2, dtype=torch.bool)
    target = torch.tensor([[1.0, 0.0]])

    loss = proper_scoring_loss(masked_log_softmax(logits, mask), target, mask)
    loss.backward()

    assert math.isfinite(float(loss.item()))
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 1].item() > 0.99
    assert logits.grad[0, 0].item() < -0.99


def test_proper_scoring_prefers_the_honest_distribution() -> None:
    target = torch.tensor([[0.7, 0.3]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    honest = torch.tensor([[0.7, 0.3]]).log().requires_grad_(True)
    overconfident = torch.tensor([[0.99, 0.01]]).log().requires_grad_(True)

    honest_loss = proper_scoring_loss(honest, target, mask)
    overconfident_loss = proper_scoring_loss(overconfident, target, mask)

    assert honest_loss < overconfident_loss
    honest_loss.backward()
    assert honest.grad is not None


@pytest.mark.parametrize("ordinal", [False, True])
def test_padding_never_changes_a_score(ordinal: bool) -> None:
    logits = torch.tensor([[0.3, -1.2, 2.0]])
    target = torch.tensor([[0.2, 0.1, 0.7]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    ordinal_mask = torch.tensor([ordinal])
    sentinel = torch.finfo(torch.float32).min
    padded_logits = torch.tensor([[0.3, -1.2, 2.0, sentinel, sentinel]], requires_grad=True)
    padded_target = torch.tensor([[0.2, 0.1, 0.7, 0.0, 0.0]])
    padded_mask = torch.tensor([[True, True, True, False, False]])

    plain = composite_score(masked_log_softmax(logits, mask), target, mask, ordinal_mask)
    padded = composite_score(
        masked_log_softmax(padded_logits, padded_mask), padded_target, padded_mask, ordinal_mask
    )
    padded.sum().backward()

    assert torch.allclose(plain, padded)
    assert torch.isfinite(padded_logits.grad).all()
    assert torch.all(padded_logits.grad[0, 3:] == 0)


def test_a_caller_may_mark_padding_with_negative_infinity() -> None:
    # 0 * -inf is NaN. Log-probabilities that arrive with -inf on padded options
    # must not poison the row.
    log_probabilities = torch.tensor([[math.log(0.25), math.log(0.75), -math.inf]])
    target = torch.tensor([[0.25, 0.75, 0.0]])
    mask = torch.tensor([[True, True, False]])

    score = composite_score(log_probabilities, target, mask, torch.tensor([True]))

    assert torch.isfinite(score).all()


def test_targets_must_not_use_padded_options() -> None:
    with pytest.raises(ValueError, match="padded"):
        validate_distributions(
            torch.tensor([[0.5, 0.5]]), torch.tensor([[0.5, 0.5]]), torch.tensor([[1, 0]])
        )


def test_validate_rejects_probabilities_that_do_not_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        validate_distributions(
            torch.tensor([[0.5, 0.2]]), torch.tensor([[1.0, 0.0]]), torch.tensor([[1, 1]])
        )


def test_weights_are_validated() -> None:
    mask = torch.ones(1, 2, dtype=torch.bool)
    log_probabilities = torch.tensor([[0.5, 0.5]]).log()
    target = torch.tensor([[0.5, 0.5]])

    with pytest.raises(ValueError, match="spherical_weight"):
        composite_score(log_probabilities, target, mask, spherical_weight=2.0)
    with pytest.raises(ValueError, match="rps_weight"):
        composite_score(log_probabilities, target, mask, rps_weight=-1.0)


def test_score_shapes_are_validated() -> None:
    mask = torch.ones(1, 2, dtype=torch.bool)
    target = torch.tensor([[0.5, 0.5]])

    with pytest.raises(ValueError, match="identical shape"):
        composite_score(torch.zeros(1, 3), target, mask)
    with pytest.raises(ValueError, match="one value per row"):
        composite_score(target.log(), target, mask, torch.tensor([True, False]))
