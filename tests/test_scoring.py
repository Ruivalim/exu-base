from __future__ import annotations

import math

import pytest
import torch

from exu.scoring import (
    composite_score,
    log_score,
    proper_scoring_loss,
    ranked_probability_score,
    spherical_score,
    validate_distributions,
)


def test_log_score_is_bounded_by_the_floor() -> None:
    probabilities = torch.tensor([[0.0, 1.0]])
    target = torch.tensor([[1.0, 0.0]])

    score = log_score(probabilities, target, log_floor=1e-4)

    assert math.isclose(float(score.item()), math.log(1e-4), rel_tol=1e-6)


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
    probabilities = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    target = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    mask = torch.ones(2, 3)
    ordinal = torch.tensor([False, True])

    without = composite_score(probabilities, target, mask)
    with_ordinal = composite_score(probabilities, target, mask, ordinal)

    assert torch.allclose(with_ordinal[0], without[0])
    assert with_ordinal[1] < without[1]


def test_proper_scoring_prefers_the_honest_distribution() -> None:
    target = torch.tensor([[0.7, 0.3]])
    mask = torch.ones(1, 2)
    honest = torch.tensor([[0.7, 0.3]], requires_grad=True)
    overconfident = torch.tensor([[0.99, 0.01]], requires_grad=True)

    honest_loss = proper_scoring_loss(honest, target, mask)
    overconfident_loss = proper_scoring_loss(overconfident, target, mask)

    assert honest_loss < overconfident_loss
    honest_loss.backward()
    assert honest.grad is not None


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
    mask = torch.ones(1, 2)
    probabilities = torch.tensor([[0.5, 0.5]])
    target = torch.tensor([[0.5, 0.5]])

    with pytest.raises(ValueError, match="spherical_weight"):
        composite_score(probabilities, target, mask, spherical_weight=2.0)
    with pytest.raises(ValueError, match="rps_weight"):
        composite_score(probabilities, target, mask, rps_weight=-1.0)
    with pytest.raises(ValueError, match="log_floor"):
        log_score(probabilities, target, log_floor=0.0)
