"""Strictly proper scoring rules: the reward that makes honesty optimal.

A scoring rule takes a reported distribution ``q`` and the outcome ``y`` and
returns a number. It is *strictly proper* when the expected score is maximized
only by reporting the true distribution. That property is the whole point of the
reward here: whatever the optimizer does, the only way to raise it is to be
honest about the probabilities.

Contrast with a binary reward (1 for a correct guess, 0 otherwise): its expected
value is linear in ``q``, so the optimum is to dump all mass on the most likely
class. Naive RL maximizes accuracy and destroys calibration.

Three rules are combined here:

- log score: ``sum(y * log q)``. Punishes low probability on what happened.
  Bounded below by ``log(log_floor)`` so one confident miss cannot dominate.
- spherical score: ``<y, q> / ||q||``, in ``[0, 1]``. Rewards mass in the right
  place without the log's gradient spikes.
- ranked probability score: quadratic distance between cumulative distributions.
  Ordinal questions only; it teaches that missing by one level beats missing by
  three.
"""

from __future__ import annotations

import torch
from torch import Tensor


def validate_distributions(probabilities: Tensor, target: Tensor, option_mask: Tensor) -> None:
    """Reject malformed predictions or targets before they corrupt training."""
    if target.shape != option_mask.shape:
        raise ValueError("target and option_mask must have identical shape")
    if probabilities.shape != target.shape:
        raise ValueError("probabilities and target must have identical shape")
    if torch.any(target < 0):
        raise ValueError("target probabilities cannot be negative")
    if torch.any(target.masked_select(~option_mask.bool()) != 0):
        raise ValueError("target assigns mass to padded options")
    if torch.any(probabilities < 0):
        raise ValueError("predicted probabilities cannot be negative")
    if torch.any(probabilities.masked_select(~option_mask.bool()) != 0):
        raise ValueError("predictions assign mass to padded options")
    ones = torch.ones_like(target.sum(dim=-1))
    if not torch.allclose(target.sum(dim=-1), ones, atol=1e-5):
        raise ValueError("each target distribution must sum to one")
    if not torch.allclose(probabilities.sum(dim=-1), ones, atol=1e-5):
        raise ValueError("predicted probabilities must sum to one")


def log_score(probabilities: Tensor, target: Tensor, log_floor: float = 1e-4) -> Tensor:
    """Log score per row. Higher is better. ``log_floor`` bounds the penalty."""
    if not 0 < log_floor <= 1:
        raise ValueError("log_floor must be in (0, 1]")
    return (target * probabilities.clamp_min(log_floor).log()).sum(dim=-1)


def spherical_score(probabilities: Tensor, target: Tensor) -> Tensor:
    """Spherical score per row, in ``[0, 1]``. Higher is better."""
    eps = torch.finfo(probabilities.dtype).tiny
    return (target * probabilities).sum(dim=-1) / probabilities.norm(dim=-1).clamp_min(eps)


def ranked_probability_score(probabilities: Tensor, target: Tensor, option_mask: Tensor) -> Tensor:
    """Ranked probability score per row. Lower is better, in ``[0, 1]``."""
    cumulative_error = (probabilities.cumsum(dim=-1) - target.cumsum(dim=-1)).square()
    valid_count = option_mask.sum(dim=-1).sub(1).clamp_min(1)
    return (cumulative_error * option_mask).sum(dim=-1) / valid_count


def composite_score(
    probabilities: Tensor,
    target: Tensor,
    option_mask: Tensor,
    ordinal_mask: Tensor | None = None,
    *,
    spherical_weight: float = 0.75,
    rps_weight: float = 1.0,
    log_floor: float = 1e-4,
) -> Tensor:
    """Per-row strictly proper reward. Higher is better, one value per row.

    Every question gets log and spherical score. Ordinal questions additionally
    subtract ranked probability score, so near misses beat far ones.
    """
    _validate_weights(spherical_weight, rps_weight)
    score = log_score(probabilities, target, log_floor) + spherical_weight * spherical_score(
        probabilities, target
    )
    if ordinal_mask is not None:
        if ordinal_mask.shape != score.shape:
            raise ValueError("ordinal_mask must have one value per row")
        rps = ranked_probability_score(probabilities, target, option_mask)
        score = score - rps_weight * rps * ordinal_mask.to(score.dtype)
    return score


def proper_scoring_loss(
    probabilities: Tensor,
    target: Tensor,
    option_mask: Tensor,
    ordinal_mask: Tensor | None = None,
    *,
    spherical_weight: float = 0.75,
    rps_weight: float = 1.0,
    log_floor: float = 1e-4,
) -> Tensor:
    """Negative mean composite score, for the direct (non-RL) baseline."""
    score = composite_score(
        probabilities,
        target,
        option_mask,
        ordinal_mask,
        spherical_weight=spherical_weight,
        rps_weight=rps_weight,
        log_floor=log_floor,
    )
    return -score.mean()


def _validate_weights(spherical_weight: float, rps_weight: float) -> None:
    if not 0 <= spherical_weight <= 1:
        raise ValueError("spherical_weight must be between zero and one")
    if rps_weight < 0:
        raise ValueError("rps_weight cannot be negative")
