"""Learning a calibrated distribution from feedback on one sampled answer.

The RLCD policy in :mod:`exu.policy` scores its samples against the whole target
distribution. With the whole target at hand the direct loss is the better
estimator, and it has won every comparison in ``BENCHMARKS.md``. A sampled
estimator earns its place only when the target is never seen: the model picks an
answer, and something outside it (a judge, a test that passed, a user who
accepted) says whether that one answer was right. That is bandit feedback.

Rewarding the right answer directly breaks the property the project exists for.
The expected reward of sampling ``k ~ q`` is ``sum_k q_k y_k``, linear in ``q``,
so its optimum puts all mass on the most likely option. This module keeps the
proper score instead and estimates the target it needs from the feedback.

Let ``y_k`` be the probability that answer ``k`` is judged right, so ``y`` is the
target distribution the direct loss would read. Sample ``k`` from a behaviour
policy ``pi`` and observe ``r`` in ``{0, 1}``. Then

    y_hat = r * e_k / pi_k                      (importance weighted)
    y_hat = b + (r - b_k) * e_k / pi_k          (doubly robust, any fixed b)

both satisfy ``E[y_hat] = y``. The log and spherical scores are linear in ``y``,
and the ranked probability score is linear in ``y`` up to a term that does not
depend on the model, so the composite loss against ``y_hat`` has the gradient of
the loss against ``y`` in expectation, and the same optimum: honest
probabilities. ``b`` defaults to the model's own detached distribution, which
makes ``y_hat`` exact wherever the model already is.

Two conditions carry the unbiasedness, and both are enforced here:

- every valid option must have a positive probability under ``pi``, or its target
  mass is never observed. ``pi`` mixes the model with a uniform share
  ``exploration > 0``, which also bounds the weight ``1 / pi_k`` by
  ``K / exploration``;
- ``pi`` and ``b`` must not depend on the draw they weight, so both are detached.

``y_hat`` is not a distribution: it can exceed one and, with a baseline, go
negative. It is scored by :func:`bandit_loss`, never by
:func:`exu.scoring.log_score`, which skips non-positive target mass by design and
would drop the negative half of the estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .scoring import ranked_probability_score, spherical_score

_CONTROLS = ("none", "model")


@dataclass(frozen=True, slots=True)
class BanditConfig:
    """How answers are drawn and how the feedback becomes a target estimate.

    ``control="model"`` is the doubly robust estimator with the model's own
    distribution as baseline; ``"none"`` is plain importance weighting.
    """

    samples_per_question: int = 1
    exploration: float = 0.1
    control: str = "model"

    def __post_init__(self) -> None:
        if self.samples_per_question < 1:
            raise ValueError("samples_per_question must be at least one")
        if not 0 < self.exploration <= 1:
            raise ValueError("exploration must be in (0, 1]: an option never drawn is never seen")
        if self.control not in _CONTROLS:
            raise ValueError(f"control must be one of {_CONTROLS}")


def behaviour_policy(probabilities: Tensor, option_mask: Tensor, exploration: float) -> Tensor:
    """Mix the model with a uniform share over the valid options, detached."""
    if not 0 < exploration <= 1:
        raise ValueError("exploration must be in (0, 1]")
    mask = option_mask.bool()
    uniform = mask.to(probabilities.dtype) / mask.sum(dim=-1, keepdim=True).clamp_min(1)
    model = probabilities.detach().masked_fill(~mask, 0.0)
    model = model / model.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(model.dtype).tiny)
    return (1 - exploration) * model + exploration * uniform


def sample_answers(
    behaviour: Tensor, samples: int, generator: torch.Generator | None = None
) -> Tensor:
    """Draw ``samples`` answer indices per question, shape ``(B, samples)``."""
    return torch.multinomial(behaviour, samples, replacement=True, generator=generator)


def simulated_feedback(
    target: Tensor, answers: Tensor, generator: torch.Generator | None = None
) -> Tensor:
    """Judge each answer right with probability equal to its target mass.

    For a target that is the share of annotators who chose each option, this is
    one annotator drawn at random saying whether they agree with the answer. It
    exists to measure the estimator on data that has a target; with real
    feedback the rewards come from outside and the target is never read.
    """
    chance = target.gather(-1, answers).clamp(0.0, 1.0)
    draw = torch.rand(chance.shape, generator=generator, device=chance.device, dtype=chance.dtype)
    return (draw < chance).to(target.dtype)


def estimate_target(
    answers: Tensor,
    rewards: Tensor,
    behaviour: Tensor,
    baseline: Tensor | None = None,
) -> Tensor:
    """Unbiased estimate of the target from rewards on sampled answers.

    ``answers`` and ``rewards`` are ``(B, G)``; ``behaviour`` is the ``(B, K)``
    distribution the answers were drawn from. The ``G`` single-draw estimates are
    averaged. ``baseline`` is ``b`` in the doubly robust form, zero when omitted.
    """
    if answers.shape != rewards.shape or answers.dim() != 2:
        raise ValueError("answers and rewards must share a (batch, samples) shape")
    if answers.shape[0] != behaviour.shape[0]:
        raise ValueError("answers and behaviour must have the same batch size")
    reference = torch.zeros_like(behaviour) if baseline is None else baseline.detach()
    propensity = behaviour.gather(-1, answers)
    if torch.any(propensity <= 0):
        raise ValueError("an answer was drawn with zero probability under the behaviour policy")
    residual = (rewards - reference.gather(-1, answers)) / propensity
    correction = torch.zeros_like(behaviour).scatter_add_(-1, answers, residual)
    return reference + correction / answers.shape[1]


def bandit_loss(
    log_probabilities: Tensor,
    estimate: Tensor,
    option_mask: Tensor,
    ordinal_mask: Tensor | None = None,
    *,
    spherical_weight: float = 0.75,
    rps_weight: float = 1.0,
) -> Tensor:
    """Negative mean composite score against a target estimate.

    The same composite as :func:`exu.scoring.composite_score`, written to keep the
    negative and above-one components of ``estimate`` instead of skipping them.
    """
    if log_probabilities.shape != estimate.shape or estimate.shape != option_mask.shape:
        raise ValueError("log_probabilities, estimate and option_mask must have identical shape")
    mask = option_mask.bool()
    estimate = estimate.masked_fill(~mask, 0.0)
    probabilities = log_probabilities.exp().masked_fill(~mask, 0.0)
    score = (estimate * log_probabilities.masked_fill(~mask, 0.0)).sum(dim=-1)
    score = score + spherical_weight * spherical_score(probabilities, estimate)
    if ordinal_mask is not None:
        if ordinal_mask.shape != score.shape:
            raise ValueError("ordinal_mask must have one value per row")
        rps = ranked_probability_score(probabilities, estimate, option_mask)
        score = score - rps_weight * rps * ordinal_mask.to(score.dtype)
    return -score.mean()


def simulated_bandit_loss(
    log_probabilities: Tensor,
    target: Tensor,
    option_mask: Tensor,
    ordinal_mask: Tensor | None = None,
    *,
    config: BanditConfig | None = None,
    spherical_weight: float = 0.75,
    rps_weight: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Draw answers, simulate feedback from ``target``, score against the estimate.

    The only read of ``target`` is inside :func:`simulated_feedback`, one Bernoulli
    draw per sampled answer, so the model learns from what bandit feedback would
    give it and nothing more.
    """
    settings = config or BanditConfig()
    probabilities = log_probabilities.detach().exp()
    behaviour = behaviour_policy(probabilities, option_mask, settings.exploration)
    answers = sample_answers(behaviour, settings.samples_per_question, generator)
    rewards = simulated_feedback(target, answers, generator)
    baseline = probabilities.masked_fill(~option_mask.bool(), 0.0)
    estimate = estimate_target(
        answers, rewards, behaviour, baseline if settings.control == "model" else None
    )
    return bandit_loss(
        log_probabilities,
        estimate,
        option_mask,
        ordinal_mask,
        spherical_weight=spherical_weight,
        rps_weight=rps_weight,
    )
