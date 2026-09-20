"""RLCD: policy-gradient training with a strictly proper reward.

The training loop is deliberately small. For each question the model scores its
options, then:

1. sample ``G`` Gaussian perturbations of the logits with standard deviation
   ``sigma``; mask them to the valid options and project them to sum to zero,
   because adding a constant to logits does not change the softmax and that
   degree of freedom would only add variance;
2. softmax each perturbed version: ``G`` candidate distributions per question;
3. score every candidate against the target with a strictly proper scoring rule,
   with no gradient;
4. advantage: reward minus the group baseline, normalized by a standard
   deviation (Laya's fine-tune normalizes by the batch, GRPO by the
   group; both are available);
5. the loss is the negative mean of advantage times the Gaussian log-probability
   of the sample given the current logits. The sample is detached, so the
   gradient flows only through the current logits, and the sign of the advantage
   pushes logits toward perturbations that scored above the baseline.

Sample perturbations that paid off and the logits move toward them; ones that
did not, and they move away, with no direct supervision from the target.

An optional cross-entropy term against the soft target can be added with full
weight. Laya's fine-tune trains hybrid, not pure policy, so the default here is
hybrid too. Set ``cross_entropy_weight=0`` for pure policy-gradient base
training.

A caveat worth repeating, because it is easy to overclaim: the reward is
differentiable in the logits, so nothing forces a sampled estimator here. The
sampling adds noise that acts as a regularizer; it is not what makes the model
honest. Honesty comes from the scoring rule. If pure policy does not beat the
direct baseline on held-out ECE or NLL, keep the direct baseline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .model import masked_softmax
from .scoring import composite_score

_ADVANTAGE_MODES = ("batch", "group", "none")


@dataclass(frozen=True, slots=True)
class PolicyConfig:
    """Hyperparameters of the perturbed-logit policy."""

    samples_per_question: int = 4
    sigma_start: float = 0.4
    sigma_end: float = 0.1
    cross_entropy_weight: float = 1.0
    spherical_weight: float = 0.75
    rps_weight: float = 1.0
    log_floor: float = 1e-4
    advantage_norm: str = "batch"

    def __post_init__(self) -> None:
        if self.samples_per_question < 1:
            raise ValueError("samples_per_question must be positive")
        if self.sigma_start <= 0 or self.sigma_end < 0:
            raise ValueError("sigma values must be positive")
        if self.cross_entropy_weight < 0:
            raise ValueError("cross_entropy_weight cannot be negative")
        if self.advantage_norm not in _ADVANTAGE_MODES:
            raise ValueError(f"advantage_norm must be one of {_ADVANTAGE_MODES}")
        if self.advantage_norm != "none" and self.samples_per_question < 2:
            raise ValueError("advantage estimation needs at least two samples per question")

    @classmethod
    def base(cls) -> PolicyConfig:
        """Pure policy-gradient base training: G=8, sigma from 1.0 to 0.3."""
        return cls(
            samples_per_question=8,
            sigma_start=1.0,
            sigma_end=0.3,
            cross_entropy_weight=0.0,
        )

    @classmethod
    def finetune(cls) -> PolicyConfig:
        """Hybrid fine-tune: G=4, sigma from 0.4 to 0.1, cross-entropy at 1.0."""
        return cls()


@dataclass(frozen=True, slots=True)
class PolicyMetrics:
    """Diagnostics of one policy-gradient step."""

    loss: float
    policy_loss: float
    cross_entropy: float
    mean_reward: float
    mean_advantage: float
    sigma: float


def sigma_for(step: int, total_steps: int, start: float, end: float) -> float:
    """Linearly anneal the exploration standard deviation over training."""
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    if step < 0:
        raise ValueError("step cannot be negative")
    progress = min(1.0, step / total_steps)
    return start + (end - start) * progress


def sample_perturbations(
    logits: Tensor,
    option_mask: Tensor,
    samples: int,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Draw masked, zero-sum-projected standard-normal noise of shape (B, G, K)."""
    if samples < 1:
        raise ValueError("samples must be positive")
    mask = option_mask.bool().unsqueeze(1)
    shape = (logits.size(0), samples, logits.size(-1))
    if generator is None:
        noise = torch.randn(shape, dtype=logits.dtype, device=logits.device)
    else:
        noise = torch.randn(shape, generator=generator, dtype=logits.dtype, device=logits.device)
    noise = noise * mask
    count = option_mask.sum(dim=-1).clamp_min(1).to(noise.dtype).reshape(-1, 1, 1)
    mean = noise.sum(dim=-1, keepdim=True) / count
    return (noise - mean) * mask


def gaussian_log_prob(sample: Tensor, logits: Tensor, sigma: float, option_mask: Tensor) -> Tensor:
    """Log-probability of ``sample`` under ``N(logits, sigma^2)`` per valid option.

    ``sample`` must be detached; the gradient flows through ``logits`` only, so
    this is the score-function term ``(sample - logits) / sigma^2``.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    mask = option_mask.bool().unsqueeze(1)
    # Mask before dividing: the padded logits sit at the dtype minimum, and
    # dividing them by sigma would overflow to -inf and turn 0 * inf into NaN.
    residual = (sample - logits.unsqueeze(1)) * mask
    residual = residual / sigma
    squared = residual.square()
    count = option_mask.sum(dim=-1).to(residual.dtype).reshape(-1, 1)
    normalizer = -0.5 * count * math.log(2.0 * math.pi) - count * math.log(sigma)
    return -0.5 * squared.sum(dim=-1) + normalizer


def group_advantages(rewards: Tensor, mode: str = "batch", eps: float = 1e-8) -> Tensor:
    """Advantage of each sample against its question's mean reward.

    ``batch`` divides by the standard deviation of the whole batch (Laya's
    fine-tune), ``group`` by each question's own standard deviation (GRPO),
    ``none`` returns raw rewards with no baseline.
    """
    if rewards.dim() != 2:
        raise ValueError("rewards must be a (questions, samples) tensor")
    if mode not in _ADVANTAGE_MODES:
        raise ValueError(f"mode must be one of {_ADVANTAGE_MODES}")
    if mode == "none":
        return rewards
    centered = rewards - rewards.mean(dim=1, keepdim=True)
    if mode == "group":
        deviation = rewards.std(dim=1, keepdim=True, unbiased=False)
    else:
        deviation = rewards.std(unbiased=False)
    return centered / deviation.clamp_min(eps)


def policy_gradient_loss(
    logits: Tensor,
    target: Tensor,
    option_mask: Tensor,
    *,
    config: PolicyConfig | None = None,
    sigma: float = 0.4,
    ordinal_mask: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, PolicyMetrics]:
    """Return the RLCD loss for one batch and its diagnostics.

    ``logits`` must require grad. The returned loss is
    ``policy_loss + cross_entropy_weight * cross_entropy``.
    """
    settings = config or PolicyConfig()
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    mask = option_mask.bool()
    batch, options = logits.shape
    samples = settings.samples_per_question

    noise = sample_perturbations(logits.detach(), mask, samples, generator)
    sampled = (logits.detach().unsqueeze(1) + sigma * noise).masked_fill(
        ~mask.unsqueeze(1), torch.finfo(logits.dtype).min
    )
    candidates = torch.softmax(sampled, dim=-1).masked_fill(~mask.unsqueeze(1), 0.0)

    flat_candidates = candidates.reshape(batch * samples, options)
    flat_target = (
        target.unsqueeze(1).expand(batch, samples, options).reshape(batch * samples, options)
    )
    flat_mask = mask.unsqueeze(1).expand(batch, samples, options).reshape(batch * samples, options)
    flat_ordinal = (
        None
        if ordinal_mask is None
        else ordinal_mask.unsqueeze(1).expand(batch, samples).reshape(batch * samples)
    )
    rewards = composite_score(
        flat_candidates,
        flat_target,
        flat_mask,
        flat_ordinal,
        spherical_weight=settings.spherical_weight,
        rps_weight=settings.rps_weight,
        log_floor=settings.log_floor,
    ).reshape(batch, samples)

    advantages = group_advantages(rewards.detach(), settings.advantage_norm)
    log_probability = gaussian_log_prob(sampled, logits, sigma, mask)
    policy_loss = -(advantages.detach() * log_probability).mean()

    cross_entropy = torch.zeros((), dtype=logits.dtype, device=logits.device)
    if settings.cross_entropy_weight > 0:
        probabilities = masked_softmax(logits, mask)
        cross_entropy = (
            -(target * probabilities.clamp_min(settings.log_floor).log()).sum(dim=-1).mean()
        )
    loss = policy_loss + settings.cross_entropy_weight * cross_entropy

    metrics = PolicyMetrics(
        loss=float(loss.detach().item()),
        policy_loss=float(policy_loss.detach().item()),
        cross_entropy=float(cross_entropy.detach().item()),
        mean_reward=float(rewards.detach().mean().item()),
        mean_advantage=float(advantages.detach().mean().item()),
        sigma=float(sigma),
    )
    return loss, metrics
