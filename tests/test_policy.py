from __future__ import annotations

import math

import pytest
import torch

from exu.policy import (
    PolicyConfig,
    gaussian_log_prob,
    group_advantages,
    policy_gradient_loss,
    sample_perturbations,
    sigma_for,
)


def test_sigma_schedule_anneals_linearly() -> None:
    assert sigma_for(0, 10, 1.0, 0.3) == pytest.approx(1.0)
    assert sigma_for(5, 10, 1.0, 0.3) == pytest.approx(0.65)
    assert sigma_for(10, 10, 1.0, 0.3) == pytest.approx(0.3)
    assert sigma_for(50, 10, 1.0, 0.3) == pytest.approx(0.3)


def test_sigma_schedule_validates_its_inputs() -> None:
    with pytest.raises(ValueError, match="total_steps"):
        sigma_for(0, 0, 1.0, 0.3)
    with pytest.raises(ValueError, match="step"):
        sigma_for(-1, 10, 1.0, 0.3)


def test_perturbations_are_masked_and_sum_to_zero() -> None:
    logits = torch.zeros(2, 4)
    mask = torch.tensor([[True, True, True, False], [True, True, True, True]])
    generator = torch.Generator().manual_seed(3)

    noise = sample_perturbations(logits, mask, samples=64, generator=generator)

    assert noise.shape == (2, 64, 4)
    assert torch.all(noise[0, :, 3] == 0)
    assert torch.allclose(noise.sum(dim=-1), torch.zeros(2, 64), atol=1e-5)


def test_gaussian_log_prob_gradient_is_the_score_function() -> None:
    logits = torch.zeros(2, 3, requires_grad=True)
    mask = torch.tensor([[True, True, True], [True, True, False]])
    sigma = 0.7
    generator = torch.Generator().manual_seed(11)
    noise = sample_perturbations(logits.detach(), mask, samples=4, generator=generator)
    sample = (logits.detach().unsqueeze(1) + sigma * noise).detach()

    log_probability = gaussian_log_prob(sample, logits, sigma, mask)
    log_probability.sum().backward()

    expected = (sample - logits.detach().unsqueeze(1)) / sigma**2
    expected = expected * mask.unsqueeze(1)
    assert torch.allclose(logits.grad, expected.sum(dim=1), atol=1e-6)


def test_gaussian_log_prob_rejects_a_non_positive_sigma() -> None:
    with pytest.raises(ValueError, match="sigma"):
        gaussian_log_prob(
            torch.zeros(1, 1, 2), torch.zeros(1, 2), 0.0, torch.ones(1, 2, dtype=torch.bool)
        )


def test_group_advantages_center_and_scale() -> None:
    rewards = torch.tensor([[1.0, 2.0, 3.0], [5.0, 5.0, 5.0]])

    grouped = group_advantages(rewards, "group")

    assert grouped[0].mean().item() == pytest.approx(0.0, abs=1e-6)
    assert grouped[0].std(unbiased=False).item() == pytest.approx(1.0, abs=1e-6)
    assert torch.allclose(grouped[1], torch.zeros(3))


def test_group_advantages_none_returns_raw_rewards() -> None:
    rewards = torch.tensor([[1.0, 2.0]])

    assert torch.allclose(group_advantages(rewards, "none"), rewards)


def test_group_advantages_reject_bad_input() -> None:
    with pytest.raises(ValueError, match="mode"):
        group_advantages(torch.ones(1, 2), "median")
    with pytest.raises(ValueError, match=r"\(questions, samples\)"):
        group_advantages(torch.ones(2), "group")


def test_policy_config_rejects_a_single_sample_with_advantages() -> None:
    with pytest.raises(ValueError, match="two samples"):
        PolicyConfig(samples_per_question=1, advantage_norm="group")


def test_policy_config_presets_match_the_documented_recipes() -> None:
    base = PolicyConfig.base()
    finetune = PolicyConfig.finetune()

    assert (base.samples_per_question, base.sigma_start, base.sigma_end) == (8, 1.0, 0.3)
    assert base.cross_entropy_weight == 0.0
    assert (finetune.samples_per_question, finetune.sigma_start, finetune.sigma_end) == (
        4,
        0.4,
        0.1,
    )
    assert finetune.cross_entropy_weight == 1.0


def test_policy_gradient_loss_is_finite_and_differentiable() -> None:
    torch.manual_seed(0)
    logits = torch.randn(3, 3, requires_grad=True)
    mask = torch.tensor([[True, True, False], [True, True, True], [True, True, True]])
    target = torch.tensor([[0.5, 0.5, 0.0], [0.1, 0.2, 0.7], [0.3, 0.3, 0.4]])
    generator = torch.Generator().manual_seed(1)

    loss, metrics = policy_gradient_loss(logits, target, mask, sigma=0.4, generator=generator)
    loss.backward()

    assert math.isfinite(metrics.loss)
    assert metrics.cross_entropy > 0
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.all(logits.grad[0, 2] == 0)


def test_pure_policy_gradient_learns_the_target_without_cross_entropy() -> None:
    generator = torch.Generator().manual_seed(5)
    logits = torch.zeros(1, 3, requires_grad=True)
    target = torch.tensor([[0.1, 0.8, 0.1]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    config = PolicyConfig(
        samples_per_question=16,
        sigma_start=0.5,
        sigma_end=0.5,
        cross_entropy_weight=0.0,
    )
    optimizer = torch.optim.Adam([logits], lr=0.05)

    for _ in range(300):
        optimizer.zero_grad()
        loss, _metrics = policy_gradient_loss(
            logits, target, mask, config=config, sigma=0.5, generator=generator
        )
        loss.backward()
        optimizer.step()

    probabilities = torch.softmax(logits.detach(), dim=-1)
    assert probabilities.argmax().item() == 1
    assert probabilities[0, 1].item() > 0.5


def test_cross_entropy_weight_changes_the_objective() -> None:
    torch.manual_seed(2)
    logits = torch.zeros(1, 3, requires_grad=True)
    target = torch.tensor([[0.1, 0.8, 0.1]])
    mask = torch.ones(1, 3, dtype=torch.bool)

    pure, _ = policy_gradient_loss(
        logits,
        target,
        mask,
        config=PolicyConfig(cross_entropy_weight=0.0, samples_per_question=2),
        sigma=0.4,
        generator=torch.Generator().manual_seed(9),
    )
    hybrid, metrics = policy_gradient_loss(
        logits,
        target,
        mask,
        config=PolicyConfig(cross_entropy_weight=1.0, samples_per_question=2),
        sigma=0.4,
        generator=torch.Generator().manual_seed(9),
    )

    assert metrics.cross_entropy > 0
    assert hybrid.item() == pytest.approx(pure.item() + metrics.cross_entropy, abs=1e-5)
