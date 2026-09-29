from __future__ import annotations

import itertools

import pytest
import torch

from exu.bandit import (
    BanditConfig,
    bandit_loss,
    behaviour_policy,
    estimate_target,
    sample_answers,
    simulated_bandit_loss,
    simulated_feedback,
    verdict_loss,
)
from exu.model import masked_log_softmax
from exu.scoring import log_score, proper_scoring_loss


def _expected_over_feedback(behaviour, target, estimate_for):
    """Exact expectation of ``estimate_for(k, r)`` over ``k ~ behaviour``, ``r ~ Bern(y_k)``."""
    total = None
    for k, r in itertools.product(range(behaviour.shape[-1]), (0.0, 1.0)):
        chance = target[0, k] if r else 1 - target[0, k]
        weight = behaviour[0, k] * chance
        if weight == 0:
            continue
        value = weight * estimate_for(k, r)
        total = value if total is None else total + value
    return total


def test_behaviour_policy_explores_every_valid_option_and_skips_padding() -> None:
    probabilities = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.2, 0.8, 0.0, 0.0]])
    mask = torch.tensor([[True, True, True, False], [True, True, False, False]])

    behaviour = behaviour_policy(probabilities, mask, exploration=0.1)

    assert torch.allclose(behaviour.sum(dim=-1), torch.ones(2))
    assert torch.all(behaviour[mask] > 0)
    assert torch.all(behaviour[~mask] == 0)
    assert behaviour[0, 1] == pytest.approx(0.1 / 3)


def test_behaviour_policy_rejects_no_exploration() -> None:
    with pytest.raises(ValueError, match="exploration"):
        behaviour_policy(torch.ones(1, 2) / 2, torch.ones(1, 2, dtype=torch.bool), 0.0)
    with pytest.raises(ValueError, match="never drawn"):
        BanditConfig(exploration=0.0)


def test_config_rejects_unknown_estimator_and_no_samples() -> None:
    with pytest.raises(ValueError, match="estimator"):
        BanditConfig(estimator="median")
    with pytest.raises(ValueError, match="samples_per_question"):
        BanditConfig(samples_per_question=0)


@pytest.mark.parametrize("control", ["none", "model"])
def test_estimate_is_unbiased_for_the_target(control: str) -> None:
    target = torch.tensor([[0.6, 0.3, 0.1, 0.0]])
    model = torch.tensor([[0.05, 0.15, 0.8, 0.0]])
    mask = torch.tensor([[True, True, True, False]])
    behaviour = behaviour_policy(model, mask, exploration=0.2)
    baseline = model if control == "model" else None

    def estimate_for(k: int, r: float) -> torch.Tensor:
        return estimate_target(torch.tensor([[k]]), torch.tensor([[r]]), behaviour, baseline)

    expected = _expected_over_feedback(behaviour, target, estimate_for)

    assert torch.allclose(expected, target, atol=1e-6)


def test_doubly_robust_estimate_is_exact_where_the_model_is() -> None:
    target = torch.tensor([[0.7, 0.3]])
    behaviour = behaviour_policy(target, torch.ones(1, 2, dtype=torch.bool), 0.1)

    right = estimate_target(torch.tensor([[0]]), torch.tensor([[1.0]]), behaviour, target)
    wrong = estimate_target(torch.tensor([[0]]), torch.tensor([[0.0]]), behaviour, target)

    # One draw moves the estimate, but less than plain weighting would.
    plain = estimate_target(torch.tensor([[0]]), torch.tensor([[1.0]]), behaviour)
    assert (right - target).abs().max() < (plain - target).abs().max()
    assert wrong[0, 0] < 0.7 < right[0, 0]


def test_estimate_rejects_a_draw_the_behaviour_could_not_make() -> None:
    behaviour = torch.tensor([[1.0, 0.0]])
    with pytest.raises(ValueError, match="zero probability"):
        estimate_target(torch.tensor([[1]]), torch.tensor([[1.0]]), behaviour)


def test_estimate_rejects_mismatched_shapes() -> None:
    behaviour = torch.full((2, 3), 1 / 3)
    with pytest.raises(ValueError, match="shape"):
        estimate_target(torch.zeros(2, 2, dtype=torch.long), torch.zeros(2, 3), behaviour)
    with pytest.raises(ValueError, match="batch size"):
        estimate_target(torch.zeros(1, 2, dtype=torch.long), torch.zeros(1, 2), behaviour)


def test_bandit_loss_keeps_negative_estimate_mass() -> None:
    log_probabilities = torch.log(torch.tensor([[0.5, 0.25, 0.25]]))
    estimate = torch.tensor([[1.5, -0.5, 0.0]])
    mask = torch.ones(1, 3, dtype=torch.bool)

    loss = bandit_loss(log_probabilities, estimate, mask, spherical_weight=0.0)

    # log_score drops the negative component; the bandit loss must not.
    assert log_score(log_probabilities, estimate).item() == pytest.approx(
        1.5 * torch.log(torch.tensor(0.5)).item()
    )
    expected = -(1.5 * torch.log(torch.tensor(0.5)) - 0.5 * torch.log(torch.tensor(0.25)))
    assert loss.item() == pytest.approx(expected.item())


@pytest.mark.parametrize("ordinal", [False, True])
@pytest.mark.parametrize("control", ["none", "model"])
def test_expected_gradient_is_the_full_target_gradient(ordinal: bool, control: str) -> None:
    target = torch.tensor([[0.5, 0.3, 0.2, 0.0]])
    mask = torch.tensor([[True, True, True, False]])
    ordinal_mask = torch.tensor([ordinal])
    logits = torch.tensor([[0.3, -1.2, 0.8, 0.0]])

    def gradient(loss_of) -> torch.Tensor:
        z = logits.clone().requires_grad_(True)
        loss_of(masked_log_softmax(z, mask)).backward()
        return z.grad

    full = gradient(lambda lp: proper_scoring_loss(lp, target, mask, ordinal_mask))
    probabilities = masked_log_softmax(logits, mask).exp().masked_fill(~mask, 0.0)
    behaviour = behaviour_policy(probabilities, mask, exploration=0.1)
    baseline = probabilities if control == "model" else None

    def gradient_for(k: int, r: float) -> torch.Tensor:
        estimate = estimate_target(torch.tensor([[k]]), torch.tensor([[r]]), behaviour, baseline)
        return gradient(lambda lp: bandit_loss(lp, estimate, mask, ordinal_mask))

    expected = _expected_over_feedback(behaviour, target, gradient_for)

    assert torch.allclose(expected, full, atol=1e-5)


def test_simulated_feedback_rewards_at_the_target_rate() -> None:
    target = torch.tensor([[0.0, 0.25, 0.75, 1.0]]).expand(20_000, 4)
    answers = torch.tensor([[0, 1, 2, 3]]).expand(20_000, 4)
    generator = torch.Generator().manual_seed(5)

    rewards = simulated_feedback(target, answers, generator)

    rates = rewards.mean(dim=0)
    assert rates[0] == 0 and rates[3] == 1
    assert rates[1] == pytest.approx(0.25, abs=0.015)
    assert rates[2] == pytest.approx(0.75, abs=0.015)


def test_sample_answers_follows_the_behaviour() -> None:
    behaviour = torch.tensor([[0.0, 1.0, 0.0]])
    answers = sample_answers(behaviour, 50, torch.Generator().manual_seed(1))
    assert answers.shape == (1, 50)
    assert torch.all(answers == 1)


def _train_free_logits(target: torch.Tensor, loss_of, steps: int = 4000) -> torch.Tensor:
    torch.manual_seed(0)
    mask = torch.ones_like(target, dtype=torch.bool)
    logits = torch.zeros_like(target, requires_grad=True)
    optimizer = torch.optim.SGD([logits], lr=0.05)
    for _ in range(steps):
        optimizer.zero_grad()
        # The losses average over rows; scale back so every row moves at lr.
        (loss_of(masked_log_softmax(logits, mask), mask) * target.shape[0]).backward()
        optimizer.step()
    return masked_log_softmax(logits.detach(), mask).exp()


@pytest.mark.parametrize("estimator", ["verdict", "doubly-robust", "importance"])
def test_bandit_training_converges_to_the_honest_distribution(estimator: str) -> None:
    target = torch.tensor([[0.6, 0.3, 0.1]]).expand(64, 3)
    generator = torch.Generator().manual_seed(9)
    config = BanditConfig(exploration=0.1, estimator=estimator)

    learned = _train_free_logits(
        target,
        lambda lp, mask: simulated_bandit_loss(
            lp, target, mask, config=config, generator=generator
        ),
    )

    assert torch.allclose(learned.mean(dim=0), target[0], atol=0.05)


def test_rewarding_the_sampled_answer_directly_is_not_honest() -> None:
    # The estimator exists because the obvious policy gradient, REINFORCE on the
    # judge's verdict, maximizes expected accuracy and collapses onto the argmax.
    target = torch.tensor([[0.6, 0.3, 0.1]]).expand(64, 3)
    generator = torch.Generator().manual_seed(9)

    def reinforce(lp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        answers = sample_answers(lp.detach().exp(), 1, generator)
        rewards = simulated_feedback(target, answers, generator)
        advantage = rewards - rewards.mean()
        return -(advantage * lp.gather(-1, answers)).mean()

    learned = _train_free_logits(target, reinforce)

    assert learned.mean(dim=0)[0] > 0.9


def _expected_verdict_loss(logits, target, mask, behaviour):
    def loss_for(k: int, r: float) -> torch.Tensor:
        lp = masked_log_softmax(logits, mask)
        return verdict_loss(lp, torch.tensor([[k]]), torch.tensor([[r]]), mask)

    return _expected_over_feedback(behaviour, target, loss_for)


def test_expected_verdict_loss_is_minimized_by_the_target() -> None:
    target = torch.tensor([[0.5, 0.3, 0.2, 0.0]])
    mask = torch.tensor([[True, True, True, False]])
    # A behaviour far from the target: the optimum must not depend on it.
    behaviour = torch.tensor([[0.1, 0.1, 0.8, 0.0]])
    honest = torch.log(target.clamp_min(1e-30)).masked_fill(~mask, 0.0).requires_grad_(True)

    at_target = _expected_verdict_loss(honest, target, mask, behaviour)
    at_target.backward()

    assert honest.grad.abs().max() < 1e-6
    generator = torch.Generator().manual_seed(4)
    for _ in range(20):
        moved = honest.detach() + 0.3 * torch.randn(1, 4, generator=generator)
        assert _expected_verdict_loss(moved, target, mask, behaviour) > at_target.detach()


def test_verdict_loss_is_the_binary_log_score_of_the_answer() -> None:
    log_probabilities = torch.log(torch.tensor([[0.5, 0.25, 0.25]]))
    mask = torch.ones(1, 3, dtype=torch.bool)
    answers = torch.tensor([[0, 1]])
    rewards = torch.tensor([[1.0, 0.0]])

    loss = verdict_loss(log_probabilities, answers, rewards, mask)

    expected = -(torch.log(torch.tensor(0.5)) + torch.log(torch.tensor(0.75))) / 2
    assert loss.item() == pytest.approx(expected.item())


def test_verdict_loss_prices_a_confident_wrong_answer_exactly() -> None:
    # 1 - q_k rounds to zero in float32 here; the log-sum-exp of the rest does not.
    logits = torch.tensor([[40.0, 0.0, 0.0]], requires_grad=True)
    mask = torch.ones(1, 3, dtype=torch.bool)
    log_probabilities = masked_log_softmax(logits, mask)

    loss = verdict_loss(log_probabilities, torch.tensor([[0]]), torch.tensor([[0.0]]), mask)
    loss.backward()

    assert loss.item() == pytest.approx(40.0 - torch.log(torch.tensor(2.0)).item(), rel=1e-4)
    assert torch.all(torch.isfinite(logits.grad))


def test_verdict_loss_on_a_single_option_question_has_no_nan() -> None:
    logits = torch.tensor([[1.0, 0.0]], requires_grad=True)
    mask = torch.tensor([[True, False]])
    log_probabilities = masked_log_softmax(logits, mask)

    loss = verdict_loss(log_probabilities, torch.tensor([[0]]), torch.tensor([[1.0]]), mask)
    loss.backward()

    assert loss.item() == pytest.approx(0.0, abs=1e-6)
    assert torch.all(torch.isfinite(logits.grad))


def test_verdict_loss_rejects_an_answer_on_a_padded_option() -> None:
    mask = torch.tensor([[True, True, False]])
    with pytest.raises(ValueError, match="padded"):
        verdict_loss(torch.zeros(1, 3), torch.tensor([[2]]), torch.tensor([[1.0]]), mask)
