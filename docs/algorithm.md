# Reward and policy-gradient math

This is what the code in `rlcd/scoring.py` and `rlcd/policy.py` computes, and why.

Notation: `y` is the target distribution over `K` options (one value per option,
summing to one), `q` is the model's distribution, `p` is a valid-option mask, and
`z` are the option logits.

## The reward

### Log score

```
S_log(q, y) = sum_k y_k * log(max(q_k, floor))
```

Bounded below by `log(floor)` (default `floor = 1e-4`, so about -9.21), so one
confident miss cannot dominate the batch. The floor costs a small indifference
region near zero probability, which is the usual price for a bounded log score.

The log score is the negative cross-entropy, so a direct cross-entropy baseline
and a log-score reward optimize the same quantity up to sign.

### Spherical score

```
S_sph(q, y) = <y, q> / ||q||
```

In `[0, 1]`, one for an exact answer. It rewards mass in the right place without
the log's gradient spikes near zero probability.

### Ranked probability score

For ordinal questions only:

```
RPS(q, y) = (1 / (K - 1)) * sum_k (CDF(q)_k - CDF(y)_k)^2
```

Lower is better, in `[0, 1]`. It teaches the geometry of the rubric: missing by
one level is cheaper than missing by three.

### Composite

```
S(q, y) = S_log + w_sph * S_sph - w_rps * RPS * 1[ordinal]
```

Defaults `w_sph = 0.75`, `w_rps = 1.0`. Every term is strictly proper, and a
positive combination of strictly proper scores is strictly proper. The reward is
one number per row, and higher is better. `proper_scoring_loss` returns its
negation for direct training.

## The policy

The model's action is not a token, it is a perturbation of its own logits. The
policy is a Gaussian centered on the current logits:

```
eps ~ N(0, I)                    # standard normal, shape (G, K)
eps_valid = eps * p
eps_projected = (eps_valid - mean_valid(eps_valid)) * p
a = z.detach() + sigma * eps_projected     # the sampled action
q_g = softmax(a)                            # one candidate distribution per sample
R_g = S(q_g, y)                             # reward, no gradient
```

Masking zeroes the padded options. Projection removes the component along the
all-ones direction, which the softmax ignores anyway; without it, that degree of
freedom would only add variance.

Advantage, per question over its `G` samples:

```
A_g = (R_g - mean_g R) / std
```

`std` is either the batch-wide standard deviation (`advantage_norm="batch"`, Laya's
fine-tune) or the per-question standard deviation
(`advantage_norm="group"`, GRPO). `advantage_norm="none"` returns raw rewards.

The surrogate loss:

```
log pi(a | z) = sum_k -0.5 * ((a_k - z_k) / sigma)^2 * p_k  + const
L_policy = - mean_g [ A_g.detach() * log pi(a_g | z) ]
```

`a` is detached, so the gradient flows through `z` only. Differentiating the
Gaussian log-density gives the score function

```
d/dz log pi(a | z) = (a - z) / sigma^2 = eps_projected / sigma
```

so the update moves `z` along `A * eps / sigma`: perturbations that scored above
the baseline pull the logits toward themselves. `gaussian_log_prob` is tested
against this identity directly.

### Hybrid objective

```
L = L_policy + ce_weight * CE(q, y)
CE(q, y) = - sum_k y_k * log(max(q_k, floor))
```

`ce_weight = 1.0` matches Laya's fine-tune; `0.0` is pure policy-gradient
base training (`PolicyConfig.base()`).

### Sigma schedule

```
sigma(step) = start + (end - start) * min(1, step / total_steps)
```

Base training runs `start = 1.0`, `end = 0.3`, `G = 8`, no cross-entropy.
Fine-tuning runs `start = 0.4`, `end = 0.1`, `G = 4`, cross-entropy at 1.0.

## Two implementation traps

Both were real failures during development, and both are covered by tests.

- Padded logits sit at the dtype minimum. Dividing them by `sigma` overflows to
  `-inf`, and `0 * inf` is `NaN`, which poisons the whole loss. The fix is to
  multiply by the mask before dividing by `sigma`.
- The sampled action must be built from detached logits. If the sample carries
  gradient, the reward path leaks into the policy term and the loss is no longer
  a score-function estimator.

## What the reward does not buy you

The reward is differentiable in the logits, so sampling is optional. Its real
contributions are the noise as a regularizer, the log floor and the spherical and
RPS terms. Honesty comes from the scoring rule being strictly proper, not from
the sampling. Run the direct baseline, then the policy version, and keep the
policy version only if it wins on held-out ECE or NLL.
