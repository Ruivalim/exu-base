# Reward and policy-gradient math

This is what the code in `exu/scoring.py`, `exu/policy.py` and `exu/bandit.py`
computes, and why.

Notation: `y` is the target distribution over `K` options (one value per option,
summing to one), `q` is the model's distribution, `p` is a valid-option mask, and
`z` are the option logits.

## The reward

### Log score

```
S_log(q, y) = sum_k y_k * log q_k
```

Computed from `log_softmax` of the logits (`masked_log_softmax`), in float32, never
as `log(softmax(z))`: softmax underflows to an exact zero on a confident miss.
There is no floor. A clamp at `1e-4` made the rule improper for any component
below about `2.7e-4`, where reporting zero outscored the truth, and it switched
off the gradient of every confidently wrong row. The gradient of this term in the
logits is `y - q`, bounded by one whatever the miss, so nothing needs a bound to
stay stable. Mass the target does not claim is skipped, so `0 * log 0` reads as 0.

The log score is the negative cross-entropy, so a direct cross-entropy baseline
and a log-score reward optimize the same quantity up to sign.

### Spherical score

```
S_sph(q, y) = <y, q> / ||q||
```

In `[0, 1]`. Honesty maximizes it, and at `q = y` it equals `||y||`: one for a
one-hot target, `0.707` for an exact `[0.5, 0.5]`. It rewards mass in the right place without
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
CE(q, y) = - sum_k y_k * log q_k        (from log_softmax, no floor)
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

## What the policy optimizes is a smoothed reward

The policy gradient ascends `J(z) = E[S(softmax(z + sigma * noise), y)]`, not `S` at
the unperturbed logits. `S` is strictly proper, `J` is not the same function, and
its optimum sits slightly on the confident side of the truth. For a binary target
of `0.30` the optimum centre is `0.277` at `sigma = 0.4`, `0.286` at `0.3` and
`0.298` at `0.1`, by quadrature of this reward. Hard labels do not escape it,
because the expectation is linear in `y`. What ships is the last sigma, which is
one reason to anneal it, and one reason to compare against the direct baseline on
held-out data instead of trusting propriety alone.

## What removing the floor costs in `batch` mode

`advantage_norm="batch"` divides by the standard deviation of the raw rewards of
the whole batch. With an unbounded log score, one confidently wrong row inflates
that deviation and the other rows of its batch learn slower for that step. In a
probe of 8 rows with one miss they lost 6% to 22% of their gradient under the
fine-tune preset and 23% to 82% under the base preset, at logit gaps of 12 and 50.
It is a slowdown and not an instability, and it ends on its own, because the
wrong row now has a gradient and its gap shrinks. `advantage_norm="group"` does
not have the effect: each question divides by its own deviation, and additive
logit noise keeps that spread near `sigma` whatever the gap.

## What the reward does not buy you

The reward is differentiable in the logits, so sampling is optional. Its real
contributions are the noise as a regularizer and the spherical and RPS terms. Honesty comes from the scoring rule being strictly proper, not from
the sampling. Run the direct baseline, then the policy version, and keep the
policy version only if it wins on held-out ECE or NLL.

## Feedback on one sampled answer

Everything above reads the whole target `y`. A sampled estimator has no direct
alternative only when `y` is never seen: the model draws an answer `k`, and a judge,
a test or a user says whether that one answer was right, `r` in `{0, 1}`. With
`y_k` the chance that `k` is judged right, `exu/bandit.py` turns that feedback into
an estimate of `y` and scores it with the same composite.

The obvious policy gradient, REINFORCE on `r`, is not an option. Its expected reward
is `sum_k q_k y_k`, linear in `q`, so it is maximized by all mass on the most likely
answer: accuracy, not honesty. `tests/test_bandit.py` trains free logits on a
`[0.6, 0.3, 0.1]` target both ways; REINFORCE ends at `0.99` on the first option,
the estimator below at the target.

Draw `k` from a behaviour policy that mixes the model with a uniform share,

```
pi = (1 - epsilon) * q.detach() + epsilon * uniform
```

and estimate the target from the one observation:

```
y_hat = r * e_k / pi_k                    # importance weighted
y_hat = b + (r - b_k) * e_k / pi_k        # doubly robust, b = q.detach() by default
```

Both have `E[y_hat] = y` for any fixed `b`, as long as every valid option has
`pi_k > 0`, which `epsilon > 0` guarantees and also bounds the weight by
`K / epsilon`. The log and spherical scores are linear in `y`; the ranked
probability score is linear in `y` up to a term the model cannot change. So the
gradient of the composite loss against `y_hat` is, in expectation, the gradient
against `y`, and the optimum is the same honest distribution. A test checks this
exactly, enumerating every answer and verdict.

`y_hat` is not a distribution: it exceeds one, and with a baseline it goes negative.
It is scored by `bandit_loss`, which keeps every component, and never by
`log_score`, which skips non-positive target mass and would drop half the estimate.

Nothing in the data format carries a verdict on a sampled answer yet, so this runs
only as a probe: `simulated_bandit_loss` draws the verdict from the target (one
annotator, picked at random, agreeing or not) and reads the target nowhere else.

