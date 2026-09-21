# Evaluation

Accuracy alone hides the failure that matters here. A model can be right most of
the time and still be untrustworthy if its confidence is uncalibrated. Every
report carries calibration and the trivial baselines beside the accuracy.

```bash
exu-evaluate \
  --checkpoint artifacts/my-model \
  --data data.jsonl --split test \
  --order-permutations 4 --latency \
  --output report.json
```

## Metrics

| Metric | Meaning |
| --- | --- |
| `nll` | Negative log-likelihood against the target distribution |
| `brier` | Sum of squared differences, per row |
| `accuracy` | Agreement between the predicted argmax and the target argmax |
| `soft_accuracy` | Inner product of prediction and target |
| `ece` | Expected calibration error over the maximum probability, 15 bins |
| `rps` | Ranked probability score, ordinal questions only |
| `ordinal_mae` | Absolute error of the expected level, ordinal questions only |

Reported overall, per question kind, and per task family.

`accuracy` and `ece` are hard-decision metrics: both compare the prediction with
`argmax(target)`. With a soft target a perfectly honest forecast still has a
non-zero ECE, for example `0.25` for an exact `[0.25, 0.75]`, because its
confidence of `0.75` is measured against a label that is "right" every time. For
soft targets read `nll` and `brier` first, and treat `ece` as a statement about the
top choice only.

## Confidence: two scales

- `confidence`, the maximum probability, is what the ECE was measured on. Put
  business thresholds on this.
- `entropy_confidence`, one minus normalized entropy, is smaller for the same
  distribution. A top probability of 0.85 over four options is about 0.58 here.

The runtime exposes both. Do not copy a threshold across the two.

## Baselines

Every report includes uniform guessing (flat `1/K` over the valid options), the
per-question prior and the majority class. The prior is usually the strongest
trivial baseline.

A baseline is a bar only if it could be deployed, so it must not read the labels
it is scored against. The prior and the majority class are fitted on **reference
labels**: the `train` split of `--data` by default, or `--reference FILE
--reference-split NAME` when the evaluation file holds no training rows. The rule
that keeps this honest is testable: mutate the evaluation targets and the
baseline's predictions must not move.

```
prior[q, k] = (count[q, k] + 1/K) / (n[q] + 1)
```

`count` is the reference target mass for option `k` of question `q`, soft targets
included, and `n` the number of reference rows for that question. It is symmetric
Dirichlet smoothing with a total pseudocount of one. A question the reference
never saw has `n = 0` and gets exactly `1/K`, so the fallback and the smoothing
are one rule, and the prior never predicts an exact zero. That matters because
`nll` clamps at float32 `tiny`: one outcome the reference never saw would cost 87
nats on its own. It is a reproducible default, not a claim that this amount of
smoothing is optimal. Count the original rows, not epochs of them and not
option-shuffled copies.

A question is identified by its kind, its instruction and its options. For
`choice` the options match in any order. A `score` question keeps its order,
because the order is the meaning.

What the report carries under `baselines`:

| Key | Meaning |
| --- | --- |
| `uniform` | Flat `1/K`. Needs no labels. |
| `prior` | The reference-fitted prior. `null` when no reference is usable. |
| `prior_seen`, `prior_unseen` | The same, on the rows whose question the reference has and has not. Present only when both kinds exist. |
| `majority` | Argmax of the same prior, `accuracy` only. A tie goes to the first option in canonical order, not to the first one shown. |
| `prior_in_sample` | The mean of the evaluation targets themselves. A diagnostic, never a bar. |
| `reference` | Where the prior came from: path, split, rows, the smoothing, `seen_rows`, `unseen_rows`, and `unavailable` with the reason when there is none. |

`majority` reports no `nll`: a one-hot forecast puts an exact zero on every outcome
it misses, so its NLL is the error rate times 87 and says nothing.

`prior_in_sample` is the best constant-per-question forecast for log loss and
Brier on that sample, in hindsight, so beating it on those two means the model
used the state. It reads the labels it is scored against: a question that occurs
once gets its own target back, and on small splits it is mostly an echo. The
evaluator never falls back to it. If the reference rows are among the rows being
scored, or the reference split does not exist, `prior` and `majority` come back
`null` and `reference.unavailable` says why.

The uniform forecast is the
floor: completely uninformative, and it pins the NLL at exactly `log K` when the
question's option count matches (verified in the tests), so a model that
cannot beat it has learned nothing. Its `ECE` is not zero: the maximum
probability is `1/K`, and a 15-bin ECE reads that as a small miscalibration. Do
not pool questions with different option counts when reading the uniform
baseline: use its per-kind and per-family breakdowns instead.

This is not decoration. In Laya the base multilingual
checkpoints scored below the majority-class baseline on unseen task families, and
nobody would have noticed without the comparison.

## Order robustness

`--order-permutations N` permutes each question's options `N` times and measures
how often the answer moves to a different option, mapping back through the
permutation. `stability` close to 1 means the model reads the criteria. Laya
changed its answer in 15% to 23% of cases, the signature of a
model that learned position.

Also reported: `mean_winner_probability`, the mass the model keeps on the
unpermuted winner under permutation.

## Selective coverage

Accuracy when answering only the most confident fraction of questions. If this
curve does not rise as coverage falls, confidence cannot be used to route work to
a human. The report includes the top 50% and top 80%.

## Latency

`--latency` measures p50 and p95 for 1, 5, 10 and 50 questions in one call. Latency
grows with the number of questions because the state is re-encoded once per
question, even in a batch. Measure on the hardware you will deploy on, not the
one you trained on.

## Before you claim a number

- Calibration claims must come from held-out data, never from training data.
- Compare against the baselines above, not against zero.
- Use identical questions byte for byte and a fixed seed across models.
- If a fitted temperature sits on a bound (0.1 or 10), treat it as an alarm.
- Report the split, the seed and the command that produced the number.

If an RLCD run does not beat the direct baseline on held-out ECE or NLL, the direct
baseline is the one to ship.
