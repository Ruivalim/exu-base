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

## Confidence: two scales

- `confidence`, the maximum probability, is what the ECE was measured on. Put
  business thresholds on this.
- `entropy_confidence`, one minus normalized entropy, is smaller for the same
  distribution. A top probability of 0.85 over four options is about 0.58 here.

The runtime exposes both. Do not copy a threshold across the two.

## Baselines

Every report includes uniform guessing (flat `1/K` over the valid options), the
per-question prior (the mean target for that question) and the majority class.
The prior is usually the strongest trivial baseline. The uniform forecast is the
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
