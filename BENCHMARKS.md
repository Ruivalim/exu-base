# Benchmarks

Everything measured on Exu so far, with the numbers as they came out. Nothing here is
a published result of someone else, and nothing is tuned to look good: the project
is pre-alpha, the runs are small, and several of them say the toolkit's headline
method loses to its own baseline.

Last updated 2026-09-21.

## How to read this file

- **Proper scores first.** `nll` and `brier` are strictly proper and are what decides
  a comparison. `accuracy` and `ece` compare against `argmax(target)`. With soft
  targets an exactly honest forecast still has a non-zero ECE, so on the datasets
  below, where most targets are soft, ECE is a statement about the top choice only.
- **Every model number sits next to its trivial bars**: `uniform` (flat `1/K`) and
  `prior` (the per-question label distribution, fitted on the training split and
  never on the rows being scored). A model that does not beat `prior` on NLL has
  learned nothing beyond label frequency.
- **Paired differences.** Arms share seeds, so the same seed fixes the data order
  and the head initialisation. A difference is reported as mean ± sample standard
  deviation over seeds, with the count of seeds where the arm won. Three seeds make
  a weak test: read a tie as a tie.
- **One dataset, one encoder, three epochs** unless a section says otherwise. None of
  this is a verdict on RLCD in general. The setting its authors report gains in,
  fine-tuning on teacher-model soft targets, was not tested.

## Setup

| | |
| --- | --- |
| Training machine | 2x NVIDIA RTX 3060 12 GB, one job per GPU |
| Probe machine | 1x NVIDIA RTX 4060 8 GB, shared with a desktop session |
| Precision | bfloat16 autocast, log term of the reward always in float32 |
| Main encoder | `google-bert/bert-base-uncased` |
| Other encoders tried | `answerdotai/ModernBERT-base` |
| Encoder not used | the package default, `google-bert/bert-base-multilingual-cased`, ran out of memory on the shared 8 GB card (about 178M parameters plus AdamW state) |
| Optimiser | AdamW, encoder lr `2.5e-5`, head lr `1e-4`, weight decay `0.01`, grad clip `1.0`, constant lr, no warmup |
| Batch | 8 with `--grad-accum 2`, so 16 questions per update |
| RLCD defaults | `G = 4`, sigma `0.4` to `0.1`, cross-entropy weight `1.0`, `--advantage-norm batch` |
| Reward | log score from `log_softmax` with no floor, plus `0.75` spherical, minus RPS on ordinal questions |
| Calibration | `--calibrate` on a separate calibration split, temperatures per (type, option-count bucket) |

Code: the tree that follows `607e818` (log floor removed, baselines fitted on
reference labels). The `oldfloor` arms ran `607e818` itself from a worktree and were
scored with the current evaluator, so every arm is measured with the same ruler.

## Datasets

No dataset row is in this repository, and none may be: see `CONTRIBUTING.md`. Only
numbers derived from them are recorded here.

| Dataset | Hub id | License on the Hub | What was used |
| --- | --- | --- | --- |
| GoEmotions, raw | `google-research-datasets/go_emotions` | apache-2.0 | 58,009 comments, 28 emotions, one `choice` question. Target is the share of rater marks per label. 85.3% of targets are soft, 3.64 raters per comment, smallest positive mass 0.077. Splits by id hash: 46,370 / 2,869 / 2,886 / 5,884 (train / validation / calibration / test). |
| MultiNLI | `nyu-mll/multi_nli` | cc-by-3.0, cc-by-sa-3.0, mit, other (by genre) | Hard labels. 50,000-row deterministic subsample for training, and 16,000-row and 6,000-row subsamples for probes. Validation from `validation_matched`. |
| ChaosNLI, MNLI part | `metaeval/chaos-mnli-ambiguity` | **none declared** | 1,599 items labelled by 100 annotators each, smallest positive mass 0.01. Used as calibration (810) and test (789) for models trained on MultiNLI. Leak guards are on text: no ChaosNLI pair appears in training, and 142 training rows plus 4,011 validation rows that share a premise with it were dropped. |
| Measuring Hate Speech | `ucberkeley-dlab/measuring-hate-speech` | cc-by-4.0 | 17,352 comments with 3 or more annotators, times 4 ordinal facets (`insult`, `dehumanize`, `violence` on 5 levels, `hatespeech` on 3). 69,408 `score` questions with soft targets. Splits by comment hash, so a comment never crosses splits: 55,656 / 3,392 / 3,560 / 6,800. |

Two things the datasets taught before any model ran:

- ChaosNLI alone cannot train a model: 1,092 training rows, the encoder memorises
  them, and test NLL (1.077, temperature 3.66) never beats the prior (1.065).
- No human-annotated target reaches the region where the old log floor was improper
  (components below `2.7e-4`). Even 100 annotators bottom out at 0.01. Targets that
  small only come from a teacher model.

---

## 1. Direct baseline against RLCD, GoEmotions

Five arms, seeds 17, 23 and 42, 3 epochs, `--option-shuffle`. Calibrated test split,
5,884 rows. All 15 jobs finished, about 30 minutes each. Bars: `prior` 2.8267,
`uniform` 3.3322.

| Arm | NLL | Brier | Accuracy | ECE | Order stability | Temperature |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 1.8633 ± 0.0130 | 0.2197 ± 0.0055 | 0.5493 ± 0.0040 | 0.1139 ± 0.0091 | 0.9683 ± 0.0082 | 1.089 ± 0.028 |
| rlcd, `batch` | 1.9218 ± 0.0075 | 0.2257 ± 0.0051 | 0.5437 ± 0.0107 | 0.0920 ± 0.0032 | 0.9478 ± 0.0079 | 1.066 ± 0.055 |
| rlcd, `group` | 2.0779 ± 0.0640 | 0.2538 ± 0.0170 | 0.5074 ± 0.0239 | 0.0942 ± 0.0096 | 0.8935 ± 0.0348 | 1.243 ± 0.142 |
| baseline, old floor | 1.8626 ± 0.0104 | 0.2193 ± 0.0049 | 0.5521 ± 0.0025 | 0.1133 ± 0.0039 | 0.9654 ± 0.0044 | 1.075 ± 0.009 |
| rlcd `batch`, old floor | 1.9184 ± 0.0027 | 0.2258 ± 0.0042 | 0.5451 ± 0.0019 | 0.0971 ± 0.0008 | 0.9495 ± 0.0024 | 1.064 ± 0.037 |

Paired against `baseline`, same seed:

| Arm | NLL | Brier | Accuracy | ECE | Order stability |
| --- | --- | --- | --- | --- | --- |
| rlcd, `batch` | +0.0585 ± 0.0106 (0/3 better) | +0.0060 ± 0.0011 (0/3) | -0.0056 ± 0.0078 (1/3) | -0.0219 ± 0.0070 (3/3) | -0.0205 ± 0.0045 (0/3) |
| rlcd, `group` | +0.2146 ± 0.0512 (0/3) | +0.0341 ± 0.0117 (0/3) | -0.0419 ± 0.0217 (0/3) | -0.0197 ± 0.0178 (3/3) | -0.0748 ± 0.0271 (0/3) |

What it says:

- **The direct baseline beats RLCD on both proper scores, in all three seeds.** The
  NLL gap is 4.5 times the baseline's seed noise, about 9.6 standard errors
  (p ≈ 0.01 in a paired t-test with 2 degrees of freedom, which is a weak test). By
  the project's own rule, RLCD stays only if it clearly wins, so here the baseline
  stays.
- **`batch` remains the default advantage normalisation.** `group` loses everywhere
  except ECE and varies a lot across seeds.
- **RLCD has the lower ECE, 3 seeds out of 3.** That is real, and it is measured on
  targets that are 85% soft, so it does not overrule NLL and Brier.
- Every arm came out overconfident before calibration (temperature above 1), and
  the difference between RLCD and the baseline is inside the noise. The
  overconfidence that sigma smoothing predicts (section 8) did not show at this
  scale.

## 2. Old log floor against no floor

Same experiment. New code minus `607e818`, paired by seed:

| Metric | Baseline mode | RLCD `batch` |
| --- | --- | --- |
| NLL | +0.0007 ± 0.0034 | +0.0034 ± 0.0098 |
| Brier | +0.0005 ± 0.0006 | -0.0001 ± 0.0026 |
| Accuracy | -0.0028 ± 0.0031 | -0.0014 ± 0.0089 |
| ECE | +0.0006 ± 0.0054 | -0.0050 ± 0.0028 (3/3) |
| Order stability | +0.0029 ± 0.0038 | -0.0018 ± 0.0063 |

**Removing the floor changed nothing measurable here.** The change stands on
correctness (the reward is strictly proper, and a confidently wrong row keeps its
gradient) and is now measured to cost nothing. It also buys nothing on this
dataset, as expected: no target component is below 0.077, and a model at 55%
accuracy is almost never wrong with 99.99% confidence. How often a prediction fell
below `1e-4` during training was **not** counted, so that last sentence is an
inference.

## 3. Option shuffling, GoEmotions, one seed

Seed 17, same settings, with and without `--option-shuffle`.

| Run | NLL | Brier | Accuracy | ECE | Temperature | Order stability |
| --- | --- | --- | --- | --- | --- | --- |
| baseline, no shuffle | 1.8536 | 0.2148 | 0.5563 | 0.1162 | 1.114 | **0.471** |
| baseline, shuffle | 1.8562 | 0.2160 | 0.5525 | 0.1197 | 1.116 | **0.976** |
| rlcd, no shuffle | 1.9218 | 0.2231 | 0.5457 | 0.1208 | 0.929 | **0.698** |
| rlcd, shuffle | 1.9119 | 0.2212 | 0.5469 | 0.0994 | 1.062 | **0.954** |

**Without `--option-shuffle` the model learns position.** The baseline changes its
answer in 53% of option permutations. With shuffling the problem is gone and NLL
does not get worse. Train with `--option-shuffle`, always, for `choice` questions.

The same checkpoint also shows it at inference: asking one question with its six
options in four different orders moved the top probability from 0.62 to 0.44.

## 4. Measuring Hate Speech, ordinal questions with soft targets

Measuring Hate Speech is the Berkeley D-Lab corpus described under Datasets.

**In progress.** Same three arms and seeds as section 1, first use of the `score`
kind and of the ranked-probability term against a distribution. Bars on the test
split (6,800 rows): `prior` NLL 1.2838, RPS 0.1039; `uniform` NLL 1.4817, RPS 0.1477.

Finished so far (2026-09-21 15:35 UTC), calibrated test split:

| Arm | Seeds done | NLL | Brier | Accuracy | ECE | Temperature |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 2 of 3 | 1.0505 ± 0.0025 | 0.1938 ± 0.0005 | 0.6121 ± 0.0033 | 0.0790 ± 0.0022 | 1.517 ± 0.228 |
| rlcd, `batch` | 1 of 3 | 1.0502 | 0.1936 | 0.6172 | 0.0804 | 1.462 |
| rlcd, `group` | 0 of 3 | | | | | |

A quick check before launching it, 8,000 rows and one epoch: NLL 1.0735 against the
prior's 1.2919, RPS 0.0559 against 0.1108, per-facet NLL from 0.574 (`hatespeech`) to
1.412 (`dehumanize`). So this task is learnable with the default recipe.

Too early to read the arms against each other: one paired seed.

**Known problem surfaced here:** order stability came out at 0.23. The evaluator's
order-robustness pass permutes `score` questions too, and training never shuffles
them, because the order of an ordinal scale is its meaning. The number is
meaningless for ordinal questions and the pass should skip them.

Queued behind it: the `baseline` and `rlcd batch` arms again with
`answerdotai/ModernBERT-base`, on this dataset and on GoEmotions.

## 5. NLI: the marker readout does not learn, and what does

Training on MultiNLI (hard labels), `bert-base-uncased`, baseline mode, validation
on MultiNLI matched validation. `log 3 = 1.0986` is the uniform guess.

### The failure

| Run | Updates | Train loss | Validation NLL | Accuracy |
| --- | --- | --- | --- | --- |
| 6,000 rows, 2 epochs, shuffle | 750 | 0.670 | 1.0986 | 0.34 |
| 50,000 rows, 1 epoch, shuffle | 3,125 | 0.669 | 1.0986 | 0.32 |
| 16,000 rows, no shuffle | 1,000 | 0.670 | 1.0986 | 0.32 |
| 16,000 rows, `--encoder-lr 5e-5 --head-lr 5e-4` | 1,000 | 0.673 | 1.0986 | 0.32 |
| 16,000 rows, 5 epochs, no shuffle, cross-entropy only | 5,000 | 1.0998 | 1.0986 | 0.358 |
| 16,000 rows, no shuffle, **ModernBERT-base** | 1,000 | 0.670 | 1.0984 | 0.324 |

Train loss is the composite loss, whose uniform-guess level is 0.6656, except in the
cross-entropy-only row, where that level is 1.0986.

The model answers 1/3 for everything. In the trained checkpoint the gap between the
largest and the smallest logit of a row averages 0.002, against 7.7 in a GoEmotions
checkpoint. More steps, a higher learning rate, no shuffling and a better encoder
all leave it there. The same recipe learns GoEmotions inside the first epoch.

### Bisecting from a classifier that works

Same 16,000 rows, same encoder, same learning rates, batch 16, 1,000 updates, no
warmup:

| What reads the answer | Accuracy |
| --- | --- |
| BERT with a linear head on `[CLS]`, premise and hypothesis as two segments | 0.796 |
| The same, with the single JSON string Exu uses as its state | 0.755 |
| The same, fed Exu's own sequence (instruction, three `[MASK]` options, state) | 0.763 |
| The same, read after Exu's type embedding and its two decision layers | 0.721 |
| **Exu itself: one logit per `[MASK]` marker, shared scorer** | **0.32** |

So the data, the step count, the learning rate, the state format, the sequence
layout and the decision layers are all fine. **What fails is reading the answer at
the markers.** Also ruled out, each at about 0.33: a zero-initialised type
embedding, a single decision layer, and decision layers initialised from the
encoder's top layers.

Why GoEmotions works and NLI does not, as far as the evidence goes: in GoEmotions
an option's name has a lexical link to the text ("gratitude" and "thank you"), so
each marker can compute its own match and gets a gradient early. In NLI
"entailment" has no lexical link to anything. The answer is a global property of
the state, every marker sees the same thing, and the softmax cancels what is common
to all options.

### The readout that unlocked it

Added to each option's logit, leaving the existing scorer in place:

```
logit_k += < W e_k , LayerNorm(h_cls) > / sqrt(H)
```

`e_k` is the mean of the **input** embeddings of the option's own text tokens, which
is the same vector in every example, unlike the contextual state at the `[MASK]`.
It is **centred across the options of the question** and rescaled. Same 16,000
rows, one epoch, seed 17:

| Variant | Train loss | Validation NLL | Accuracy |
| --- | --- | --- | --- |
| Exu as it is | 0.670 | 1.0986 | 0.32 |
| marker state times `[CLS]` | 0.682 | 1.0986 | 0.324 |
| the same, marker states centred | 0.750 | 1.1399 | 0.354 |
| the same, option identified by its contextual text tokens, centred | 0.750 | 1.1406 | 0.354 |
| input embeddings of the option text, **not** centred | 0.671 | 1.0988 | 0.324 |
| input embeddings of the option text, **centred**, no shuffle | 0.434 | 0.6453 | **0.740** |
| input embeddings of the option text, **centred**, `--option-shuffle` | 0.434 | 0.6454 | **0.738** |

That matches the classifier that reads `[CLS]` through Exu's layers (0.721).
Centring is what does it: the three NLI options share almost every word ("the
hypothesis must be ... given the premise"), so without it the three readout vectors
are nearly equal and cancel in the softmax, which is the same cancellation that
blocks the marker scorer.

Status: **a probe, applied by monkeypatching, not in `src/`.** One seed, one epoch,
hard-label validation. Not yet measured: ChaosNLI's soft targets, whether it costs
anything on GoEmotions, more seeds. The two centred variants that read contextual
states gave the same training loss to four decimals, which may be a bug in that
probe rather than a result.

## 6. Encoders, small comparison

GoEmotions, 3,000 training rows, 1 epoch (375 updates), baseline mode, seed 17,
uncalibrated. 400 validation rows and 600 test rows, so the noise is large.

| Encoder | `--option-shuffle` | Validation NLL | Test NLL | Test accuracy |
| --- | --- | --- | --- | --- |
| `answerdotai/ModernBERT-base` | yes | 2.4261 | 2.3788 | 0.418 |
| `bert-base-uncased` after a short masked-LM run on Wikipedia (fewer than 60 updates) | yes | 2.5545 | 2.4456 | 0.443 |
| `bert-base-uncased` | no | 2.5484 | 2.4727 | 0.470 |

ModernBERT runs end to end in Exu and gives the lower NLL here. There is no
like-for-like `bert-base-uncased` run with shuffling at this size: the second row,
which is nearly the same model, is the closest. The continued-pretraining row is a
plumbing test of two stages (masked LM first, Exu second), not an attempt at domain
adaptation: held-out masked-LM loss went from 2.1046 to 1.8156 in that short run.

A better encoder does not fix section 5: ModernBERT collapses to the uniform guess
on NLI exactly like BERT.

## 7. Inference

`exu-decide --metrics`, GoEmotions baseline checkpoint (`bert-base-uncased`, 28
options), 200 test questions, batch 16:

| Device | Checkpoint load | Inference, total | First batch | Questions per second |
| --- | --- | --- | --- | --- |
| RTX 4060 | 1.5 s | 1.23 s | 364 ms | 162 |
| CPU | 1.1 s | 12.9 s | 1,119 ms | 15.5 |

One question on the GPU: 1.4 s to load plus 321 ms to answer, nearly all warm-up. The
CLI is for batches. A service should keep a `DecisionRuntime` resident.

## 8. Numerical probes

Measured on `607e818`, where the floor still existed, to decide whether to remove it.

**The floor made lying pay.** Composite score, float32, and the lie also wins in
float64:

| Target | Honest report | Reporting `[0, 1]` |
| --- | --- | --- |
| `[1e-5, 0.99999]` | 0.7498903871 | 0.7499004006 |
| `[2e-4, 0.9998]` | 0.7479466200 | 0.7480079532 |

The binary threshold is `floor * e * (1 - p/2)`: `2.717912e-04` with the log term
alone, `2.717635e-04` with the spherical term at 0.75.

**It also switched off the gradient of confident misses.** Hard label, logit gap
9.3: gradient `6.9e-5` with the floor, `1.0` without. In RLCD with the default
config, gap 12: `3e-8`.

**One confident miss in a batch of 8**, mean gradient on the wrong logit over 300
noise seeds, through the real policy loss. "Miss" is the wrong row, "others" the
mean of the other seven.

| Design, fine-tune preset | Gap 12: miss | others | Gap 50: miss | others |
| --- | --- | --- | --- | --- |
| Floor `1e-4` everywhere (old) | 0.0000 | 0.1057 | 0.0000 | 0.1057 |
| Floor `1e-12` everywhere | 0.1482 | 0.0990 | 0.0000 | 0.0868 |
| **No floor (adopted)** | 0.1482 | 0.0990 | 0.1304 | 0.0827 |
| No floor, batch deviation on centred rewards | 0.3254 | 0.2631 | 0.3254 | 0.2631 |
| No floor, `group` | 0.2958 | 0.2530 | 0.2958 | 0.2530 |

| Design, base preset | Gap 12: miss | others | Gap 50: miss | others |
| --- | --- | --- | --- | --- |
| Floor `1e-4` everywhere (old) | 0.0010 | 0.0282 | 0.0000 | 0.0282 |
| Floor `1e-12` everywhere | 0.0276 | 0.0217 | 0.0000 | 0.0094 |
| **No floor (adopted)** | 0.0276 | 0.0217 | 0.0066 | 0.0051 |
| No floor, batch deviation on centred rewards | 0.0979 | 0.0778 | 0.0979 | 0.0778 |
| No floor, `group` | 0.0788 | 0.0787 | 0.0788 | 0.0787 |

With the floor the wrong row was dead. Without it the row learns, and in `batch`
mode the other rows lose 6% to 22% (fine-tune) and 23% to 82% (base) of their
gradient for that step, because the batch deviation is taken over raw rewards. It
is a slowdown, not an instability, and section 2 shows it did not turn into a loss
of NLL in real training.

**RLCD optimises a smoothed reward.** The policy gradient ascends
`E[S(softmax(z + sigma * noise), y)]`, whose optimum sits on the confident side of
the truth. Binary target, by quadrature:

| True `p` | sigma 1.0 | 0.4 | 0.3 | 0.1 |
| --- | --- | --- | --- | --- |
| 0.30 | 0.2146 | 0.2771 | 0.2860 | 0.2983 |
| 0.10 | 0.0455 | 0.0868 | 0.0924 | 0.0991 |
| 0.02 | 0.0076 | 0.0171 | 0.0183 | 0.0198 |

Through the real loss, identical rows with target `[0.3, 0.7]`: the fine-tune preset
converges to 0.2785 (sigma 0.4) and 0.2983 (0.1), the base preset to 0.2147 (1.0)
and 0.2861 (0.3). A calculation at `K = 2`. It did not show up in section 1.

**The `prior` baseline used to read the labels it was scored on.** On the test split
of the repository's fixture (4 rows, 1 to 2 rows per question): fitted on reference
labels 0.7230 NLL and 0.25 accuracy, fitted in-sample 0.3466 and 0.75, `uniform`
0.7945 and 0.25. Leave-one-out gives accuracy 0.00 on a balanced question for
`n = 2, 4, 10, 100`. On GoEmotions, with 5,884 rows of one question, the two priors
are 2.8267 and 2.8256: the leak vanishes with large `n`, as expected.

## What has not been measured

- How often training puts a prediction below `1e-4`. No instrumentation exists.
- RLCD with teacher-model soft targets, which is where its authors report gains.
- Any other encoder at full scale, more epochs, another sigma schedule.
- The new readout of section 5 on anything but one seed of hard-label MultiNLI.
- The `noul` kind on real data. Only the fixture exercises it.
- Held-out task families. Every dataset above is a single question, or a few
  questions that all appear in training.
- The float16 path, which has no `GradScaler`, and any GPU older than Ampere.
- Whether the temperature map absorbs the bias of section 8's smoothing.

## Reproducing

The experiment driver and the dataset converters are working scripts outside the
repository. What they run is plain `exu-train` and `exu-evaluate`:

```bash
D=path/to/dataset.jsonl
uv run exu-train --mode baseline --encoder google-bert/bert-base-uncased \
  --train $D --train-split train --validation $D --validation-split validation \
  --test $D --test-split test --calibration $D --calibration-split calibration \
  --calibrate --option-shuffle --epochs 3 --batch-size 8 --grad-accum 2 \
  --seed 17 --output artifacts/run-baseline-s17

uv run exu-evaluate --checkpoint artifacts/run-baseline-s17 --data $D --split test \
  --order-permutations 2 --output artifacts/run-baseline-s17/report.json
```

The RLCD arms replace `--mode baseline` with `--mode rlcd --advantage-norm batch` (or
`group`). Seeds were 17, 23 and 42. Numbers in the tables come from `report.json`
(calibrated) unless a section says uncalibrated, in which case they come from
`training.json`.
