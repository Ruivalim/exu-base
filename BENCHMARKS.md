# Benchmarks

Everything measured on Exu so far, with the numbers as they came out. Nothing here is
a published result of someone else, and nothing is tuned to look good: the project
is pre-alpha, the runs are small, and several of them say the toolkit's headline
method loses to its own baseline.

Last updated 2026-09-22.

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
| Rented machine, section 9 | 1x NVIDIA RTX 5090 32 GB, up to four jobs sharing the card |
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
| BoolQ | `google/boolq` | cc-by-sa-3.0 | 12,697 yes-or-no questions about a passage, hard labels, one `noul` question with the BoolQ question and the passage in the state. The first real data for the `noul` kind. 62% of answers are yes. Splits by text hash: 8,473 / 954 from BoolQ's train, and its validation split halved into calibration (1,611) and test (1,659). |
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

Same three arms and seeds as section 1, `bert-base-uncased`, 3 epochs. First use of
the `score` kind and of the ranked-probability term against a distribution.
Calibrated test split, 6,800 rows. All 9 jobs finished, about 35 minutes each. Bars:
`prior` NLL 1.2838 and RPS 0.1039, `uniform` NLL 1.4817 and RPS 0.1477.

| Arm | NLL | Brier | Accuracy | ECE | RPS | Ordinal MAE | Temperature |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 1.0533 ± 0.0052 | 0.1952 ± 0.0024 | 0.6094 ± 0.0052 | 0.0789 ± 0.0015 | 0.0491 ± 0.0013 | 0.4617 ± 0.0077 | 1.494 ± 0.166 |
| rlcd, `batch` | 1.0527 ± 0.0030 | 0.1946 ± 0.0009 | 0.6143 ± 0.0029 | 0.0816 ± 0.0077 | 0.0490 ± 0.0003 | 0.4633 ± 0.0022 | 1.438 ± 0.033 |
| rlcd, `group` | 1.1440 ± 0.0175 | 0.2207 ± 0.0075 | 0.6083 ± 0.0028 | 0.1433 ± 0.0182 | 0.0602 ± 0.0035 | 0.5522 ± 0.0307 | 3.013 ± 0.089 |

Paired against `baseline`, same seed:

| Arm | NLL | Brier | Accuracy | ECE | RPS |
| --- | --- | --- | --- | --- | --- |
| rlcd, `batch` | -0.0005 ± 0.0033 (2/3 better) | -0.0006 ± 0.0020 (2/3) | +0.0049 ± 0.0045 (2/3) | +0.0027 ± 0.0067 (1/3) | -0.0000 ± 0.0010 (2/3) |
| rlcd, `group` | +0.0907 ± 0.0130 (0/3) | +0.0256 ± 0.0061 (0/3) | -0.0010 ± 0.0078 (1/3) | +0.0644 ± 0.0197 (0/3) | +0.0111 ± 0.0024 (0/3) |

What it says:

- **Here RLCD and the direct baseline tie.** Every paired difference of the `batch`
  arm is inside its own spread. On GoEmotions the baseline won by 0.0585 nats, so
  the two datasets disagree on the size of the gap and agree that RLCD does not win.
- **`group` loses again, and by more**: a temperature of 3.0 says it came out badly
  overconfident. `batch` stays the default on two datasets out of two.
- The model beats the prior by 0.23 nats of NLL and halves its RPS, so the ordinal
  path (the `score` kind, the ranked-probability term, no shuffling of levels) works
  on real data. Per-facet NLL in one baseline run: `hatespeech` 0.590, `violence`
  1.002, `insult` 1.223, `dehumanize` 1.394.
- All arms are strongly overconfident before calibration (temperature 1.4 to 3.0),
  more than on GoEmotions.

A quick check before launching it, 8,000 rows and one epoch, had given NLL 1.0735
against the prior's 1.2919, which is how the task was known to be learnable.

**A bug surfaced here, since fixed:** order stability came out at 0.23, which is
chance for five levels. The order-robustness pass permuted `score` questions too,
and training never shuffles them, because the order of an ordinal scale is its
meaning. The pass now leaves ordinal questions out. The reports of this experiment
were written before the fix, so their stability column is not shown and should be
ignored.

Running now: the `baseline` and `rlcd batch` arms again with
`answerdotai/ModernBERT-base`, on this dataset and then on GoEmotions.

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

### The cause: a symmetric saddle, and balanced labels

Three explanations were tried and thrown out before the one that held. `marker`
scorer, `bert-base-uncased`, seed 17:

| Hypothesis | Test | Result | Verdict |
| --- | --- | --- | --- |
| The three NLI option descriptions are nearly identical | NLI with bare option names | NLL 1.0986, accuracy 0.354 | rejected |
| "entailment" means nothing to the encoder | NLI with options `yes` / `maybe` / `no` | NLL 1.0986, accuracy 0.330 | rejected |
| The answer needs a lexical cue from the option | BoolQ, `No` / `Yes` | accuracy 0.703, NLL 0.636 | rejected: it learns |
| The same | BoolQ with options `alpha` / `beta` | accuracy 0.715, NLL 0.590 | rejected: it still learns |
| **The training labels are balanced** | NLI, training split skewed to 60 / 25 / 15 | **accuracy 0.689**, NLL 0.722 | **confirmed** |
| **The same, from the other side** | BoolQ, training split balanced to 50 / 50 | **NLL 0.6931 = log 2**, train loss flat at 0.167 | **confirmed** |

Validation and test splits are the same in every row. Only the training labels of
the last two rows were resampled.

The mechanism: the scorer is shared across markers, so at the start every option
gets nearly the same logit, and the gradient that reaches anything the options have
in common is `sum_k (q_k - y_k) * c = 0`. Something has to make the options
distinguishable first. A skewed label prior does, because scoring the frequent
option higher pays at once, and that forces the model to tell the markers apart by
their text. Lexical overlap between an option and the state does too, which is
GoEmotions. With balanced labels and no overlap nothing does, and the model sits at
the saddle for as long as it is trained. MultiNLI is balanced to within 0.4%.
GoEmotions and Measuring Hate Speech are heavily skewed, which is why they never
showed this.

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
Centring is what does it: whatever the options share produces the same logit
everywhere and cancels in the softmax, which is the same cancellation that blocks
the marker scorer. Centred and rescaled identities are distinguishable from the
first step, so the saddle is gone by construction.

The table above comes from probes applied by monkeypatching. The two centred
variants that read contextual states gave the same training loss to four decimals,
which may be a bug in that probe rather than a result.

### The same thing as a real option: `--scorer marker-cls`

Implemented in `src/exu/model.py`, off by default, stored in the checkpoint. Same
16,000 MultiNLI rows, one epoch, seed 17, `--option-shuffle`, then calibrated and
tested on ChaosNLI (789 items, 100 annotators each):

| | NLL | Brier | Accuracy | ECE |
| --- | --- | --- | --- | --- |
| MultiNLI validation, hard labels | 0.6403 | | 0.752 | 0.035 |
| ChaosNLI test, uncalibrated | 1.3316 | 0.3000 | 0.494 | 0.194 |
| ChaosNLI test, temperature 4.05 | 1.0728 | 0.1901 | 0.494 | 0.065 |
| ChaosNLI bars: `prior` / `uniform` | 1.0993 / 1.0986 | | | |

Order stability 0.996. The hard-label result reproduces the probe. The ChaosNLI rows
say something else, and it is not flattering: a model trained on single labels is
badly overconfident on items where 100 people disagree (NLL worse than the uniform
guess before calibration), and a temperature of 4 only brings it just under the
bars. The floor of NLL on that split, the mean entropy of the targets, is 0.7457.
Learning the task and being calibrated on its ambiguous cases are different
problems, and only the first is solved here.

It does not cost anything where the marker scorer already worked. GoEmotions, 3,000
rows, one epoch, `--option-shuffle`, seed 17, uncalibrated:

| Scorer | Validation NLL | Test NLL | Test accuracy |
| --- | --- | --- | --- |
| `marker` | 2.5154 | 2.4312 | 0.460 |
| `marker-cls` | 2.3911 | 2.3786 | 0.423 |

One seed and 600 test rows: read it as "no harm", not as a gain.

More runs with the real flag, all `bert-base-uncased`, `--option-shuffle`,
uncalibrated. NLI rows are one epoch on 16,000 rows, scored on MultiNLI validation.
BoolQ rows are three epochs, scored on its test split (`prior` NLL 0.663, accuracy
0.622). Measuring Hate Speech rows are one epoch on 8,000 rows.

| Task | Scorer | Seed | NLL | Accuracy |
| --- | --- | --- | --- | --- |
| NLI, balanced | `marker-cls` | 17 / 23 / 42 | 0.6403 / 0.6231 / 0.6282 | 0.752 / 0.739 / 0.753 |
| NLI, balanced, RLCD mode | `marker-cls` | 17 | 0.6709 | 0.721 |
| NLI, balanced, bare option names | `marker-cls` | 17 | 0.6319 | 0.747 |
| BoolQ, natural 62% yes | `marker` | 17 / 23 / 42 | 0.6356 / 0.6282 / 0.7402 | 0.703 / 0.710 / 0.694 |
| BoolQ, natural 62% yes | `marker-cls` | 17 / 23 / 42 | 0.7770 / 0.5911 / 0.6756 | 0.610 / 0.687 / 0.689 |
| BoolQ, options `alpha` / `beta` | `marker-cls` | 17 | 0.6043 | 0.679 |
| BoolQ, balanced training | `marker` | 17 | 0.6931 | 0.598 |
| BoolQ, balanced training | `marker-cls` | 17 | 0.6235 | 0.656 |
| Measuring Hate Speech (RPS 0.0556 and 0.0538) | `marker` / `marker-cls` | 17 | 1.0761 / 1.0745 | 0.592 / 0.598 |

What it adds up to:

- `marker-cls` holds across seeds on NLI, works in RLCD mode, and does not need the
  option descriptions.
- Where `marker` does learn, it is as good or slightly better and steadier: on BoolQ
  its accuracy is 0.694 to 0.710 across seeds, against 0.610 to 0.689 for
  `marker-cls`, with NLL mixed. So `marker-cls` is a remedy for one condition, not a
  new default on this evidence.
- The `score` kind is unaffected.
- A practical test for the condition: if the first epoch ends with validation NLL at
  exactly `log K`, the model is on the saddle.

Inference was exercised on a `marker-cls` checkpoint through `exu-decide`: three
handwritten premise and hypothesis pairs came out as entailment 0.92, contradiction
0.93 and neutral 0.72, and reordering the options changed nothing.

All of that was later measured at full size with three seeds, and the picture
sharpened: see section 9. In short, `marker-cls` costs NLL wherever `marker` learns,
and is the only one of the two that learns where `marker` sits on the saddle.

## 6. Encoders, small comparison

GoEmotions, 3,000 training rows, 1 epoch (375 updates), baseline mode, seed 17,
uncalibrated. 400 validation rows and 600 test rows, so the noise is large.

| Encoder | `--option-shuffle` | Validation NLL | Test NLL | Test accuracy |
| --- | --- | --- | --- | --- |
| `answerdotai/ModernBERT-base` | yes | 2.4261 | 2.3788 | 0.418 |
| `bert-base-uncased` after a short masked-LM run on Wikipedia (fewer than 60 updates) | yes | 2.5545 | 2.4456 | 0.443 |
| `bert-base-uncased` | yes | 2.5154 | 2.4312 | 0.460 |
| `bert-base-uncased` | no | 2.5484 | 2.4727 | 0.470 |

ModernBERT runs end to end in Exu and gives the lower NLL here, 2.43 against 2.52
on validation with the same settings. The continued-pretraining row is a
plumbing test of two stages (masked LM first, Exu second), not an attempt at domain
adaptation: held-out masked-LM loss went from 2.1046 to 1.8156 in that short run.

A better encoder does not fix section 5: ModernBERT collapses to the uniform guess
on NLI exactly like BERT. Section 9 repeats this with three seeds, and compares the
two encoders at full size.

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

## 9. Full-size runs on a rented GPU, 2026-09-21

Preliminary, in the sense that nothing here has been rerun. Every block is three
seeds (17, 23, 42) under the protocol of "Reproducing": `--option-shuffle`, batch 8
with `--grad-accum 2`, `--calibrate` on the calibration split, test numbers from
`report.json`. Three epochs, except the NLI blocks, which are one epoch on the
16,000-row MultiNLI subsample and are tested on ChaosNLI. 45 jobs, none failed.

Code: the current tree, with `--scorer`, the order-robustness skip for `score`
questions and the confident-miss counter. The Measuring Hate Speech block with
ModernBERT is the exception: it ran on the 3060s with the tree of sections 1 to 4,
which has the same training algorithm and no counter.

Hardware note. One job at batch 8 keeps an RTX 5090 at 58% and one CPU core at 100%:
the Python loop is the limit, not the card. Four jobs sharing the card ran at 13.5
batches per second each, 54 together against 32 alone, so 1.7 times the throughput.
A GoEmotions job took 9.8 minutes alone (about 30 on a 3060) and 24.5 minutes with
three neighbours. The `marker` baseline reproduces across machines: 1.8676 here
against 1.8633 on the 3060s (section 1).

### `marker` against `marker-cls` where `marker` already learns

`bert-base-uncased`, baseline mode.

| Dataset | Scorer | NLL | Brier | Accuracy | ECE | Order stability |
| --- | --- | --- | --- | --- | --- | --- |
| GoEmotions | `marker` | 1.8676 ± 0.0076 | 0.2195 ± 0.0042 | 0.5514 ± 0.0022 | 0.1095 ± 0.0125 | 0.9645 ± 0.0052 |
| GoEmotions | `marker-cls` | 1.9238 ± 0.0268 | 0.2345 ± 0.0098 | 0.5303 ± 0.0129 | 0.0848 ± 0.0090 | 0.9952 ± 0.0008 |
| Measuring Hate Speech | `marker` | 1.0515 ± 0.0043 | 0.1942 ± 0.0019 | 0.6162 ± 0.0071 | 0.0820 ± 0.0031 | not applicable |
| Measuring Hate Speech | `marker-cls` | 1.0726 ± 0.0124 | 0.2044 ± 0.0053 | 0.5867 ± 0.0150 | 0.0855 ± 0.0084 | not applicable |

Bars: GoEmotions `prior` 2.8267, Measuring Hate Speech `prior` 1.2838. On Measuring
Hate Speech the RPS is 0.0486 ± 0.0010 for `marker` and 0.0531 ± 0.0027 for
`marker-cls`, against 0.1039 for `prior`.

Paired, `marker-cls` minus `marker`, same seed:

| Dataset | NLL | Brier | Accuracy | Order stability |
| --- | --- | --- | --- | --- |
| GoEmotions | +0.0562 ± 0.0192 (0/3 better) | +0.0150 ± 0.0056 (0/3) | -0.0212 ± 0.0120 (0/3) | +0.0306 ± 0.0044 (3/3) |
| Measuring Hate Speech | +0.0211 ± 0.0145 (0/3 better) | +0.0102 ± 0.0059 (0/3) | -0.0295 ± 0.0121 (0/3) | not applicable |

`marker-cls` loses on both proper scores and on accuracy, on every seed, on both
datasets. On GoEmotions the difference is close to three times its own spread. What
it buys there is order stability, 0.995 against 0.965, and a lower ECE, which does
not outweigh the NLL. The small-sample row of section 5 that read as "no harm" was
noise: at full size there is a cost. The default stays `marker`.

On Measuring Hate Speech the report now says `skipped_ordinal: 6800` and gives no
stability number, because every question is a scale and the order of a scale is its
meaning. The earlier tree permuted those options and reported a stability near 0.25,
which meant nothing.

### The saddle with three seeds

Baseline mode. The NLI validation column is MultiNLI with hard labels, the test
column is ChaosNLI with 100 annotators per item, calibrated.

| Task | Encoder | Scorer | Validation accuracy | Test NLL | Test accuracy | Order stability |
| --- | --- | --- | --- | --- | --- | --- |
| NLI | `bert-base-uncased` | `marker` | 0.3443 ± 0.0159 | 1.0985 ± 0.0001 | 0.4212 ± 0.0433 | 0.3555 |
| NLI | `bert-base-uncased` | `marker-cls` | 0.7477 ± 0.0040 | 1.0528 ± 0.0118 | 0.5049 ± 0.0063 | 0.9932 |
| NLI | `ModernBERT-base` | `marker` | 0.3447 ± 0.0179 | 1.0986 ± 0.0003 | 0.3583 ± 0.1566 | 0.3937 |
| NLI | `ModernBERT-base` | `marker-cls` | 0.8720 ± 0.0052 | 1.0253 ± 0.0092 | 0.5526 ± 0.0241 | 0.9816 |
| BoolQ, balanced training | `bert-base-uncased` | `marker` | 0.6307 ± 0.0197 | 0.6930 ± 0.0002 | 0.6052 ± 0.0277 | 0.7119 |
| BoolQ, balanced training | `bert-base-uncased` | `marker-cls` | 0.5971 ± 0.0938 | 0.6394 ± 0.0456 | 0.6050 ± 0.0933 | 0.9998 |

ChaosNLI bars: `uniform` 1.0986, `prior` 1.0993. Balanced BoolQ bars: `prior`
0.6931.

- `marker` ends at exactly `log 3` on NLI and `log 2` on balanced BoolQ, on every
  seed, with either encoder. Its accuracy on balanced BoolQ is what a constant
  answer gets on a test split that is 62% yes.
- `marker-cls` leaves the saddle on all nine runs. Paired NLL against `marker`:
  -0.0458 ± 0.0118 on NLI with BERT, -0.0732 ± 0.0095 with ModernBERT, -0.0537 ±
  0.0455 on balanced BoolQ, 3 of 3 seeds each.
- Leaving the saddle is not the same as learning the task. On balanced BoolQ the
  calibrated NLL improves but the accuracy stays at the constant answer, the
  uncalibrated validation NLL is worse than `log 2` (0.7724 ± 0.1234), and the fitted
  temperature is 4.9 ± 4.4. On NLI it does learn: 0.75 validation accuracy with BERT
  and 0.87 with ModernBERT.
- The ChaosNLI test still asks for temperatures of 2.8 to 3.9. With ModernBERT the
  calibrated model now beats `prior` with room (1.0253 against 1.0993), which BERT
  only barely did (section 5).

### RLCD against the baseline, again

| Dataset | Encoder | Baseline NLL | RLCD NLL | Paired, RLCD minus baseline |
| --- | --- | --- | --- | --- |
| GoEmotions | `ModernBERT-base` | 1.8549 ± 0.0119 | 1.8980 ± 0.0252 | +0.0432 ± 0.0345 (0/3 better) |
| Measuring Hate Speech | `ModernBERT-base` | 1.0516 ± 0.0041 | 1.0469 ± 0.0079 | -0.0047 ± 0.0104 (2/3 better) |
| NLI, `marker-cls` | `ModernBERT-base` | 1.0253 ± 0.0092 | 1.0487 ± 0.0501 | +0.0234 ± 0.0417 (1/3 better) |

`--advantage-norm batch` everywhere. On GoEmotions with ModernBERT Brier and accuracy
tie (+0.0004 ± 0.0069 and +0.0047 ± 0.0133). On Measuring Hate Speech RLCD is ahead on
accuracy on all three seeds (+0.0098 ± 0.0068) and ties on RPS (0.0479 against
0.0485). On NLI one RLCD seed fell back onto the saddle even with `marker-cls`
(validation accuracy 0.324, test NLL 1.1045), and the other two matched the
baseline (0.878 and 0.864 validation accuracy).

Put next to sections 1 and 4, that is five comparisons: RLCD loses on GoEmotions
with both encoders, ties on Measuring Hate Speech with both, and is unstable on NLI.
It has not won one yet. The setting its authors report gains in, teacher-model soft
targets, is still untested.

### Encoders at full size

Baseline mode, `marker` except on NLI.

| Dataset | `bert-base-uncased` | `ModernBERT-base` |
| --- | --- | --- |
| GoEmotions, NLL | 1.8676 ± 0.0076 | 1.8549 ± 0.0119 |
| Measuring Hate Speech, NLL | 1.0515 ± 0.0043 | 1.0516 ± 0.0041 |
| NLI with `marker-cls`, validation accuracy | 0.7477 ± 0.0040 | 0.8720 ± 0.0052 |
| NLI with `marker-cls`, ChaosNLI NLL | 1.0528 ± 0.0118 | 1.0253 ± 0.0092 |

ModernBERT is worth 12 points of accuracy on NLI and nothing measurable on the other
two. A job costs about 1.5 times as long. These are not paired runs of one
experiment: the BERT and ModernBERT blocks share seeds and protocol but ran
separately.

### How often the model confidently misses

`confident_miss_rate` in `training.json` is the share of claimed target components
the model gave less than `1e-4` to, which is exactly where the old log floor acted.
Per epoch, over every run above that has the counter:

| Mode | Dataset | Lowest and highest rate over epochs and seeds |
| --- | --- | --- |
| baseline | GoEmotions, both scorers and encoders | 0 to 8.3e-05 |
| baseline | Measuring Hate Speech | 0 with `marker`, up to 1.8e-05 with `marker-cls` |
| baseline | NLI | 0 |
| baseline | balanced BoolQ | 0 with `marker`, up to 1.7e-03 with `marker-cls` |
| RLCD | GoEmotions, ModernBERT | 2.3e-04 to 1.9e-03 |
| RLCD | NLI, `marker-cls` | 0 |

In RLCD mode the sampled candidates miss at the same rate as the centre (2.7e-04 to
1.9e-03). RLCD makes confident misses ten to a hundred times more often than the
baseline, which fits the overconfident optimum of the sigma-smoothed objective
(section 8). Even so the old floor touched at most 0.2% of components, which is why
removing it changed nothing measurable in section 2. That was an inference, and is
now a count.

### A per-option readout sits on the same plateau (probe, one seed)

A probe outside the package reads one sequence per option, `[CLS] type instruction
[SEP] option [SEP] state [SEP]`, scores the `[CLS]` vector with one shared
`LayerNorm` and `Linear(H, 1)`, and takes the softmax across options. Same reward,
learning rates, batch and data as above: `bert-base-uncased`, 16,000 NLI rows, seed
17, no `--option-shuffle` (without a set head the answer cannot depend on option
order). Validation is MultiNLI with hard labels, uncalibrated.

| Run | Validation NLL | Validation accuracy | Mean logit spread |
| --- | --- | --- | --- |
| Per-option readout, 1 epoch | 1.0986 | 0.329 | 0.001 |
| Per-option readout, 3 epochs | 0.6903 | 0.761 | 4.9 |
| Per-option readout, 1 epoch, skewed labels (60/25/15) | 0.8117 | 0.670 | 2.8 |
| `marker`, 1 epoch (above) | 1.0986 | 0.344 | 0.002 |
| `marker-cls`, 1 epoch (above) | 0.6233 | 0.748 | |

The one-epoch run stayed at `log 3` with a flat training loss for the whole epoch,
like the marker readout. The three-epoch run, same seed and same data order, left
the plateau during its first epoch (mean training loss 0.625 against 0.672) and
reached 0.761. The only difference between the two first epochs is GPU
non-determinism, so when this readout leaves the plateau is not stable. With skewed
labels it learns within one epoch, as `marker` did in section 5 (0.689).

So the plateau does not come from reading a `[MASK]` position: it comes from one
scorer shared by options that look nearly the same at the start, under balanced
labels, and a per-option readout shares that. It can leave on its own, late and
unreliably. What `marker-cls` adds is a readout vector per option, built from the
option's own text, which breaks the symmetry from the first updates on every seed
tried. Whether `marker` itself would leave the plateau on NLI with three epochs was
not run. On balanced BoolQ, three epochs, it did not on any of three seeds.

A decoder backbone with the same per-option readout is section 10.

## 10. A decoder backbone, probe results, 2026-09-21

Same probe as the end of section 9, with `Qwen/Qwen3-0.6B` as the backbone: one
sequence per option, `State: <state> / type / Question: <instruction> / Candidate:
<option> / Decision: <EOS>`, the hidden state of the last token scored by one shared
`LayerNorm` and `Linear(1024, 1)`, softmax across options. The language-model head is
not loaded. Full fine-tune, no LoRA, float32 weights with bfloat16 autocast, AdamW
with backbone lr `1e-5` and head lr `1e-4`, weight decay `0.01`, grad clip `1.0`,
batch 8 with `--grad-accum 2`. Reward, calibration and metrics are the package's.
Nothing here is in the package: `exu-train`, `exu-evaluate` and `exu-decide` do not
know this path. Preliminary in every respect.

### NLI, three seeds

16,000 MultiNLI rows, one epoch, tested on ChaosNLI, calibrated. The encoder rows
are the section 9 numbers.

| Backbone | Scorer or readout | Validation accuracy | ChaosNLI NLL | ChaosNLI Brier | Temperature | Minutes per run |
| --- | --- | --- | --- | --- | --- | --- |
| `bert-base-uncased` | `marker` | 0.3443 ± 0.0159 | 1.0985 ± 0.0001 | 0.2087 | 1.0 | 3 (5090) |
| `bert-base-uncased` | `marker-cls` | 0.7477 ± 0.0040 | 1.0528 ± 0.0118 | 0.1789 | 3.4 | 3 (5090) |
| `ModernBERT-base` | `marker-cls` | 0.8720 ± 0.0052 | 1.0253 ± 0.0092 | 0.1605 | 3.3 | 7.5 (5090) |
| `Qwen3-0.6B` | per-option, last token | 0.8643 ± 0.0102 | 0.9979 ± 0.0139 | 0.1474 ± 0.0092 | 2.7 ± 0.8 | 36 (3060) |

- The decoder learns this balanced task from the first updates, with no remedy: the
  training loss was below the uniform guess within 200 batches on every seed. That
  is not immunity to the saddle of section 5: on balanced BoolQ, below, two of three
  seeds sit on it for the whole epoch.
- On ChaosNLI it is the first model under 1.0, with room over `prior` (1.0993), and
  the best Brier. Uncalibrated it is not better than the encoders (1.24 ± 0.15
  against 1.23 to 1.40): it is as overconfident on ambiguous items, and one seed
  (17, temperature 1.8) made that look better than it is.
- Validation accuracy ties ModernBERT with `marker-cls`.
- Cost: about 12 times a BERT run per epoch, for 3 options per question.

### GoEmotions, 16,000 training rows, one seed

Seed 17. The training sample is a third of the split; validation, calibration and
test are the full splits. Test numbers calibrated. `sft` is `--mode baseline`, the
package's direct loss, for two epochs. The second-stage arms start from its saved
weights and add one epoch each; the second stage checks validation before its first
update and reproduced the first stage's number exactly (1.8742).

| Arm | Epochs | Test NLL | Brier | Accuracy | ECE | Confident-miss rate, last epoch |
| --- | --- | --- | --- | --- | --- | --- |
| `sft` (baseline) | 2 | 1.8653 | 0.2191 | 0.5505 | 0.127 | 1.0e-04 |
| `sft` then baseline | 2 + 1 | 1.9276 | 0.2328 | 0.5321 | 0.128 | 7e-04 |
| `sft` then RLCD | 2 + 1 | 1.9864 | 0.2448 | 0.5119 | 0.133 | 1.0e-03 |
| `bert-base-uncased` `marker`, full split, 3 epochs (section 9) | 3 | 1.8676 ± 0.0076 | 0.2195 | 0.5514 | 0.110 | |

- With a third of the data and one epoch less, the decoder ties the BERT baseline
  trained on the whole split. It took 41 minutes per epoch on an RTX 5090 with
  gradient checkpointing, against 25 for the BERT job sharing the card with three
  others.
- A third epoch hurts either way: training loss kept falling (1.06) while validation
  rose. 16,000 rows are not enough for a 0.6B model to keep learning past two
  epochs, so this second stage started with no room to improve.
- In that situation RLCD lost 0.059 more than the direct loss did, from the same
  weights and the same number of updates, and its confident-miss rate was ten times
  the first stage's. Sixth comparison, still no win for RLCD. Whether it helps when
  the model still has room is the run on the full split, below.

### GoEmotions, full training split

The full 46,370 training rows, `--max-length 128` (cuts 5 of 58,007 questions;
without it one batch of outliers takes the whole 32 GB card). `sft` is one epoch of
the direct loss, and has three seeds; the other arms are seed 17 only. The two second stages start from its weights, checked
against its validation number before the first update (1.8476, exact), and add one
epoch each. The from-scratch RLCD arm has two epochs, so every final arm has two.
About 61 minutes per epoch on an RTX 5090 (97 on a slower host).

| Arm | Epochs | Test NLL | Brier | Accuracy | ECE | Confident-miss rate, last epoch |
| --- | --- | --- | --- | --- | --- | --- |
| `sft` (baseline), seed 17 | 1 | 1.8435 | 0.2142 | 0.5648 | 0.135 | 3.8e-05 |
| `sft` (baseline), 3 seeds | 1 | **1.8417 ± 0.0019** | **0.2134 ± 0.0011** | **0.5587 ± 0.0092** | 0.116 ± 0.017 | 4e-05 to 8e-05 |
| `sft` then baseline | 1 + 1 | 1.8704 | 0.2194 | 0.5568 | 0.143 | 2.2e-04 |
| `sft` then RLCD | 1 + 1 | 1.9129 | 0.2244 | 0.5457 | 0.125 | 3.3e-03 |
| RLCD from scratch | 2 | 1.8933 | 0.2220 | 0.5542 | 0.114 | 1.3e-03 |
| `bert-base-uncased` `marker`, 3 epochs, 3 seeds (section 9) | 3 | 1.8676 ± 0.0076 | 0.2195 | 0.5514 | 0.110 | |
| `ModernBERT-base` `marker`, 3 epochs, 3 seeds (section 9) | 3 | 1.8549 ± 0.0119 | 0.2174 | 0.5542 | 0.102 | |

- The seed 17 `sft` weights are published at
  [Ruivalim/exu-qwen3-0.6b-goemotions](https://huggingface.co/Ruivalim/exu-qwen3-0.6b-goemotions)
  with a standalone runtime and a model card. The package does not load them.
- One epoch of the decoder on the full split is the best GoEmotions number
  measured so far, and it holds across seeds: 1.8417 ± 0.0019, against 1.8676 ±
  0.0076 for BERT and 1.8549 ± 0.0119 for ModernBERT, each trained for three
  epochs. The gap to ModernBERT is 0.013, seven times the decoder's own spread. The
  three seeds land within 0.004 of each other, tighter than any encoder block.
- More data moved the ceiling (1.8653 on 16,000 rows, 1.8435 on 46,370) but did not
  open a second epoch: the direct loss also loses 0.027 when it continues, with the
  training loss still falling (1.27). At this learning rate the decoder's peak on
  GoEmotions is one pass over the data whatever the split size.
- Given the same starting point and the same extra updates, RLCD loses 0.042 more
  than the direct loss does, and it makes confident misses fifteen times as often.
  From scratch with two epochs it lands between the two, and 0.050 behind the
  one-epoch `sft`. Seventh and eighth comparisons; RLCD has not won one on any
  backbone.
- A second stage started at the peak has no room to improve in either mode, so
  this does not test "RLCD on a model that still has room". A lower learning rate,
  or a second stage after a shorter first one, would.

### Measuring Hate Speech, three seeds

The `score` kind: four ordinal facets on 3 or 5 levels, soft targets, RPS in the
reward. One epoch, `--max-length 512` (no question cut). Seed 42 ran on an RTX 5090
(27 minutes), seeds 17 and 23 on the RTX 3060s at batch 4 with `--grad-accum 4`
(about three hours each). Test numbers calibrated. Encoder rows are the section 9
`marker` numbers, three epochs.

| Model | NLL | Brier | RPS | Ordinal MAE | Accuracy | ECE |
| --- | --- | --- | --- | --- | --- | --- |
| decoder, 1 epoch | 1.0546 ± 0.0059 | 0.1961 ± 0.0024 | 0.0498 ± 0.0011 | 0.4686 ± 0.0061 | 0.6136 ± 0.0109 | 0.083 ± 0.011 |
| `bert-base-uncased`, 3 epochs | 1.0515 ± 0.0043 | 0.1942 ± 0.0019 | 0.0486 ± 0.0010 | 0.4592 ± 0.0071 | 0.6162 ± 0.0071 | 0.082 ± 0.003 |
| `ModernBERT-base`, 3 epochs | 1.0516 ± 0.0041 | 0.1954 ± 0.0012 | 0.0485 ± 0.0011 | 0.4572 ± 0.0071 | 0.6088 ± 0.0023 | 0.083 ± 0.005 |

Bars: `prior` NLL 1.2838, RPS 0.1039.

- A tie, with the decoder a hair behind on every proper score: 0.003 of NLL and
  0.001 of RPS, inside one spread. The first kind where the decoder does not lead,
  and the only one here where the encoders had three epochs against its one; whether
  a second epoch helps the decoder on this task was not run (on GoEmotions it did
  not).
- Confident-miss rate 0 on every seed. Fitted temperatures 1.06 to 1.12, against
  1.55 ± 0.28 for the BERT arm: the decoder is closer to calibrated as trained.

### BoolQ, three seeds, on the 3060s

One epoch, batch 4 with `--grad-accum 4` (16 questions per update, as everywhere
else), `--max-length 512` (cuts 37 of 10,644 questions), test numbers calibrated.
The `noul` kind: two sequences per question that differ in one token, `No` or
`Yes`. About 30 minutes per run on an RTX 3060 for the natural split, 21 for the
balanced one. Encoder rows are the section 5 and 9 numbers with `bert-base-uncased`,
three epochs.

| Training split | Model | Test NLL | Accuracy | Brier | Logit spread |
| --- | --- | --- | --- | --- | --- |
| natural, 62% yes | decoder, seed 17 / 23 / 42 | 0.3963 / 0.6324 / 0.5214 | 0.828 / 0.615 / 0.755 | 0.249 / 0.443 / 0.345 | 2.2 / 0.6 / 1.7 |
| natural, 62% yes | `marker`, 3 seeds, 3 epochs | 0.6356 / 0.6282 / 0.7402 | 0.703 / 0.710 / 0.694 | | |
| balanced | decoder, seed 17 / 23 / 42 | 0.6931 / 0.6841 / 0.4244 | 0.492 / 0.512 / 0.818 | 0.500 / 0.491 / 0.269 | 0.005 / 0.008 / 2.2 |
| balanced | `marker`, 3 seeds, 3 epochs | 0.6930 ± 0.0002 | 0.605 | 0.500 | |
| balanced | `marker-cls`, 3 seeds, 3 epochs | 0.6394 ± 0.0456 | 0.605 | 0.449 | |

Bars: `prior` NLL 0.663 on the natural split, 0.6931 on the balanced one.

- When the decoder leaves the plateau it is far ahead of every encoder run on this
  task: 0.828 accuracy and NLL 0.396 on seed 17, against 0.69 to 0.71 and 0.63 to
  0.74 for `marker` with three times the epochs. Seed 42 on the balanced split, which
  no encoder run ever learned, reaches 0.818.
- Whether it leaves is a matter of seed. On the balanced split two seeds stayed at
  exactly `log 2` for the whole epoch, with a logit spread under 0.01: the same
  saddle as section 5, on a decoder. On the natural split all three left, but seed
  23 barely did (spread 0.6, accuracy 0.615, the majority-class rate). The training
  loss of a run on the plateau is 0.163 for two options, which is the uniform guess
  under the composite reward, not a sign of learning.
- So the decoder does not remove the condition, it changes the odds: three
  balanced options with long, distinct candidate texts (NLI) were learned on every
  seed, two options that differ in a single token were learned on one seed in three.
  A per-option readout vector built from the candidate text, the idea behind
  `marker-cls`, is the untested remedy for this path.

## What has not been measured

- RLCD with teacher-model soft targets, which is where its authors report gains.
- RLCD as a second stage on top of a model already trained in baseline mode. Every
  RLCD run so far started from a fresh head.
- More epochs, another sigma schedule, another learning rate.
- A decoder backbone beyond section 10: more than one epoch with a decaying
  learning rate, a shorter first stage before RLCD, a prefix shared across the K
  sequences, LoRA, and a remedy for its plateau on balanced two-option questions.
- Whether four jobs sharing one GPU change a result. Runs are seeded and independent,
  and the `marker` baseline matches the one-job-per-GPU number, but no run was
  repeated both ways.
- The `noul` kind beyond BoolQ, and BoolQ beyond one question wording.
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
