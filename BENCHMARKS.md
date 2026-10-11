# Benchmarks

Everything measured on Exu so far, with the numbers as they came out. Nothing here is
a published result of someone else, and nothing is tuned to look good: the project
is pre-alpha, the runs are small, and several of them say the toolkit's headline
method loses to its own baseline.

Last updated 2026-10-11.

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
  fine-tuning on teacher-model soft targets, is section 11, on one dataset and one
  backbone.

## Setup

| | |
| --- | --- |
| Training machine | 2x NVIDIA RTX 3060 12 GB, one job per GPU |
| Rented machine, section 9 | 1x NVIDIA RTX 5090 32 GB, up to four jobs sharing the card |
| Training machine, section 12 | the same 2x RTX 3060 12 GB, one job per GPU, bfloat16 base |
| Rented machines, section 11 | 1x NVIDIA A40 48 GB or RTX 3090 24 GB, RunPod secure cloud, one job at a time |
| Probe machine | 1x NVIDIA RTX 4060 8 GB, shared with a desktop session |
| Section 13 | GoEmotions on the 2x RTX 3060 12 GB, one job per GPU; Measuring Hate Speech on the RTX 4060, one job at a time (two did not fit in 8 GB) |
| Section 14 | Exu arms on the 2x RTX 3060 12 GB (one decoder seed on a rented RTX 3090); Exu decoder on Qwen3.5-0.8B on rented RTX PRO 4500, A40, RTX A6000 and RTX 4090 pods, one run per pod, and on Qwen3.5-2B on A40 or RTX A6000 pods; Exu decoder with LoRA on the 2x RTX 3060 (0.8B and 2B); Unsloth 0.8B on the RTX 4060, 2B and 4B on a rented RTX 3090 (seed 3407) and RTX 4090 (seeds 17, 23); Clef on a rented GPU; latency on the RTX 4060 |
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
It has not won one yet, and section 10 adds three more with a decoder, none won.

Where that leaves the method. In every comparison here the target distribution was
in hand, from human raters. In that setting the direct objective is the optimal
estimator by construction: a strictly proper score is maximised by the truth, and
its gradient points at it. The policy gradient is a noisy estimate of that same
gradient with the bias of the sigma-smoothed optimum on top (section 8), so at best
it ties, and the confident-miss rates say it does not reach the tie. The setting in
which a policy gradient earns its place is the one with no target, only a reward
per sampled answer: a judge, a preference. Scale is not the missing piece on its
own: the gap did not shrink from 16,000 to 46,370 rows, nor from 110M to 600M
parameters. What scale can change is the availability of such targets. Section 11
runs the nearest cheap test, a language model as teacher on 5,000 GoEmotions
comments, which is the setting the method's authors report gains in. RLCD lost
there too, but that teacher hands over a whole distribution, so the direct loss
still has its target and the reward-only setting stays untested.

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
These runs used a probe outside the package. Since then the same layout and head
are the package's `--backbone decoder`: loaded into `DecoderDecisionModel`, the
weights of the GoEmotions `sft` run below build token-identical sequences for all
5,884 test questions and give the same uncalibrated test NLL, 1.84898 (package in
float32 on an RTX 4060, probe in bfloat16 autocast). Preliminary in every respect.

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

## 11. RLCD against a teacher model's targets, 2026-09-24

The setting the method's authors report gains in: fine-tuning on soft targets
produced by a teacher model instead of by people. 5,000 comments sampled with a
fixed seed from the GoEmotions training split, in two datasets that differ only in
the training targets. `human` keeps the raters' share of marks per label. `teacher`
replaces it with what `gpt-6-luna` (through the `codex` CLI) estimated for the same
quantity: the share of marks a group of annotators would give each of the 28
emotions. Validation, calibration and test are the full human splits in both
(2,869 / 2,886 / 5,884), so every arm is calibrated and scored against the raters.
A first teacher, `claude-sonnet-4-6`, ran out of quota after 400 comments and was
dropped: a dataset uses one teacher, never a mix.

The teacher is weak on this task. Its top label matches the raters' top label on
31% of the training comments, and its targets are flatter (mean entropy 1.21
against 0.86).

Qwen3-0.6B through the package, `exu-train --backbone decoder`, two epochs, batch 8
with `--grad-accum 2`, `--max-length 128`, `--grad-checkpointing`, backbone lr
`1e-5`, head lr `1e-4`, RLCD with its defaults. About 44 minutes per arm on the A40
and on the RTX 3090 in bfloat16, 68 on the RTX 3090 in float32.

The `bf16` rows ran with a bug since fixed: the package loaded Qwen3 in its
published dtype, bfloat16, and trained with bfloat16 master weights, where most
updates at a `1e-5` learning rate round away. The backbone now always loads in
float32, with bfloat16 only as autocast, and seed 17 was rerun that way. Seed 42 was
not run.

| Target | Mode | Run | Test NLL | Brier | Accuracy | ECE | Confident-miss rate, last epoch |
| --- | --- | --- | --- | --- | --- | --- | --- |
| human | baseline | bf16, seed 17 | **1.9495** | **0.232** | **0.528** | 0.083 | 0 |
| human | RLCD | bf16, seed 17 | 2.0100 | 0.245 | 0.511 | 0.071 | 1.4e-04 |
| human | baseline | bf16, seed 23 | **1.9455** | **0.233** | **0.530** | 0.102 | 0 |
| human | RLCD | bf16, seed 23 | 2.0044 | 0.245 | 0.511 | 0.118 | 4.9e-04 |
| human | baseline | fp32, seed 17 | **1.9228** | **0.229** | **0.550** | 0.118 | 0 |
| human | RLCD | fp32, seed 17 | 2.0088 | 0.239 | 0.518 | 0.097 | 4.2e-04 |
| teacher | baseline | bf16, seed 17 | **2.3106** | **0.325** | **0.389** | 0.128 | 1.0e-04 |
| teacher | RLCD | bf16, seed 17 | 2.3533 | 0.335 | 0.366 | 0.115 | 3.6e-04 |
| teacher | baseline | bf16, seed 23 | **2.2808** | **0.316** | **0.392** | 0.114 | 0 |
| teacher | RLCD | bf16, seed 23 | 2.3303 | 0.329 | 0.368 | 0.107 | 7.2e-04 |
| teacher | baseline | fp32, seed 17 | **2.3687** | **0.340** | **0.368** | 0.129 | 1.0e-03 |
| teacher | RLCD | fp32, seed 17 | 2.3908 | 0.341 | 0.354 | 0.127 | 2.6e-03 |

Test numbers calibrated, from `training.json`. The confident-miss rate is measured
on the training targets, so on the teacher's for the teacher arms. Bars on the same
test split: `uniform` NLL 3.3322, `prior` 2.8269 fitted on the human training
targets (2.9522 on the teacher's), majority class accuracy 0.293.

- RLCD loses all six pairs, by 0.022 to 0.086 of NLL, and does no better on Brier or
  accuracy in any of them. With sections 9 and 10 that is fourteen comparisons on
  three backbones, none won.
- Its calibrated ECE is lower in five of the six pairs. ECE here compares against
  the top label only; the proper scores decide, and they go the other way.
- Against the teacher's targets RLCD makes confident misses more often in every
  pair, as it did against human targets in sections 9 and 10.
- The teacher costs every arm 0.33 to 0.45 of NLL against the human arm of the same
  mode and run. All twelve arms beat `prior` by a wide margin, so the teacher arms
  learned something, just less of what the raters do.
- Float32 moved the human baseline down by 0.027 and the teacher baseline up by
  0.058, on one seed each. That does not settle which precision trains better here;
  the fix stands on the rounding argument above. The gap between the modes kept its
  sign in both.
- What this does not test. The teacher returned a whole distribution per comment,
  so the direct loss had the full target and is the optimal estimator again, as in
  section 9. A policy gradient has no direct alternative only when there is a
  reward per sampled answer and no distribution, a judge or a preference, and that
  was not run.

```bash
D=path/to/goemotions-teacher-5000.jsonl
uv run exu-train --backbone decoder --mode baseline \
  --train $D --train-split train --validation $D --validation-split validation \
  --calibration $D --calibration-split calibration --test $D --test-split test \
  --epochs 2 --batch-size 8 --grad-accum 2 --max-length 128 --grad-checkpointing \
  --calibrate --seed 17 --output artifacts/teacher-baseline-s17
```

## 12. A larger base read through its own labels, 2026-09-26

Prompted by Mica v0.1 4B ([akivet/Mica-v0.1-4B](https://github.com/akivet/Mica-v0.1-4B),
Apache-2.0), a fine-tune of `Qwen/Qwen3.5-4B` that reads a decision out of the
language-model head: every option gets a one-token label (`A`, `B`, ..., or `No` /
`Yes` for a `noul`), all options sit in one prompt, and the distribution is the
softmax of those label logits at the answer position. One sequence per question, no
new head. Nothing here is in the package: these are probes outside the repository,
run with the package's metrics and calibration, and the prompt is Mica's own
rendering code, run from its repository.

Test NLL calibrated, temperatures fitted on the calibration split (up to 1,000
rows). Batch 1, bfloat16, reference PyTorch kernels for the Gated DeltaNet layers.

### Zero-shot, no training

GoEmotions and Measuring Hate Speech on a seeded sample of 2,000 test rows (prior on
the sample 2.8355 and 1.2853), ChaosNLI and BoolQ on the whole test split.

| Model | GoEmotions | ChaosNLI | BoolQ | Measuring Hate Speech |
| --- | --- | --- | --- | --- |
| `prior` | 2.8355 | 1.0982 | 0.6630 | 1.2853 |
| `Qwen3-0.6B` | 3.3420 (T at bound) | 1.1675 (T at bound) | 0.6746 (T at bound) | 1.4804 (T at bound) |
| `Qwen3.5-0.8B` | 3.0704 | 1.0417 | 0.6073 | 1.4126 |
| `Qwen3.5-2B` | 3.2538 | 1.0140 | 0.5197 | 1.4130 |
| `Qwen3.5-4B` | 2.7906 | **0.9524** | 0.4482 | **1.2153** |
| Mica v0.1 4B (revision `ca36594`) | **2.6304** | 0.9636 | **0.3277** | 1.2961 |

- The current decoder base, `Qwen3-0.6B`, cannot be read this way: every fitted
  temperature sits on the upper bound (10) and every NLL is above `prior`.
- Untrained, `Qwen3.5-4B` already beats every trained model in this file on
  ChaosNLI (0.9524 against 0.9979 for the section 10 decoder). On the two tasks whose
  target is one population's taste, GoEmotions and Measuring Hate Speech, zero-shot
  stays near `prior`.
- Order: with the options reversed, the top answer stays the same on 63% (GoEmotions)
  and 76% (ChaosNLI) of rows for `Qwen3.5-4B`, 74% and 97% for Mica, and below chance
  on ChaosNLI for the 0.8B and 2B models, which pick a position more than an answer.
- Every model is overconfident zero-shot: fitted temperatures 1.0 to 4.5 wherever
  the fit did not hit the bound.

### LoRA fine-tune, three seeds

`Qwen3.5-4B` frozen in bfloat16, a rank-16 LoRA (alpha 32, float32) on the attention
and Gated DeltaNet projections (`q/k/v/o_proj`, `in_proj_qkv/z/a/b`, `out_proj`;
14.4M trainable), the log score against the soft target over the label logits,
`choice` options shuffled per example. 8,500 training rows per dataset (all 8,473 for
BoolQ), the same rows for every seed, one epoch, batch 16, lr `1e-4` with 5% warmup
and linear decay, seeds 17, 23 and 42: the seed changes the order, the shuffles and
the adapter initialisation. About 1.4 to 1.8 rows per second and 8.4 to 8.8 GiB peak
on an RTX 3060, 79 to 102 minutes of training per run. Whole test splits.

| Dataset | LoRA 4B, 8,500 rows | Best trained model above | Its training rows |
| --- | --- | --- | --- |
| GoEmotions | 1.8443 ± 0.0028 | **1.8417 ± 0.0019** (decoder, section 10) | 46,370 |
| ChaosNLI | **0.9303 ± 0.0182** | 0.9979 ± 0.0139 (decoder, section 10) | 16,000 |
| BoolQ | **0.2108 ± 0.0033** (accuracy 0.922) | 0.3963 (decoder seed 17; 0.5214 and 0.6324 on the others) | 8,473 |
| Measuring Hate Speech | **1.0221 ± 0.0033** (RPS 0.0437) | 1.0515 ± 0.0043 (`bert-base-uncased`, RPS 0.0486) | 55,656 |

Per seed (17, 23, 42): GoEmotions 1.8474, 1.8434, 1.8421; ChaosNLI 0.9305, 0.9484,
0.9121; BoolQ 0.2139, 0.2111, 0.2074; Measuring Hate Speech 1.0232, 1.0183, 1.0247.

- With the same or fewer training rows (the same on BoolQ, about half on ChaosNLI, a
  sixth on Measuring Hate Speech), the adapter wins three of four on every seed and
  ties GoEmotions, 0.003 behind a model trained on 5.5 times its data.
- The seed spread is small next to those gaps: 0.003 on three datasets, 0.018 on
  ChaosNLI, whose worst seed (0.9484) is still ahead of the decoder's mean.
- Shuffling the options in training removes most of the order effect: the top answer
  survives reversal on 88% to 94% of GoEmotions rows and 94% to 96% of ChaosNLI rows.
- Fitted temperatures 1.01 to 1.26, except ChaosNLI at 3.1 to 3.3: trained on
  MultiNLI's hard labels and tested on ChaosNLI's spread, as the encoders were in
  section 9.

### LoRA fine-tune, the whole training split

The same recipe, seed 17, with every training row of the two datasets where the other
side had used many more: the comparison at equal data.

| Dataset | LoRA 4B, 8,500 rows (3 seeds) | LoRA 4B, all rows (seed 17) | Best trained model above, same rows |
| --- | --- | --- | --- |
| GoEmotions (46,370 rows) | 1.8443 ± 0.0028 | **1.7904** (accuracy 0.574) | 1.8417 ± 0.0019 (decoder, section 10) |
| Measuring Hate Speech (55,656 rows) | 1.0221 ± 0.0033 | **1.0004** (RPS 0.0400) | 1.0515 ± 0.0043 (`bert-base-uncased`, 3 epochs, RPS 0.0486) |

- At equal data the GoEmotions tie becomes a gap of 0.051, about twenty times the
  spread of either side's seeds. Measuring Hate Speech gains another 0.022, and the
  adapter saw each row once where the encoder saw it three times.
- One seed. The 8,500-row spread (0.003) says a second seed is unlikely to close
  either gap, but it has not been run.
- Cost: 7.9 hours (GoEmotions) and 8.7 hours (Measuring Hate Speech) of training on
  one RTX 3060, 1.6 to 1.8 rows per second, 8.7 GiB peak. One GoEmotions row above
  1,024 tokens was skipped. Temperatures 1.01 and 1.04; the top answer survives
  reversal on 94% of GoEmotions rows.
- One adapter trained on all four datasets has not been run.

### Fast kernels and latency, 2026-09-27

Every run above used the reference PyTorch implementation of the Gated DeltaNet
layers. `flash-linear-attention` (0.5.2, Triton) replaces it without code changes;
`causal-conv1d` was not installed, since the convolution it would replace is a
depthwise `conv1d` of width 4.

Equivalence. On 300 seeded test rows with a trained adapter, the label logits differ
by at most 0.125 (0.25 on one row), one bfloat16 step at their magnitude; the top
answer is unchanged on 300 of 300 ChaosNLI rows and 294 of 300 GoEmotions rows, the
six flips being near-ties among 28 options. Trained end to end with the kernels, same
rows and seed 17, whole test split:

| Dataset | Reference kernels | `flash-linear-attention` | Seed spread (section above) |
| --- | --- | --- | --- |
| BoolQ | 0.2139 | 0.2138 | ± 0.0033 |
| Measuring Hate Speech | 1.0232 | 1.0235 | ± 0.0033 |

Speed on one RTX 3060: training runs 1.4 to 1.6 times faster (BoolQ 1.4 to 2.2 rows
per second, 100 to 64 minutes; Measuring Hate Speech 1.8 to 2.5, 79 to 56 minutes),
at 8.3 to 8.5 GiB peak against 8.4 to 8.8.

Latency per question, batch 1, CUDA synchronised, the same 300 test rows, p50. The
package models are minimal checkpoints of the same architecture (latency does not
depend on the weights), timed through `DecisionRuntime.decide`; the encoder runs in
float32, the 4B in bfloat16.

| Model | ChaosNLI (3 options) | GoEmotions (28 options) |
| --- | --- | --- |
| `bert-base-uncased`, marker readout | 7.8 ms | 8.5 ms |
| `Qwen3-0.6B`, decoder backbone (one sequence per option) | 50 ms | 212 ms |
| `Qwen3.5-4B` label readout, reference kernels | 147 ms | 168 ms |
| `Qwen3.5-4B` label readout, `flash-linear-attention` | 115 ms | 125 ms |

- The 4B pays one prefill per question, so its latency barely moves with the number
  of options; the section 10 decoder pays one per option and is slower than the 4B
  from GoEmotions' 28 options.
- The encoder stays an order of magnitude ahead of both. Batched calls of several
  questions were not measured for the 4B.

### One adapter, three datasets, the fourth held out, 2026-09-27

The same recipe with the fast kernels, seed 17, one adapter trained on three datasets
at once: the same 8,500 rows of each that the single-task adapters used (25,473 rows,
25,500 without BoolQ), shuffled together, one epoch. Four adapters, each leaving one
dataset out, all scored on the four whole test splits. 2.8 to 3.1 hours of training
each on one RTX 3060, 2.2 to 2.6 rows per second, 8.5 GiB peak, no row skipped.

Zero-shot `Qwen3.5-4B` was scored again on the whole GoEmotions and Measuring Hate
Speech test splits for this comparison: 2.7808 and 1.2045 (`prior` 2.8267 and 1.2838
on the whole splits).

| Held out | GoEmotions | ChaosNLI | BoolQ | Measuring Hate Speech |
| --- | --- | --- | --- | --- |
| GoEmotions | *2.9337* | 0.9177 | 0.2028 | 1.0231 |
| ChaosNLI | 1.8499 | *0.8804* | 0.2064 | 1.0210 |
| BoolQ | 1.8534 | 0.9208 | *0.3407* | 1.0241 |
| Measuring Hate Speech | 1.8480 | 0.9038 | 0.2019 | *1.2472* |
| Single-task adapter, 3 seeds | 1.8443 ± 0.0028 | 0.9303 ± 0.0182 | 0.2108 ± 0.0033 | 1.0221 ± 0.0033 |
| `Qwen3.5-4B` zero-shot | 2.7808 | 0.9524 | 0.4482 | 1.2045 |
| `prior` | 2.8267 | 1.0982 | 0.6630 | 1.2838 |

In italics, the dataset the adapter never saw.

- On the datasets it trained on, one adapter does the work of three: Measuring Hate
  Speech lands inside the single-task seed range, ChaosNLI inside it or below it
  (0.9038 to 0.9208 against 0.9121 to 0.9484), BoolQ 0.004 to 0.009 better than the
  single-task mean (all three below its best seed), GoEmotions 0.004 to 0.009 worse,
  one to three times its seed spread.
- Held out, the two tasks with one right answer transfer. BoolQ reaches 0.3407
  (accuracy 0.860) against 0.4482 zero-shot, without ever seeing a `noul` question in
  training. ChaosNLI reaches 0.8804 (accuracy 0.724), better than any adapter trained
  on MultiNLI (0.9121 to 0.9484), with a fitted temperature of 1.57 against 3.1 to 3.3:
  MultiNLI's hard labels were what made those adapters overconfident on ChaosNLI's
  spread.
- Held out, the two tasks whose target is one population's taste get worse than the
  untouched base, and worse than `prior`: GoEmotions 2.9337, Measuring Hate Speech
  1.2472 (accuracy 0.459 against 0.559 zero-shot). Training on the other tasks moves
  the base away from these annotators, not towards them. Order robustness on
  GoEmotions drops with it: the top answer survives reversal on 47% of rows, against
  62% zero-shot and 90% to 92% for the adapters that saw GoEmotions.
- One seed per arm. The held-out gaps to zero-shot (0.04 to 0.15) are far outside the 0.003 to
  0.018 seed spread of the single-task runs; the in-task differences are not.
- One adapter trained on all four datasets has still not been run; the in-task rows
  above are the closest measurement.

### Smaller bases, 2026-09-28

The single-task recipe with the fast kernels, unchanged, on `Qwen3.5-2B` (7.4M
trainable) and `Qwen3.5-0.8B` (5.5M): the same 8,500 rows (8,473 for BoolQ), seed 17
only, whole test splits. The 4B column is the three-seed mean from above.

| Dataset | LoRA 0.8B | LoRA 2B | LoRA 4B, 3 seeds | 4B zero-shot | `prior` |
| --- | --- | --- | --- | --- | --- |
| GoEmotions | 1.9356 | 1.8787 | **1.8443 ± 0.0028** | 2.7808 | 2.8267 |
| ChaosNLI | 1.0049 | 0.9659 | **0.9303 ± 0.0182** | 0.9524 | 1.0982 |
| BoolQ | 0.3503 (accuracy 0.847) | 0.2698 (accuracy 0.891) | **0.2108 ± 0.0033** (accuracy 0.922) | 0.4482 | 0.6630 |
| Measuring Hate Speech | 1.0608 | 1.0435 | **1.0221 ± 0.0033** | 1.2045 | 1.2838 |

Zero-shot, the same bases score 1.0417 (0.8B) and 1.0140 (2B) on ChaosNLI and 0.6073
and 0.5197 on BoolQ; their GoEmotions and Measuring Hate Speech numbers exist only on
the 2,000-row sample of the first table, all above `prior`.

- Every dataset orders by size, 4B ahead of 2B ahead of 0.8B. The 2B trails the 4B by
  0.021 to 0.059, beyond the 4B seed spread on three datasets and twice it on
  ChaosNLI. One seed for the smaller bases.
- Against the package's own models the 2B still wins three of four: BoolQ 0.2698
  against 0.3963 for the section 10 decoder, ChaosNLI 0.9659 against 0.9979, Measuring
  Hate Speech 1.0435 against 1.0515 ± 0.0043 for `bert-base-uncased` on 6.5 times the
  rows. GoEmotions loses by 0.037 to the decoder trained on 46,370 rows. The 0.8B wins
  only BoolQ; ChaosNLI sits inside the decoder's spread.
- On ChaosNLI neither smaller adapter beats the untrained 4B (0.9524). Temperatures
  3.3 (2B) and 3.4 (0.8B), as the 4B's 3.1 to 3.3: before calibration both are worse
  than `prior` (1.38 and 1.41), the same MultiNLI hard-label overconfidence. Every
  other temperature is 0.99 to 1.24.
- Shuffling fixes the position bias of the smaller bases: on ChaosNLI the top answer
  survives reversal on 88% of rows for both, against 9% (2B) and 19% (0.8B)
  zero-shot. GoEmotions 88% (2B) and 83% (0.8B).
- Cost on one RTX 3060: 37 to 39 minutes per run for both sizes, 3.6 to 3.8 rows per
  second, peak 3.9 GiB (2B) and 1.7 GiB (0.8B), against 56 to 64 minutes at 8.3 to
  8.5 GiB for the 4B. The 0.8B trains no faster than the 2B, so at batch 1 something
  other than the model's size bounds the rate; what, was not measured. Latency of the
  smaller bases was not measured either.

## 13. Feedback on one sampled answer, simulated judge, 2026-09-28

Every RLCD arm above scored its samples against the whole target, where the direct
loss is the better estimator. This section tests the setting where a sampled
estimator is needed: the model draws an answer, a judge says yes or no about that
one answer, and the target is never read. The estimators are in `exu/bandit.py` and
the math is in `docs/algorithm.md`. Nothing in the data format carries real
verdicts yet, so the judge is simulated: it says yes to answer `k` with probability
`y_k`, one annotator drawn at random agreeing or not. The runs are a probe outside
the CLI: a driver swaps the loss of `--mode baseline` per arm, everything else is
section 1's recipe (`bert-base-uncased`, 3 epochs, `--option-shuffle`, batch 8 with
`--grad-accum 2`), calibrated, scored by `exu-evaluate`.

Arms. `G` is the number of judged answers per question per pass, drawn from the
model mixed with `0.1` of uniform (`-e1`: `1.0`, all uniform).

- `baseline`: the whole target, the ceiling.
- `sampled`: one annotator's label drawn from the target per row and pass, a full
  categorical label instead of a verdict.
- `verdict-gG`: binary log score of `q_k` against the verdict, on each sampled `k`.
- `verdict-all`: every option judged once per pass, no sampling (28 verdicts).
- `dr-g8`: doubly robust estimate of the target from the same verdicts, scored with
  the composite.

### GoEmotions, three seeds

Bars on the test split: `uniform` 3.3322, `prior` 2.8267.

| Arm | Test NLL | Brier | Accuracy | ECE | Temperature | Order stability | NLL vs `baseline`, paired |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `baseline` | **1.8627 ± 0.0138** | **0.2190 ± 0.0058** | **0.5502 ± 0.0058** | 0.1160 ± 0.0070 | 1.096 ± 0.036 | 0.967 ± 0.006 | |
| `verdict-all` | 1.8935 ± 0.0070 | 0.2219 ± 0.0015 | 0.5457 ± 0.0063 | 0.1185 ± 0.0155 | 0.985 ± 0.078 | 0.950 ± 0.002 | +0.031 ± 0.021 (0/3) |
| `sampled` | 1.9112 ± 0.0159 | 0.2259 ± 0.0032 | 0.5427 ± 0.0004 | 0.1150 ± 0.0074 | 1.071 ± 0.041 | 0.937 ± 0.002 | +0.048 ± 0.017 (0/3) |
| `verdict-g28` | 1.9594 ± 0.0157 | 0.2260 ± 0.0050 | 0.5479 ± 0.0029 | 0.1069 ± 0.0126 | 1.103 ± 0.075 | 0.966 ± 0.006 | +0.097 ± 0.002 (0/3) |
| `verdict-g8` | 1.9898 ± 0.0084 | 0.2282 ± 0.0021 | 0.5471 ± 0.0034 | 0.1094 ± 0.0114 | 1.020 ± 0.047 | 0.958 ± 0.003 | +0.127 ± 0.019 (0/3) |
| `verdict-g8-e1` | 1.9938 ± 0.0144 | 0.2365 ± 0.0014 | 0.5218 ± 0.0117 | 0.0961 ± 0.0263 | 1.101 ± 0.056 | 0.947 ± 0.008 | +0.131 ± 0.028 (0/3) |
| `verdict-g4` | 2.0610 ± 0.0201 | 0.2378 ± 0.0027 | 0.5292 ± 0.0106 | 0.1049 ± 0.0133 | 1.063 ± 0.067 | 0.946 ± 0.017 | +0.198 ± 0.019 (0/3) |
| `dr-g8` | 2.1866 ± 0.0463 | 0.2562 ± 0.0061 | 0.5206 ± 0.0077 | 0.1190 ± 0.0383 | 1.674 ± 0.228 | 0.936 ± 0.011 | +0.324 ± 0.033 (0/3) |
| `verdict-g1` | 2.9669 ± 0.3510 | 0.3881 ± 0.0672 | 0.2450 ± 0.1641 | 0.0372 ± 0.0129 | 1.030 ± 0.305 | 0.675 ± 0.534 | +1.104 ± 0.341 (0/3) |

Per seed (17, 23, 42): `baseline` 1.8512, 1.8781, 1.8589; `verdict-all` 1.8987,
1.8856, 1.8962; `sampled` 1.9189, 1.9217, 1.8929; `verdict-g8` 1.9864, 1.9837,
1.9994; `verdict-g1` 2.9363, 3.3322, 2.6322.

- The verdict loss learns from yes-or-no feedback alone. With 8 verdicts per
  question per pass it lands 0.127 of NLL behind the whole target and 0.84 ahead of
  `prior`; with every option judged, 0.031 behind, closer than one full annotator
  label per pass (`sampled`, 0.048 behind). `verdict-all` against `sampled`, paired:
  better on seeds 17 and 23, 0.003 worse on 42. It never beats the baseline.
- Test NLL falls steadily with the number of verdicts: 2.97 (1), 2.06 (4), 1.99 (8),
  1.96 (28 drawn with replacement), 1.89 (28, each option once). Drawing 28 with
  replacement repeats some options and misses others, and costs 0.066.
- One verdict per question per pass is not enough on 28 options: seed 23 stayed at
  `uniform` exactly (3.3322), the other two ended at 2.94 and 2.63, near `prior`.
- The doubly robust estimate, from the same 8 verdicts, trails the verdict loss by
  0.197 and needs a temperature of 1.67 against 1.02. The likely cause is the
  importance weights, which the verdict loss does not have; this was not isolated.
- Full exploration (`-e1`) ties the default on NLL: +0.004 on average, better on
  seeds 23 and 42 by 0.003 and 0.008, worse on 17 by 0.023. It loses on Brier and
  accuracy in all three. Drawing from the model's own
  distribution did not leave the rare emotions behind in a way the proper scores see.
- The same probe run before the verdict loss existed, seed 17 on the RTX 4060: one
  verdict per question with the doubly robust estimate ended at `uniform` (3.3322),
  with plain importance weighting at 9.08 before calibration (collapsed on the
  majority class). On free per-row logits with the same targets, removing gradient
  clipping or switching Adam for SGD did not change that; information per row did.
- Every arm beats `prior` except `verdict-g1`. Its ECE of 0.037, the lowest in the
  table, is the ECE of a model near uniform, not a calibration result.

### Measuring Hate Speech, one seed

Seed 17, on the RTX 4060. Ordinal questions with 3 or 5 levels, so a verdict
carries much more of the target than on GoEmotions. The verdict loss scores the
binary event only, so the RPS term of the composite does not enter the bandit arms.
Bars: `uniform` 1.4817, `prior` 1.2838.

| Arm | Test NLL | Brier | RPS | Accuracy | ECE | Temperature | NLL vs `baseline` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `baseline` | **1.0495** | **0.1926** | **0.0483** | **0.6172** | 0.0835 | 1.75 | |
| `verdict-g8` | 1.0647 | 0.1969 | 0.0500 | 0.6124 | 0.0800 | 1.58 | +0.015 |
| `verdict-g4` | 1.0703 | 0.1995 | 0.0515 | 0.6118 | 0.0773 | 1.24 | +0.021 |
| `verdict-g1` | 1.0883 | 0.2044 | 0.0540 | 0.6091 | 0.0796 | 1.47 | +0.039 |

- On 3 to 5 options one verdict per question per pass already learns: 0.039 behind
  the whole target and 0.196 ahead of `prior`. Eight verdicts close it to 0.015,
  about three and a half times the 0.0043 seed spread of section 4's
  `bert-base-uncased`. One seed, so read these as a direction.
- The baseline reproduces section 4's number (1.0515 ± 0.0043).

What this does and does not show. The judge is simulated from the annotators, so a
verdict is exactly as noisy as one annotator; a real judge (a model, a test, a user)
brings its own bias, and nothing here measures that. The comparison is at equal
epochs, not at equal annotation cost: `verdict-all` asks 28 questions per row and
pass where `sampled` asks one. What it does show is that a proper score on the
sampled answer, with no importance weight, recovers most of what the full target
teaches, and that its cost is the number of verdicts per question, not the
optimiser.

```bash
# The driver, the run queue and the aggregator are working files outside the
# repository. Each arm runs:
uv run python bandit_probe.py verdict-g8 -- --encoder google-bert/bert-base-uncased \
  --train $D --train-split train --validation $D --validation-split validation \
  --calibration $D --calibration-split calibration --test $D --test-split test \
  --calibrate --option-shuffle --epochs 3 --batch-size 8 --grad-accum 2 --seed 17 \
  --output artifacts/bandit/exp/verdict-g8-s17
```

## 14. Typed decisions against Unsloth and Clef, 2026-10-08

Two other projects train the same kind of model: Unsloth's `FastDecisionModel`
(Qwen3.5 plus LoRA and a joint head that answers every question about a state from
one prefill) and Cloudflare's Clef (Qwen3.8-27B and Qwen3.5-9B with a joint schema
head, published weights, no training here). This section puts Exu next to both on
the benchmark Unsloth's guide uses, scored with the same proper scores.

**Data.** `typed-decisions` (Apache-2.0): synthetic agent and workflow traces, 1,200
training cases and 400 test cases, five questions per case (`choice`, `score` and
`noul`, 3 to 5 options). The gold is soft, the mean of three teacher samples; the
dataset card puts the teacher's agreement with itself at 0.735. The training cases
were split by a hash of the case id into train (884 cases, 4,420 questions),
validation (174) and calibration (142); the official 400 test cases (2,000
questions) are the test split for every arm. A `family` is workflow plus question,
so the temperature map and the per-question prior see the same cells.

**Metrics.** Accuracy against `argmax(gold)`, `KL(gold || pred)`, Brier summed over
options and NLL against the soft gold, all averaged over the 2,000 questions; these
are the dataset card's own metrics plus NLL. `uniform` reproduces the card's numbers
exactly (KL 0.444, Brier 0.238). `prior` is fitted on our train split, and our ECE is
10-bin top-1 ECE; neither matches the card's values, so ECE is only compared inside
this table.

**Arms.**

- Exu `--mode baseline`, soft cross-entropy, 5 epochs, `--option-shuffle`,
  `--calibrate`, `--max-length 768`. The decoder is `Qwen/Qwen3-0.6B` (encoder lr
  `1e-5`, batch 4 with `--grad-accum 4`, `--grad-checkpointing`) on a 3060, one seed
  of it on a rented RTX 3090 after it ran out of memory on the 3060. The same flags
  on `Qwen/Qwen3.5-0.8B`, Unsloth's smallest base, ran on rented GPUs (it does not
  fit on 12 GB). Both decoders ran again with `--lr-schedule cosine
  --warmup-steps 10`, the schedule and warm-up of Unsloth's recipe, at encoder lr
  `1e-5` and at `5e-5`, and `Qwen/Qwen3.5-2B` ran at `1e-5` with the cosine
  schedule only. Qwen3.5-0.8B and 2B also ran with `--lora-rank 16` (alpha 16, lr
  `2e-4`, cosine, the rank, alpha and rate of Unsloth's recipe), on the 3060s with the
  `causal_conv1d` and `flash-linear-attention` kernels. The other arms kept the lr constant, then the only
  schedule (now `--lr-schedule constant --warmup-steps 0` for a decoder). The encoders are
  `answerdotai/ModernBERT-base` (encoder lr `2.5e-5`, batch 8, `--grad-accum 2`) and `-large` (batch 4, `--grad-accum 4`, at
  `2.5e-5` and at `1e-5`).
- Unsloth 2026.10.2, its documented recipe unchanged (LoRA r 16, lr `2e-4`, batch 8
  with accumulation 4, cosine schedule), only `--epochs` and the seed changed. Its
  default is 2 epochs. Unsloth's `calibrate` fits a head temperature against the
  hard labels and folds it into the weights; `raw` arms undo that fold and read the
  logits at temperature 1.
- Clef and Clef-flash zero-shot, through Cloudflare's reference
  `joint_schema_model.py`, no training.
- `+ Exu temperature map`: Exu's per-type, per-bucket map fitted against the soft
  gold of the 142 calibration cases and applied to another model's raw logits. For
  Clef that means it is no longer zero-shot: it has seen 710 labelled questions.

| Arm | Seeds | Accuracy | KL | Brier | NLL | ECE |
| --- | --- | --- | --- | --- | --- | --- |
| Exu decoder Qwen3-0.6B | 3 | 0.779 ± 0.000 | 0.090 ± 0.003 | 0.050 ± 0.001 | 0.858 ± 0.003 | 0.155 ± 0.002 |
| Exu ModernBERT-base `marker-cls` | 3 | 0.728 ± 0.005 | 0.139 ± 0.012 | 0.076 ± 0.006 | 0.907 ± 0.012 | 0.122 ± 0.006 |
| Exu ModernBERT-base `marker-cls`, cosine | 3 | 0.721 ± 0.027 | 0.149 ± 0.025 | 0.082 ± 0.015 | 0.916 ± 0.025 | 0.122 ± 0.013 |
| Exu ModernBERT-base `marker` | 1 | 0.675 | 0.170 | 0.098 | 0.937 | 0.088 |
| Exu decoder Qwen3-0.6B, cosine, lr `1e-5` | 3 | 0.797 ± 0.003 | 0.083 ± 0.003 | 0.045 ± 0.002 | 0.850 ± 0.003 | 0.168 ± 0.002 |
| Exu decoder Qwen3-0.6B, cosine, lr `5e-5` | 1 | 0.715 | 0.153 | 0.084 | 0.920 | 0.128 |
| Exu decoder Qwen3.5-0.8B | 3 | 0.771 ± 0.003 | 0.102 ± 0.004 | 0.056 ± 0.003 | 0.869 ± 0.004 | 0.147 ± 0.003 |
| Exu decoder Qwen3.5-0.8B, cosine, lr `1e-5` | 3 | 0.794 ± 0.001 | 0.085 ± 0.001 | 0.047 ± 0.001 | 0.852 ± 0.001 | 0.164 ± 0.003 |
| Exu decoder Qwen3.5-0.8B, cosine, lr `5e-5` | 1 | 0.652 | 0.187 | 0.109 | 0.954 | 0.091 |
| Exu decoder Qwen3.5-0.8B, LoRA r16, lr `2e-4` | 3 | 0.802 ± 0.002 | 0.078 ± 0.001 | 0.043 ± 0.000 | 0.846 ± 0.001 | 0.169 ± 0.001 |
| Exu decoder Qwen3.5-2B, cosine, lr `1e-5` | 3 | 0.797 ± 0.005 | 0.082 ± 0.002 | 0.045 ± 0.001 | 0.849 ± 0.002 | 0.167 ± 0.005 |
| Exu decoder Qwen3.5-2B, LoRA r16, lr `2e-4` | 3 | **0.807 ± 0.001** | 0.075 ± 0.001 | 0.041 ± 0.001 | 0.842 ± 0.001 | 0.169 ± 0.001 |
| Exu ModernBERT-large `marker-cls`, lr `1e-5` | 3 | 0.723 ± 0.021 | 0.132 ± 0.017 | 0.072 ± 0.010 | 0.899 ± 0.017 | 0.116 ± 0.021 |
| Exu ModernBERT-large `marker-cls`, lr `2.5e-5` | 1 | 0.542 | 0.281 | 0.161 | 1.048 | 0.050 |
| Unsloth Qwen3.5-0.8B, 2 epochs, raw | 3 | 0.685 ± 0.016 | 0.166 ± 0.013 | 0.097 ± 0.008 | 0.934 ± 0.013 | 0.123 ± 0.009 |
| Unsloth Qwen3.5-0.8B, 2 epochs, hard-calibrated | 3 | 0.684 ± 0.017 | 0.329 ± 0.017 | 0.143 ± 0.008 | 1.096 ± 0.017 | 0.037 ± 0.008 |
| Unsloth Qwen3.5-2B, 2 epochs, raw | 2 | 0.752 ± 0.016 | 0.106 ± 0.002 | 0.059 ± 0.002 | 0.873 ± 0.002 | 0.151 ± 0.011 |
| Unsloth Qwen3.5-2B, 2 epochs, hard-calibrated | 2 | 0.752 ± 0.016 | 0.271 ± 0.022 | 0.102 ± 0.006 | 1.038 ± 0.022 | 0.019 ± 0.008 |
| Unsloth Qwen3.5-4B, 2 epochs, raw | 2 | 0.776 ± 0.004 | 0.090 ± 0.001 | 0.050 ± 0.002 | 0.858 ± 0.001 | 0.153 ± 0.001 |
| Unsloth Qwen3.5-4B, 2 epochs, hard-calibrated | 2 | 0.777 ± 0.005 | 0.325 ± 0.029 | 0.108 ± 0.009 | 1.092 ± 0.029 | 0.016 ± 0.001 |
| Unsloth Qwen3.5-0.8B, 5 epochs, raw | 3 | 0.777 ± 0.006 | 0.094 ± 0.003 | 0.051 ± 0.002 | 0.861 ± 0.003 | 0.148 ± 0.004 |
| Unsloth Qwen3.5-0.8B, 5 epochs, + Exu temperature map | 3 | 0.777 ± 0.006 | 0.093 ± 0.003 | 0.051 ± 0.001 | 0.860 ± 0.003 | 0.151 ± 0.005 |
| Unsloth Qwen3.5-0.8B, 5 epochs, hard-calibrated | 3 | 0.778 ± 0.006 | 0.311 ± 0.009 | 0.107 ± 0.002 | 1.078 ± 0.009 | 0.021 ± 0.003 |
| Unsloth Qwen3.5-2B, 5 epochs, raw | 3 | 0.790 ± 0.005 | 0.082 ± 0.002 | 0.045 ± 0.002 | 0.849 ± 0.002 | 0.152 ± 0.005 |
| Unsloth Qwen3.5-2B, 5 epochs, + Exu temperature map | 3 | 0.790 ± 0.005 | 0.080 ± 0.002 | 0.044 ± 0.002 | 0.847 ± 0.002 | 0.163 ± 0.005 |
| Unsloth Qwen3.5-4B, 5 epochs, raw | 3 | **0.807 ± 0.002** | 0.073 ± 0.001 | **0.040 ± 0.001** | 0.841 ± 0.001 | 0.162 ± 0.001 |
| Unsloth Qwen3.5-4B, 5 epochs, + Exu temperature map | 3 | **0.807 ± 0.002** | **0.072 ± 0.001** | **0.040 ± 0.001** | **0.839 ± 0.001** | 0.172 ± 0.002 |
| Unsloth Qwen3.5-4B, 5 epochs, hard-calibrated | 3 | 0.807 ± 0.002 | 0.423 ± 0.029 | 0.118 ± 0.005 | 1.190 ± 0.029 | 0.018 ± 0.002 |
| Clef 27B, zero-shot | 1 | 0.718 | 0.191 | 0.103 | 0.958 | 0.023 |
| Clef 27B + Exu temperature map | 1 | 0.718 | 0.144 | 0.078 | 0.912 | 0.104 |
| Clef-flash 9B, zero-shot | 1 | 0.698 | 0.209 | 0.111 | 0.977 | 0.012 |
| Clef-flash 9B + Exu temperature map | 1 | 0.698 | 0.160 | 0.084 | 0.928 | 0.103 |
| `uniform` | | 0.277 | 0.444 | 0.238 | 1.212 | 0.045 |
| `prior` | | 0.481 | 0.327 | 0.181 | 1.095 | 0.036 |

Seeds: Exu 17, 23, 42; Unsloth 3407 (its default), 17, 23. "1" means seed 17 for
Exu and 3407 for Unsloth.

- **With LoRA, the Exu decoder on Qwen3.5-2B is the closest arm to Unsloth's 4B.**
  The same adapters on the 2B reach KL 0.075 ± 0.001, accuracy 0.807 ± 0.001,
  against 0.078 ± 0.001 on the 0.8B and 0.082 ± 0.002 for both the fully fine-tuned
  Exu 2B and Unsloth's 2B; NLL and Brier agree. Unlike the full fine-tune, the
  adapters do use the larger base. Unsloth's 4B stays ahead on KL (0.073 ± 0.001)
  and ties on accuracy (0.807 ± 0.002). Seed 42 of this arm, the one with the best
  validation NLL, is published as
  [exu-qwen3.5-2b-typed-decisions](https://huggingface.co/Ruivalim/exu-qwen3.5-2b-typed-decisions).
- **With LoRA, the Exu decoder on the 0.8B passes Unsloth's 2B.** Rank-16 adapters
  on Qwen3.5-0.8B reach KL 0.078 ± 0.001 and accuracy 0.802 ± 0.002, against 0.085
  ± 0.001 for the same Exu decoder fully fine-tuned, 0.094 ± 0.003 for Unsloth on
  the same base and 0.082 ± 0.002 for Unsloth's 2B; NLL and Brier agree. Unsloth's
  4B stays ahead (0.073 ± 0.001). The adapters use the rank, alpha, rate and
  schedule of Unsloth's recipe and were the first adapter setting tried, so no
  adapter tuning was read off these numbers. What still differs from Unsloth is the
  readout (one sequence per option against one prefill per case and a joint head),
  the batch (16 against 32) and the adapted layers (every linear layer here, the
  vision tower's included, which gets no gradient).
- **On the same base and epochs, the Exu decoder with a cosine schedule is ahead
  of Unsloth.** On Qwen3.5-0.8B, 5 epochs, it reaches KL 0.085 ± 0.001 against
  Unsloth's 0.094 ± 0.003, with accuracy 0.794 against 0.777; NLL and Brier agree.
  That is about where Unsloth's 2B lands (0.082 ± 0.002); Unsloth's 4B stays ahead
  of everything here (0.073 ± 0.001). The lead is a small-base result: it does not
  survive the move to 2B (next point). The Exu arm's top-1 ECE is higher (0.164
  against 0.148), which "How to read this file" explains for soft targets.
- **At 2B the Exu decoder ties Unsloth, because it gains nothing from the larger
  base.** Exu on Qwen3.5-2B reaches KL 0.082 ± 0.002, accuracy 0.797 ± 0.005,
  against 0.085 ± 0.001 on the 0.8B: inside the seed spread. Unsloth gains 0.012
  of KL over the same step (0.094 to 0.082) and lands on exactly the same numbers
  as Exu on KL, Brier and NLL (equal to the third decimal); Exu's accuracy is
  0.007 higher, about one seed spread. Exu's 2B used the 0.8B
  recipe unchanged, lr included; whether another lr lets it use the larger base
  was not tried. The fitted temperatures sit between 1.00 and 1.10, so calibration
  is not what holds it back.
- **The schedule was the difference, not the learning rate.** With the lr held
  constant, the default, the same Exu decoder on Qwen3.5-0.8B lands at 0.102 ±
  0.004, behind Unsloth; the cosine decay with 10 warm-up steps takes it to 0.085.
  On Qwen3-0.6B the same change moves 0.090 ± 0.003 to 0.083 ± 0.003. A five times
  higher lr, closer to Unsloth's, hurts on both bases (0.153 and 0.187, one seed
  each): a full fine-tune does not want a LoRA learning rate. With the schedule, LoRA
  itself goes further (first point); Unsloth's one prefill per case was not tried
  in Exu. It is a decoder result: on
  ModernBERT-base `marker-cls` the same schedule gives 0.149 ± 0.025 against
  0.139 ± 0.012, no gain inside the seed spread.
- **Without the schedule, Exu ties at best.** At matched epochs and constant lr the
  Exu decoder on Qwen3-0.6B (KL 0.090) ties Unsloth on Qwen3.5-0.8B (0.094 ±
  0.003). An earlier read of these runs, with Unsloth at its 2-epoch default, made
  Exu look 0.08 of KL better; that was the epoch count. On this data 2 epochs
  undertrain the 0.8B (accuracy 0.685 against 0.777 at 5).
- **Hard-label calibration multiplies KL by two to six.** Unsloth's `calibrate`
  targets the chance that the top answer is right, as its docs say, and gets the
  lowest ECE among the trained arms (0.016 to 0.037). Read as a distribution against
  the soft gold, every hard-calibrated arm lands near `prior` on KL and NLL (KL 0.271
  to 0.423 against 0.327), ahead of it only on Brier. The same raw logits, scored at
  temperature 1, are the best numbers in the table. Which one is right depends on
  whether the probabilities are used as probabilities or as a ranking with a
  confidence cutoff.
- **The soft-gold temperature map rescues the zero-shot models and does almost
  nothing for trained ones.** On Clef it cuts KL by 25% (27B, 0.191 to 0.144) and
  23% (flash, 0.209 to 0.160) with no change in accuracy. On Unsloth arms trained
  on the same soft targets it moves KL by 0.001 to 0.002, inside the seed spread.
  It raises ECE everywhere, for the reason in "How to read this file".
- **The trained models beat the 27B zero-shot model on every proper score.** That
  is specialist against generalist on a synthetic benchmark whose gold comes from a
  teacher model; it says nothing about Clef on other data.
- **ModernBERT.** The base encoder with `marker-cls` is the cheapest arm that
  clears `prior` well (KL 0.139, about Unsloth 0.8B at 2 epochs); plain `marker` is
  worse, as in section 5. ModernBERT-large at the base's encoder lr, `2.5e-5`, stayed
  close to `prior`; at `1e-5` it learns, and ties the base (0.132 ± 0.017 against
  0.139 ± 0.012). Seed 17 alone said 0.113, and seeds 23 and 42 took it back: the
  large encoder buys nothing measurable here at three times the parameters.
  One seed at `5e-6` gave 0.128.
- Every trained arm and every Clef arm beats `uniform` and `prior` on KL, Brier and
  NLL, except the hard-calibrated Unsloth arms (above) and ModernBERT-large at
  `2.5e-5`, which beats `prior` but stays close to it.

**Cost.** Training time for 5 epochs, one seed: Exu decoder about 3 hours on an RTX
3060 (one sequence per option, no prefix shared across options); Unsloth 0.8B 29
minutes on the RTX 4060; Unsloth 2B 21 minutes and 4B 44 minutes on an RTX 3090;
Exu ModernBERT-base about 15 minutes and ModernBERT-large about 40 on a 3060; the Exu
decoder on Qwen3.5-0.8B about two hours on an A40, RTX A6000, RTX 4090 or RTX PRO
4500, and on Qwen3.5-2B about three and a half hours on an A40 or RTX A6000,
without the `causal_conv1d` and `flash-linear-attention` kernels; with LoRA and the
kernels, the Exu decoder on Qwen3.5-0.8B about two and a half hours on a 3060, about
twice as fast as the same run without the kernels, and on Qwen3.5-2B about five and a
half hours on a 3060. Different cards, so this is an order
of magnitude, not a ratio.

**Latency**, RTX 4060, one case of five questions per call, p50 and p95 over 100
test cases after 5 warm-up cases, model already loaded:

| Model | p50 ms | p95 ms |
| --- | --- | --- |
| Exu ModernBERT-base `marker-cls`, `exu-decide --batch-size 5` | 43 | 48 |
| Unsloth Qwen3.5-0.8B, `FastDecisionModel.predict`, one prefill | 73 | 83 |
| Exu decoder Qwen3-0.6B, `exu-decide --batch-size 5` | 369 | 405 |

Unsloth ran without `torch.compile` (it failed with a stride assertion on this
card) and without the `causal_conv1d` kernel, so 73 ms is an upper bound for it.
The Exu decoder builds one sequence per option, about 18 sequences per case here,
and re-reads the state in each; that, not the base size, is the 5x. Those three rows
predate the shared prefix. With it, the five questions of a case read their state
once and the 18 options run as short tails on its cache. On the published Qwen3.5-2B
model (seed 42 with LoRA), float32, on an RTX 3060 with the kernels, a case went from
1,024 ms to 439 ms at p50 (1,116 to 447 at p95), timed the same way. Over the 2,000
test questions no answer changed, the probabilities moved by at most 0.0006 and every
score in the table stayed the same to the fourth decimal. The Qwen3-0.6B row was not
timed again.

Training shares the prefix as well, running the layers one by one (prefixes, then the
tails on a cache of that layer, each step checkpointed). Measured on Qwen3.5-2B with
LoRA r16, `--grad-checkpointing`, one epoch over 320 training questions on one RTX
3060, the training loop alone (no loading, merge or evaluation):

| States | Recipe | Batch | Time | Peak |
| --- | --- | --- | --- | --- |
| this dataset's, about 90 tokens | K full sequences | 4 x 4 | 100 s | 6.69 GB |
| this dataset's, about 90 tokens | shared prefix, layer by layer | 4 x 4 | 74 s | 6.23 GB |
| the same plus a 2,000-token history | K full sequences | 4 x 4 | out of memory | |
| the same plus a 2,000-token history | K full sequences | 1 x 16 | 1,531 s | 9.06 GB |
| the same plus a 2,000-token history | shared prefix, layer by layer | 4 x 4 | 496 s | 9.51 GB |

The peaks include about 5.5 GB of weights. Validation NLL after the epoch agreed
within the spread of two runs of the same recipe (1.1583 and 1.1586 layer by layer,
1.1502 and 1.1578 with full sequences, on the short states). This is one epoch on a
subset, so it says what training costs, not what it reaches; no full run was trained
this way.

What this does not show. It is one synthetic dataset with teacher-model gold; the
higher-lr arms have one seed; the Exu decoder was not run at 4B, and of
the encoders only ModernBERT-base ran with the cosine schedule. Unsloth's recipe was not tuned
beyond the epoch count; Exu's was tuned in its schedule and in two learning rates,
which is more tuning than Unsloth got, and the choice was read off these test
numbers, with no held-out selection.

```bash
# The converter, the scorer, the Unsloth and Clef drivers and the aggregator are
# working files outside the repository. The Exu decoder arm runs:
uv run exu-train --mode baseline --backbone decoder --encoder Qwen/Qwen3-0.6B \
  --scorer marker --grad-checkpointing \
  --train $D --train-split train --validation $D --validation-split validation \
  --calibration $D --calibration-split calibration --calibrate \
  --test $D --test-split test --option-shuffle --epochs 5 --batch-size 4 \
  --grad-accum 4 --max-length 768 --seed 17 --output artifacts/td/qwen06-dec-s17
# The cosine arms are the decoder's default now; the constant arms add
# --lr-schedule constant --warmup-steps 0
```

## What has not been measured

- Feedback on a sampled answer from a real judge. Section 13 simulates the judge
  from the annotators; a model, a test or a user as judge, with its own bias, has
  not been run, and neither has a comparison at equal annotation cost.
- RLCD as a second stage after a shorter first stage, while the model still has
  room. Section 10's second stage started at the peak.
- More epochs, another sigma schedule, another learning rate.
- A decoder backbone beyond sections 10 and 14: a shorter first stage before RLCD,
  a full run trained on the shared prefix, LoRA beyond section 14, and a remedy for its plateau on balanced two-option questions.
- Whether four jobs sharing one GPU change a result. Runs are seeded and independent,
  and the `marker` baseline matches the one-job-per-GPU number, but no run was
  repeated both ways.
- The `noul` kind beyond BoolQ, and BoolQ beyond one question wording.
- Held-out task families for the package's own models. Section 12 holds out whole
  datasets for the 4B adapter, one seed; every other dataset above is a single
  question, or a few questions that all appear in training.
- The float16 path, which has no `GradScaler`, and any GPU older than Ampere.
- Whether the temperature map absorbs the bias of section 8's smoothing.
- Section 14 beyond one synthetic dataset: the Exu decoder at 4B, another lr for it at 2B,
  ModernBERT-large with the cosine schedule, LoRA in Exu at another rank or at 4B, and any tuning of Unsloth beyond
  the epoch count.
- A hard-label temperature in Exu, for callers who threshold on the top answer's
  confidence; section 14 shows what it costs on soft targets in Unsloth.

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
