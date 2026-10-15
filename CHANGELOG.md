# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- `fit_temperature` computes its NLL from `masked_log_softmax` instead of the log
  of a clamped softmax. On a confident miss the softmax underflowed to the clamp,
  the NLL went flat in `T` and the fit stayed at 1: logits `[200, 0]` against the
  second option read 87 instead of 200, with a zero gradient. No run in
  `BENCHMARKS.md` came near it, the largest logit gap in their predictions is 17.9.
- `merge_lora` computes and stores the merged weight in float32 inside autocast or
  with a float16 default dtype. Before, the adapter product ran in the autocast
  dtype and could overflow to `inf`. `exu-train` was not affected: it merges on the
  CPU, outside autocast.

## [0.4.0] - 2026-10-13

### Added

- An optional `context` per record (string, object or list): what the answer
  depends on besides the state, such as a policy. Both builders read it before the
  state, with its own budget, `SequenceConfig.context_budget` and
  `exu-train --context-budget` (256 tokens), and refuse a context over it instead
  of cutting it; the state is cut first. Without a context every sequence is the
  same as before. `DecisionRuntime.decide`, `decide_all` and `decide_many` take a
  context, and `exu-decide` reads it from a record or from `--context` and
  `--context-file`.
- `exu-train` builds every record once before loading the model, so a question or
  a context over its budget is named by its record id instead of failing mid-epoch.
- `DecisionRuntime.decide_all(state, questions)`: several questions about one state
  in one forward pass.
- `exu-decide --input` takes a record with `questions`, a list of questions about
  its one state, each with an optional `id`; answers are named `<record id>/<question
  id or position>`. The questions of a record share a batch whenever the batch size
  allows, so a decoder reads their state once.

### Breaking

- `DecisionBuilder.build` takes an optional `context`; a builder written against
  the protocol needs the parameter.
- `EncodedOptions` has a required `prefix_length`, the leading tokens its sequences
  share, and refuses sequences that do not share it. `DecoderSequenceBuilder.tensors`
  emits `prefix_lengths`.

### Changed

- Without gradients the decoder runs each distinct prefix once and every option as a
  short tail on its cache, so the questions about one state read it once. Evaluation,
  calibration, `exu-decide` and `DecisionRuntime` get it with no flag. On Qwen3.5-2B a
  case of five questions went from 1,024 ms to 439 ms on an RTX 3060 with the same
  answers (probabilities within 0.0006).
- A Qwen3 or Qwen3.5 decoder trains on the shared prefix too, running its layers one
  by one, each on the prefixes and then on the tails over a cache of that layer;
  `--grad-checkpointing` checkpoints each such step. The gradients are those of the
  K full sequences. With LoRA on Qwen3.5-2B on an RTX 3060, 320 questions trained in
  74 s against 100 s, and with 2,000-token states in 496 s at batch 4, where the full
  sequences ran out of memory at batch 4 and took 1,531 s at batch 1. Other decoder
  families still train on the K full sequences.

## [0.3.0] - 2026-10-11

### Breaking

- The decoder backbone trains with a cosine learning-rate schedule and 10 warm-up
  updates by default, shortened when a run has fewer. A decoder run that relied on
  the old constant rate needs `--lr-schedule constant --warmup-steps 0`. The encoder
  default is unchanged.

### Added

- `exu-train --lr-schedule {constant,cosine}` and `--warmup-steps N`.
- `exu-train --lora-rank R` and `--lora-alpha A`: low-rank adapters on every linear
  layer of a frozen backbone, merged into the weights after training and before
  calibration, so the checkpoint format is unchanged. On a GPU the frozen weights
  are kept in the autocast dtype, which fits Qwen3.5-2B on a 12 GB card. `exu.lora`
  exposes `LoRALinear`, `apply_lora` and `merge_lora`.

### Fixed

- The decoder backbone loads vision-language bases such as Qwen3.5, whose text sizes
  live in `text_config`.

## [0.2.0] - 2026-10-04

### Breaking

- `TrainingBatch` holds the model's inputs as one `inputs` mapping, with
  `option_mask` as a property, instead of one field per encoder tensor. Code that
  built or read a batch by field breaks.
- A pretrained backbone loads in float32 whatever dtype it was published in (see
  Fixed). A run that relied on 0.1.0 keeping the published dtype now uses more
  memory and gives different numbers.
- Checkpoint format 2. Format 1 checkpoints still load here, but a checkpoint
  saved by this version does not load in 0.1.0.

### Added

- `--backbone decoder`, experimental: a causal backbone (default
  `Qwen/Qwen3-0.6B`) read at the last token of one sequence per option, with the
  layout and head of the probe that produced the decoder numbers in
  `BENCHMARKS.md`. `DecoderDecisionModel`, `DecoderSequenceBuilder` and
  `EncodedOptions` are public. The reward, calibration, evaluation, runtime and
  CLIs are shared with the encoder.
- `builder_for`, `build_model` and `model_from_pretrained` pick the builder and the
  model from a configuration; `DecisionBuilder` is the protocol both builders meet.
- `--grad-checkpointing` in `exu-train`, for either backbone.
- `training.json` records the architecture and the backbone and head learning
  rates.
- `exu.bandit`, a probe without a CLI mode: a proper score from a verdict on one
  sampled answer, with no target read. `verdict_loss` (the default estimator,
  binary log score of the sampled answer's probability against its verdict),
  `estimate_target` (importance-weighted or doubly robust), `bandit_loss`,
  `simulated_bandit_loss` and `BanditConfig`. Math in `docs/algorithm.md`, numbers
  in `BENCHMARKS.md` section 13. No data format carries real verdicts yet.

### Changed

- Checkpoint format 2: `exu.json` carries an `architecture` block, and the loader
  builds the model it names. Format 1 checkpoints still load, as the encoder with
  markers. Weights of one architecture are refused under the other, with a message
  saying which they look like.
- `--encoder` and `--encoder-lr` default per backbone, and `--header-budget`
  defaults to `--max-length` when that is below 192 instead of raising.
- `SequenceBuilder.tensors` builds the batch's `inputs` mapping; the runtime and
  the evaluation passes no longer build tensors by hand.
- The README recommends `--mode baseline` and its training example uses it.
  `--mode rlcd` stays, as the method under evaluation: it lost all fourteen
  comparisons in `BENCHMARKS.md`, teacher-model targets included (section 11).

### Fixed

- A pretrained backbone loads in float32 whatever dtype it was published in.
  transformers 5 keeps a checkpoint's own dtype, so a backbone published in
  bfloat16, such as Qwen3, trained with bfloat16 master weights, where most updates
  at a `1e-5` learning rate round away. Precision in the forward pass is still
  autocast's.

## [0.1.0] - 2026-09-22

First public release, under the name Exu. It is a toolkit: the training method it
implements is RLCD, and that name belongs to TypeSafe.

### Added

- Toolkit for calibrated decision models, distributed as `exu-base` and imported
  as `exu`. Implements RLCD, *Reinforcement Learning for Calibrated Decisions*,
  the training method TypeSafe introduced for its System One models, next to a
  direct baseline that optimizes the same score without sampling.
- Typed-decision contract: `choice`, `score` and `noul` primitives with options
  written at request time.
- Token-budgeted `SequenceBuilder` with one mask marker per option, explicit
  header and option limits, right-truncation of the state, and removal of
  literal mask strings from all untrusted text.
- `ExuModel`: bidirectional encoder, question-type embedding, two extra
  transformer layers, marker scorer and an act-or-escalate head.
- Strictly proper scoring rules: log score computed from `log_softmax` with no
  floor, spherical score and ranked probability score, plus the composite reward.
- Direct baseline training and RLCD training: perturbed-logit policy gradient
  with masked zero-sum noise, `G` candidates per question, GRPO-style advantage
  and a linearly annealed sigma, with an optional cross-entropy term.
- Temperature calibration: per-type and per-(type, option count) buckets, fitted
  with LBFGS on held-out data, with min-sample fallback and bound detection.
- Evaluation: NLL, Brier, accuracy, soft accuracy, ECE, ordinal RPS and MAE,
  per-kind and per-family breakdowns, uniform/prior/majority baselines, selective
  coverage, order robustness and latency percentiles.
- Portable checkpoint carrying model config, sequence config, temperature map,
  weights and tokenizer, with validated loading.
- `--scorer marker-cls`: an optional term that crosses each option's text with the
  `[CLS]` state. The shared marker scorer starts from a symmetric saddle and needs
  something to tell the options apart: with balanced training labels and no lexical
  overlap it never leaves the uniform guess (balanced MultiNLI 0.32 accuracy,
  balanced BoolQ at exactly `log 2`). `marker-cls` breaks the symmetry by
  construction (0.75 and 0.66 there). Stored in the checkpoint, default unchanged,
  older checkpoints load as `marker`.
- `training.json` reports `confident_miss_rate` per epoch, and
  `candidate_confident_miss_rate` in `rlcd` mode: how often the model gave less than
  `1e-4` to something the target claims. It is the threshold the log floor used to
  have, so how often that floor acted is now a measurement.
- Offline inference runtime with batched decisions and two documented confidence
  scales.
- `scripts/runpod_gpu.py`: rent one GPU pod on RunPod for a training run, send
  the tree and the data, run a job with a time limit and an automatic stop, bring
  the reports back, and give the pod back. Standard library only; the API key is
  never written or printed. Every guard in it was earned on a real pod.
- `exu-decide`: the runtime from the command line. One question written with
  flags, or a JSONL batch from a file or stdin, one JSON object per question. The
  input is validated whole before anything is answered. `--top-only` keeps the
  winning option and its probability, `--metrics` times every batch and prints a
  summary to stderr.
- Static explainer site with interactive scoring-rule and policy-gradient demos.

### Fixed

Found by an external review before this first release, and fixed here:

- The order-robustness pass permuted `score` questions too. Training never shuffles
  an ordinal scale, so a correct model was reported as unstable: 0.23 on a
  five-level dataset, which is chance. Ordinal questions are now left out and
  counted in `skipped_ordinal`, and `stability` is `null` when nothing is left.
- The `prior` and `majority` baselines were fitted on the labels they were scored
  against, so a question that occurred once got its own target as its prior: NLL
  0 and accuracy 1, presented as the bar to clear. They are now fitted on
  reference labels, the `train` split of `--data` by default or `--reference`,
  with `(count + 1/K) / (n + 1)` smoothing that doubles as the fallback for an
  unseen question. The report says where the prior came from and how many rows it
  covered, and returns `null` instead of falling back to the evaluation labels.
  `majority` reports accuracy only, and breaks ties on option identity. The old
  quantity is kept as `prior_in_sample`, a diagnostic. A `choice` question now
  matches in any option order. The report key `random` is now `uniform`.
- The log score was clamped at `1e-4`, in the reward and in the auxiliary
  cross-entropy. That made the reward improper for any component below about
  `2.7e-4`, where reporting zero outscored the truth, and it left every confidently
  wrong row with a gradient near zero in both training modes. The log term now
  comes from `log_softmax` on all three paths, `log_floor` is gone from the API,
  the scoring functions take log-probabilities, and `training.json` records the
  reward definition in both modes. Numbers from earlier training runs have to be
  rerun. The site's scoring mirror changed with it.
- The header budget reserved each option's ceiling instead of its real length, so
  any question with four or more options pinned the instruction to its 8-token
  floor and left most of the header unused. The question was silently truncated.
- `--option-shuffle` permuted ordinal questions too, which broke the ranked
  probability score: a far miss could score better than a near one.
- A `score` question raised `RuntimeError` on CUDA, because the level tensor was
  built on the CPU while the probabilities were on the device.
- Temperatures were fitted per exact option count but stored per bucket, so in a
  bucket holding several counts the last one overwrote the rest, and
  `min_samples` measured the wrong population.
- Literal mask text with an interleaved occurrence (`[MA[MASK]SK]`) left a real
  mask token behind, which the scorer then read as an extra option marker.
- One record without a `family` erased the entire per-family report.
- The last accumulation group of an epoch was scaled by the full `grad_accum`,
  and the sigma schedule stopped one step short of `sigma_end`.
- `--log-every 0`, documented as "0 disables step logs", was rejected by the
  argument parser.

### Changed

- The source distribution excludes `data/`, `artifacts/` and root-level JSONL
  files, so a local `uv publish` can never upload a dataset: PyPI releases cannot
  be undone.
- The release workflow refuses a tag that does not match the project version.
- The explainer site mirrors the fixed budget allocator, anneals sigma linearly
  like the library, and its dim text clears WCAG AA contrast.

[Unreleased]: https://github.com/ruivalim/exu-base/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/ruivalim/exu-base/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/ruivalim/exu-base/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/ruivalim/exu-base/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/ruivalim/exu-base/releases/tag/v0.1.0
