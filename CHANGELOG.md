# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-09-19

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
- Offline inference runtime with batched decisions and two documented confidence
  scales.
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

[Unreleased]: https://github.com/ruivalim/exu-base/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ruivalim/exu-base/releases/tag/v0.1.0
