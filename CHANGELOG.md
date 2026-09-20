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
- Strictly proper scoring rules: log score with a floor, spherical score and
  ranked probability score, plus the composite reward.
- Direct baseline training and RLCD training: perturbed-logit policy gradient
  with masked zero-sum noise, `G` candidates per question, GRPO-style advantage
  and a linearly annealed sigma, with an optional cross-entropy term.
- Temperature calibration: per-type and per-(type, option count) buckets, fitted
  with LBFGS on held-out data, with min-sample fallback and bound detection.
- Evaluation: NLL, Brier, accuracy, soft accuracy, ECE, ordinal RPS and MAE,
  per-kind and per-family breakdowns, random/prior/majority baselines, selective
  coverage, order robustness and latency percentiles.
- Portable checkpoint carrying model config, sequence config, temperature map,
  weights and tokenizer, with validated loading.
- Offline inference runtime with batched decisions and two documented confidence
  scales.
- Static explainer site with interactive scoring-rule and policy-gradient demos.

### Fixed

Found by an external review before this first release, and fixed here:

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
