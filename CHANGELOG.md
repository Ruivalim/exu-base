# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-09-19

### Added

- Typed-decision contract: `choice`, `score` and `noul` primitives with options
  written at request time.
- Token-budgeted `SequenceBuilder` with one mask marker per option, explicit
  header and option limits, right-truncation of the state, and removal of
  literal mask strings from all untrusted text.
- `RLCDModel`: bidirectional encoder, question-type embedding, two extra
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

[Unreleased]: https://github.com/ruivalim/rlcd-base/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ruivalim/rlcd-base/releases/tag/v0.1.0
