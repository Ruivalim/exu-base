#!/usr/bin/env bash
# End-to-end smoke run: train a tiny model on the fixtures, then evaluate it.
# Uses a real tokenizer, a real one-layer encoder and the real RLCD policy, so a
# broken pipeline fails here instead of in a user's first training run.
set -euo pipefail

FIXTURES="${FIXTURES:-tests/fixtures/smoke.jsonl}"
ARTIFACTS="${ARTIFACTS:-artifacts}"
OUT="$ARTIFACTS/smoke"
ENCODER_DIR="$ARTIFACTS/smoke-encoder"

rm -rf "$OUT" "$ENCODER_DIR"
uv run python scripts/make_tiny_encoder.py "$ENCODER_DIR"

uv run exu-train \
  --mode rlcd \
  --train "$FIXTURES" --train-split train \
  --validation "$FIXTURES" --validation-split validation \
  --test "$FIXTURES" --test-split test \
  --calibration "$FIXTURES" --calibration-split calibration \
  --output "$OUT" \
  --encoder "$ENCODER_DIR" \
  --epochs 3 --batch-size 4 --option-shuffle --calibrate \
  --calibration-min-samples 2 --device cpu

uv run exu-evaluate \
  --checkpoint "$OUT" \
  --data "$FIXTURES" --split test \
  --order-permutations 2 --device cpu
