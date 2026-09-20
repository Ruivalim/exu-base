# Contributing

Thanks for considering a contribution to RLCD.

## Before you open a change

1. Open an issue first for anything large: a new encoder, an architecture change,
   a training-method change, or a licensing question.
2. Keep changes small and single-purpose.
3. Do not commit personal data, credentials, uncertain-license content, or large
   checkpoints and datasets.
4. Preserve provenance, license and split for every dataset you touch.

## Local environment

```bash
make setup   # uv sync --extra dev
make test
make lint
make check   # lint + test + build, exactly what CI runs
```

Run `make` for the full target list.

## Code and tests

- Every new behavior comes with a test. New code is born with its test.
- Cover failure modes, not just the happy path: concurrency, invalid input,
  boundary values and the silent assumption nobody wrote down.
- When you change validation, keep the invalid-input test that guards it.
- Run tests, lint and build before opening a pull request.
- Prefer plain language in docstrings and comments. Explain why, not what.

## Evidence and claims

- Never present a result on synthetic data as evidence of general quality.
- Always report the trivial baselines (random, per-question prior, majority
  class) next to any accuracy or calibration number.
- Any calibration claim must come from a held-out split, never from training data.
- If a metric does not reproduce, say so. "It should work" is a guess.

## Data and checkpoints

Datasets and checkpoints carry their own licenses. A data contribution must
report source, license, language variant, transformation, annotation method, PII
review and split strategy. Synthetic data must also record the teacher model,
prompts and human review.

Do not upload artifacts to PyPI or Hugging Face without explicit authorization
from the maintainer.

## Pull requests

Describe the problem, the solution, the tests you ran and the limitations. Keep
the diff focused. If part of the work is unfinished, say which part and why.
