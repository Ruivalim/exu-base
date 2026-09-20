# AGENTS.md

Instructions for any coding agent working in this repository. Written to be read
cold, by an agent that has never seen the project.

## What this is

Exu trains encoder-only models that answer a **typed question** about a **state**
with a probability distribution, in one forward pass, without generating text.
The training method it implements is RLCD, the name TypeSafe gives to the recipe
behind its System One models: the term is theirs, the toolkit is this project's.

- The question is written at request time: `choice` (pick one of N options),
  `score` (place the state on an ordinal rubric) or `noul` (boolean). All three
  are the same mechanism, a softmax over explicit options.
- The answer is a distribution over those options plus a separate `confidence`.
  Confidence is not the maximum probability: they are different scales and the
  runtime exposes both. Read `docs/evaluation.md` before putting a business
  threshold on either.
- Training maximizes a strictly proper scoring rule, so reporting honest
  probabilities is the optimum of the objective. `--mode baseline` optimizes the
  same score directly; `--mode rlcd` adds the perturbed-logit policy gradient.
  The baseline is the bar the policy mode has to beat on held-out ECE or NLL.
- Pre-alpha. The API, the checkpoint format and the results can change.

What this is not: an LLM wrapper, a chat model, or a classifier with a fixed
label set. Options are text the model reads, so a new task is a new question, not
a new output layer.

Prior art, and what came from where, is in the "Prior art" section of the README.
The short version: the typed-decision contract comes from TypeSafe's Jev, the RLCD
name comes from TypeSafe too, the open build recipe comes from Laya, and no source
code was copied from anyone.

## Layout

```
src/exu/          the package, all of it (there is no second source tree)
  types.py         DecisionQuestion / Option contract, the three primitives
  sequence.py      SequenceBuilder: one sequence per question, markers, token budget
  model.py         ExuModel: encoder + type embedding + 2 layers + scorer + act head
  scoring.py       log, spherical and ranked-probability scores; composite reward
  policy.py        perturbed-logit policy gradient, advantages, sigma schedule
  calibration.py   temperature fitting and the per-type/per-bucket map
  data.py          JSONL records, dataset, split and augmentation plumbing
  augmentation.py  option shuffling and friends
  metrics.py       NLL, Brier, ECE, ordinal metrics, the three trivial baselines
  evaluation.py    evaluation passes: per-kind, per-family, coverage, order robustness
  evaluate.py      the `exu-evaluate` CLI
  train.py         the `exu-train` CLI
  checkpoint.py    save and load, with validation before loading
  runtime.py       DecisionRuntime: load once, decide, batched
  devices.py       device and dtype selection
tests/             synthetic fixtures only, never real data
docs/              guide, algorithm, dataset format, checkpoints, evaluation
site/              static explainer, no framework, built into _site/
scripts/           smoke.sh, tiny encoder builder, site build
examples/          end_to_end.py, offline, runs in seconds on CPU
```

## Commands

`make` is the facade; prefer it over remembering raw flags. CI runs exactly
`make check`, so that is the contract.

```bash
make setup    # uv sync --extra dev
make lint     # ruff check + ruff format --check
make test     # pytest
make check    # lint + test + build, what CI runs
make smoke    # real train + calibrate + evaluate on the CPU fixtures, seconds
make site     # static explainer into _site/ (needs Node 20+)
```

Also available without the Makefile: `uv run pytest tests/test_sequence.py`,
`uv run python examples/end_to_end.py`, `uv run exu-train --help`.

Before reporting work as done: `make check` green, and for anything touching the
pipeline, `make smoke` too. A change to sequence building, scoring, policy or
checkpoints should be exercised end to end, not only unit tested.

## Invariants

Changing one of these is an architecture decision, not a refactor. If a change
touches them, say so explicitly instead of folding it into a small diff.

1. **One sequence per question.** N questions about the same state are a batch of
   N sequences; the state is re-encoded N times. That is why latency grows with
   the number of questions.
2. **The marker is what the scorer reads.** It is the mask token the builder
   inserts before each option, and the option's logit comes from the hidden state
   at that position. Nothing else feeds the scorer.
3. **Literal mask strings are stripped** from instruction, option and state text
   before encoding. Without that, untrusted input can inject fake markers.
4. **The token budget fails loudly.** Options cede tokens before the instruction
   does, the instruction keeps a floor, and an option count that cannot fit
   raises `too many options for configured header_budget`. Do not replace that
   with a silent shrink: answering a question whose option texts have become
   indistinguishable is worse than refusing to build it.
5. **A target is a distribution**: non-negative, sums to one, zero on padded
   options. `scoring.validate_distributions` enforces it. Soft targets from
   several annotators or a teacher model are first-class.
6. **The reward stays strictly proper.** Any term that rewards a correct argmax
   breaks the property the whole project exists for.
7. **Temperatures are fitted on held-out data**, never on training data, and a
   fitted value sitting on a bound is a signal to look at, not a result to ship.
8. **A metric without its trivial baselines says nothing.** Uniform, the
   per-question prior and the majority class are reported next to every number.

## Documentation

Docs are part of the product here, and this repository has already been burned by
docs that described a different implementation. When behaviour changes, the doc
that states it changes in the same commit.

| Claim | Lives in |
| --- | --- |
| What the model is, install, serve | `README.md` |
| How to train one, start to finish, and what breaks | `docs/usage-guide.md` |
| Field-by-field data contract and split rules | `docs/dataset-format.md` |
| Checkpoint files, keys, key precedence | `docs/checkpoints.md` |
| Every metric, every baseline, latency | `docs/evaluation.md` |
| Reward and policy-gradient math | `docs/algorithm.md` |
| The build walkthrough, with the traps | `docs/exu-guide.md` |

The two walkthroughs are English/Portuguese pairs
(`exu-guide.{md,pt-BR.md}` and `usage-guide.{md,pt-BR.md}`). Keep them in step:
same sections, same numbers, same code blocks. ruff formats python blocks inside
markdown, so a code block over the line limit fails `make lint` in either file.

Numbers in the docs attributed to Laya are Laya's measurements, not runs made
here. Never present a result measured on the fixtures as evidence of quality:
the fixtures are plumbing, and the README says so. Keep the walkthrough honest
about which metrics are implemented and which are recommendations.

## Tests

New behaviour is born with its test. Cover the failure mode, not only the happy
path: a boundary token budget, a question with too many options, an input
containing the literal mask token, a target that does not sum to one, malformed
JSONL, a checkpoint missing a key.

Never commit a dataset, a checkpoint or a `.safetensors` file. `artifacts/`,
`runs/`, `_site/`, `dist/` and `graft/` are gitignored; keep it that way. The
fixtures are synthetic and small on purpose.

Do not weaken a test to make it pass. If the test itself is wrong, say why.

## Tools

Optional, but the repository is wired for them.

- **graft** (`graft` CLI, or the MCP server declared in `.mcp.json` and
  `opencode.json`): a repo context graph of small linked markdown nodes carrying
  exact file:line spans. It usually answers "where is X" or "how does X work" in
  one call, cheaper and more accurate than grepping around. The graph lives in
  `graft/` and is **gitignored**, so a fresh clone has none: run `graft build`
  first (deterministic, no API key, $0). Re-run it after a large change;
  `graft check` fails when the graph is stale relative to the code. Usage
  details are in the block at the end of this file.
- **qlty** (config in `.qlty/qlty.toml`): linters plus duplication and complexity
  smells, if you have the CLI. `qlty check` and `qlty smells` are a useful second
  opinion next to ruff, not a replacement for `make check`.
- **ruff** and **pytest** are the mandatory ones, through `make lint` and
  `make test`. Line length is 100 with E501 off; formatting is enforced.
- **Node 20+** is needed only for the site (`make site`, `node --check site/app.js`).

## Conventions

- Python 3.11+. `uv` manages the interpreter; `.python-version` is for local use.
- Value types are frozen dataclasses with `slots=True`. The tokenizer surface is
  a `Protocol`, so tests can use a fake.
- Every public name is re-exported from `src/exu/__init__.py` with an `__all__`.
  A new user-facing name goes there and, if worth mentioning, in the README.
- No new runtime dependency without a reason written down in the pull request.
- Docs and the README are English. The Makefile help and
  `docs/exu-guide.pt-BR.md` are Portuguese on purpose; leave them.
- Commits: one intent per commit, Conventional Commits style.

<!-- graft:start -->
## Graft — repo context graph

This repo is indexed in `graft/`: small linked markdown nodes that explain each
system and carry exact file:line spans, kept in sync with the code through git.

For ANY task here — understanding how something works, finding where code lives,
or scoping a change — get context from the graph before grepping or opening
source files. Re-ask freely (it's cheap) and reuse literal identifiers you
already have (symbol, error string, file name) as the query. New to this repo?
Run `graft map` first — a token-budgeted orientation (dir clusters, hubs,
hotspots), no LLM, no key.

- Run `graft ask "<your question>" --source` → ranked nodes with the relevant
  code spans inlined (each hit's ≤8-line crux by default; `--full` for whole
  definitions when the crux isn't enough). Match the tool to the task shape:
  for understanding or editing, the top node IS the answer — cite its
  `covers:` file:line spans and edit straight from `--source`. For
  exhaustive tasks ("every occurrence / every caller of this pattern"), ranked
  results are top-N, not complete — run `graft grep "<literal>"` instead
  (exhaustive over indexed files, grouped by enclosing symbol), falling back
  to raw `grep -rn` only for unindexed files.
- `graft skeleton <file>` → every definition's signature + span, ~10× cheaper
  than reading the file; use it to skim an API surface.
- `graft callers <symbol>` gives precomputed, exact edges — who calls this.
  Add `--direction out` for what it calls, or `--depth N` to walk
  transitively for the full blast radius. For structural questions, skip
  ranking and use this directly.
- Or browse: `graft/INDEX.md` lists every node; follow the links.
- Monorepos and folders of multiple repos rank fairly across sub-projects —
  hits carry `[scope/]` labels naming which one they're from. Narrow with
  `graft ask "<task>" --in <scope>/` once you know where you're working.

If a returned span is truncated ("+N more lines"), open the file at that exact
range before finalizing. Only open source files when a node genuinely lacks a
needed detail, and then at the exact file:line the node points to — never
re-read whole files.

After big code changes, refresh the graph with `graft build` (deterministic,
no API key, $0).
<!-- graft:end -->
