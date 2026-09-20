# Exu

[![CI](https://github.com/ruivalim/exu-base/actions/workflows/ci.yml/badge.svg)](https://github.com/ruivalim/exu-base/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Train encoder-only decision models that answer with **probability distributions**
instead of generated text. You give a state and a question with explicit options;
the model returns a distribution over those options in one forward pass.

**Exu is a toolkit, not a method.** The training method it implements is RLCD,
*Reinforcement Learning for Calibrated Decisions*, the name
[TypeSafe AI](https://typesafe.ai/) gives to the recipe behind Jev, its System One
model. The reward is a strictly proper scoring rule, so the only way to increase
it is to report honest probabilities. Calibration is not coaxed out of the model
with a prompt, it is the optimum of the objective.

The name honours Exu, the Orixá of the crossroads, of movement and of
communication, who in Afro-Brazilian religions opens the paths and governs the
choice made at every crossing. A model that answers with a distribution over
explicit options is a crossroads by construction.

This is the agnostic base. It is not tied to a language, a domain, or an encoder.
Swap the encoder, write your questions, and train.

> **Pre-alpha.** The API, the checkpoint format and the results can change.

## Why

A reflex decision (route a ticket, score urgency, flag fraud) may need
probabilities, but it does not need text generation. Generating a token takes
hundreds of milliseconds, needs a parser, and the "confidence" a chat model
writes is just more text. Here the output is already a distribution, so there is
nothing to parse and nothing to hallucinate.

Three primitives cover most decisions, and all three are the same mechanism:

| Primitive | Question | Output |
| --- | --- | --- |
| `choice` | pick one of N explicit options | distribution over options |
| `score` | place the state on an ordinal rubric | distribution over levels |
| `noul` | boolean question | distribution over No and Yes |

The options are written at request time and read by the model as text, so a new
task is a new question, not a new output layer. No retraining to add a label.

## Install

```bash
uv sync --extra dev
```

## The contract

```python
from exu import DecisionQuestion, Option

route = DecisionQuestion.choice(
    "Where should this ticket go?",
    [
        Option("billing", "payment, invoice or refund"),
        Option("support", "access or outage"),
    ],
)

risk = DecisionQuestion.score(
    "How risky is this request?",
    ["no risk", "review", "block"],
)

fraud = DecisionQuestion.noul("Is this message attempting fraud?")
```

## Train

```bash
exu-train \
  --mode rlcd \
  --train data/train.jsonl --train-split train \
  --validation data/train.jsonl --validation-split validation \
  --test data/train.jsonl --test-split test \
  --calibration data/train.jsonl --calibration-split calibration \
  --output artifacts/my-model \
  --encoder google-bert/bert-base-multilingual-cased \
  --epochs 4 --batch-size 8 --option-shuffle --calibrate
```

`--mode rlcd` uses the perturbed-logit policy. `--mode baseline` optimizes the
same strictly proper score directly, with no sampling. Train the baseline first:
it is the bar the RLCD mode has to beat on held-out ECE or NLL.

The step-by-step version, from writing the records to reading the report and the
failure modes, is [docs/usage-guide.md](docs/usage-guide.md).

## Evaluate

```bash
exu-evaluate \
  --checkpoint artifacts/my-model \
  --data data/train.jsonl --split test \
  --order-permutations 4 --latency
```

Prints NLL, Brier, accuracy, ECE, ordinal scores, per-kind and per-family
breakdowns, selective coverage, the three trivial baselines, order robustness
and latency.

## Serve

```python
from exu import DecisionRuntime

runtime = DecisionRuntime.load("artifacts/my-model")
decision = runtime.decide("I was charged twice for the same invoice.", route)
print(decision.label, decision.confidence)  # billing 0.87
```

## Data

One JSONL record per decision. The target is a distribution, not a label, so soft
targets from several annotators or a teacher model are first-class.

```json
{
  "id": "t-1",
  "state": {"text": "I was charged twice for the same invoice."},
  "question": {
    "kind": "choice",
    "instruction": "Where should this ticket go?",
    "options": [
      {"name": "billing", "description": "payment, invoice or refund"},
      {"name": "support", "description": "access or outage"}
    ]
  },
  "target": [1.0, 0.0],
  "split": "train",
  "family": "routing",
  "language": "en"
}
```

See [docs/dataset-format.md](docs/dataset-format.md) for the full field list and
the rules that keep held-out honest.

## How it works

`state + typed question -> one sequence per question -> bidirectional encoder ->
question-type embedding and two extra transformer layers -> read the hidden state
at each option's marker -> softmax over options`. Training samples Gaussian
perturbations of the logits and rewards each candidate distribution with a
strictly proper scoring rule. Then a temperature map, fitted on held-out data,
brings confidence in line with accuracy.

The full walkthrough is in [docs/exu-guide.md](docs/exu-guide.md), and
[docs/algorithm.md](docs/algorithm.md) has the reward and policy math.

## Development

```bash
make setup   # install dependencies
make test    # pytest
make lint    # ruff
make check   # lint + test + build, exactly what CI runs
make site    # build the static explainer into _site/
```

Run `make` for the full list.

## Status

Working: typed-decision contract, token-budgeted sequence builder with marker
injection defense, direct and RLCD training, temperature calibration,
calibration-aware evaluation, portable checkpoint, offline inference runtime.

Not done: a published checkpoint, a human-annotated calibration set, learned
act-or-escalate training (the cost-based baseline is in place), and a multi-turn
prefix objective.

Small synthetic fixtures are for plumbing only. They are not evidence of quality.

## Prior art

Two projects are the inspiration for this one, and neither contributed code.

[TypeSafe's Jev](https://typesafe.ai/) came first and is the bigger influence:
the typed-decision contract of `choice`, `score` and `noul`, answers that carry a
distribution plus a separate confidence, the idea of gating action on that
confidence, and the RLCD name for the training method. The "System One" framing
is theirs too.

[Laya](https://github.com/NandhaKishorM/laya) (Apache-2.0, by Nandakishor) is the
open build recipe this repository follows: one sequence per question with a mask
marker per option, an explicit header and option token budget, the
log + spherical + ranked-probability reward, and one temperature per question
type and option-count bucket. The benchmark and latency numbers quoted in the
docs come from Laya's measurements, not from runs made here.

This is an independent implementation. No source code was copied from either
project.

One more reference, in a different setting. Nandakishor M, *Confidence-Aware
Routing for Large Language Model Reliability Enhancement* ([arXiv:2510.01237](https://arxiv.org/abs/2510.01237),
2025) is by the same author as Laya. It estimates confidence before generation
and routes a query across four pathways: local generation, retrieval, a larger
model, or human review. That is the routing idea the act-or-escalate gate applies
to a single encoder here. It does not describe this design, and nothing in this
repository implements it.

## License

Code under [MIT](LICENSE). Datasets and checkpoints carry their own licenses.
Before publishing an artifact, confirm the encoder, tokenizer, data and resulting
weights are compatible.
