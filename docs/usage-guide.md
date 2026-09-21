# Using this repository to train a model

This is the procedural guide: you have a task and some labeled data, and you want
a trained, calibrated checkpoint at the end. It is deliberately short on theory.

- For *why* the design is what it is, and every trap worth knowing, read
  [exu-guide.md](exu-guide.md).
- For the field-by-field data contract, the checkpoint layout and every metric,
  follow the links in each step below. Nothing here repeats them.

Every command in this file was run in this repository before being written down.

## 0. Check the environment first

```bash
make setup    # uv sync --extra dev
make test     # 113 tests, a few seconds
make smoke    # train + calibrate + evaluate a tiny model on CPU
```

`make smoke` is the important one: it drives the real pipeline, with a real
tokenizer, a real encoder and the real policy, on `tests/fixtures/smoke.jsonl`.
If it passes, the machine can train. Seconds on CPU, works without a GPU.

It writes to `artifacts/`, which is gitignored. Nothing else in this guide writes
outside the directory you pass to `--output`.

## 1. Write the data

One JSONL record per decision. The target is a distribution, not a label, so soft
targets from several annotators or a teacher model are first-class.

```json
{"id": "t-1", "state": "I was charged twice for the same invoice.",
 "question": {"kind": "choice", "instruction": "Where should this ticket go?",
   "options": [{"name": "billing", "description": "payment, invoice or refund"},
               {"name": "support", "description": "access or outage"}]},
 "target": [1.0, 0.0], "split": "train", "family": "routing", "language": "en"}
```

Rules that matter, with the full list in [dataset-format.md](dataset-format.md):

- `target` must sum to one, cannot be negative, and must be one value per option.
  Training refuses anything else, naming the line in the file and the field that
  is wrong.
- `split` is what `--train-split`, `--validation-split` and `--calibration-split`
  select. Put several splits in one file if you like; the flags do the filtering.
- `family` is the task family. **Split held-out by family, not by example**, or
  your zero-shot number is a lie. This is the single most common way to fool
  yourself with this architecture.
- Write the option `description` by hand. It is what the model actually reads when
  deciding, and it is usually where the accuracy comes from.

To convert an existing dataset, build `TrainingExample` objects straight from
records, or with `DecisionQuestion`:

```python
from exu import DecisionQuestion, Option, TrainingExample, write_jsonl

question = DecisionQuestion.choice(
    "Where should this ticket go?",
    [Option("billing", "payment, invoice or refund"), Option("support", "access or outage")],
)
examples = [
    TrainingExample.from_record(
        {
            "id": f"t-{index}",
            "state": text,
            "question": {
                "kind": question.kind.value,
                "instruction": question.instruction,
                "options": [
                    {"name": option.name, "description": option.description}
                    for option in question.options
                ],
            },
            "target": [1.0, 0.0],
            "split": "train",
            "family": "routing",
        }
    )
    for index, text in enumerate(texts)
]
write_jsonl("data/tickets.jsonl", examples)
```

`TrainingExample.to_record()` gives the same dict back, which is handy when you
want to rewrite a file with a different split assignment.

## 2. Choose the encoder

`--encoder` takes a Hugging Face id or a local folder. The default is
`google-bert/bert-base-multilingual-cased`: a neutral starting point, not a
recommendation.

The encoder must be bidirectional and its tokenizer must have a mask token. The
builder raises `tokenizer must provide mask_token_id` otherwise, which rules out
decoder-only models and Electra-style encoders up front.

Measure before you commit. Tokenizer fertility on your own text decides how much
state fits in the budget, and a checkpoint that is excellent in one language can
collapse in another. See phase 2 of [exu-guide.md](exu-guide.md).

## 3. Train the direct baseline

The bar. Nothing else counts until this number exists.

```bash
uv run exu-train \
  --mode baseline \
  --train data/tickets.jsonl --train-split train \
  --validation data/tickets.jsonl --validation-split validation \
  --calibration data/tickets.jsonl --calibration-split validation \
  --output artifacts/baseline \
  --encoder google-bert/bert-base-multilingual-cased \
  --epochs 4 --batch-size 8 --option-shuffle --calibrate
```

- `--output` must not exist. A partial failure removes the folder rather than
  leaving half a checkpoint.
- `--calibration` should be held-out. Fitting temperatures on training data is the
  trap that makes calibration look better than it is.
- `--option-shuffle` permutes a question's options every pass. Without it the
  model learns position instead of criteria.
- `--scorer marker-cls` is for training data with **balanced labels and no lexical
  overlap** between the options and the state. There the default `marker` scorer
  never leaves the uniform guess (validation NLL stuck at exactly `log K`): balanced
  MultiNLI stayed at 0.32 accuracy where `marker-cls` reached 0.75. If the first
  epoch ends at `log K`, this is the flag to try. The choice is stored in the
  checkpoint, so inference needs no flag.
- `--device` defaults to `auto`: CUDA, then MPS, then CPU.

Useful knobs, with their defaults: `--epochs 4`, `--batch-size 8`,
`--grad-accum 1`, `--seed 17`, `--encoder-lr 2.5e-5`, `--head-lr 1e-4`,
`--weight-decay 0.01`, `--max-grad-norm 1.0`, `--max-length 512`,
`--header-budget 192`, `--calibration-min-samples 64`, `--log-every 0`.

## 4. Train the policy

Same pipeline, `--mode rlcd` instead of `baseline`, and the policy knobs become
relevant.

```bash
uv run exu-train \
  --mode rlcd \
  --train data/tickets.jsonl --train-split train \
  --validation data/tickets.jsonl --validation-split validation \
  --calibration data/tickets.jsonl --calibration-split validation \
  --output artifacts/exu \
  --encoder google-bert/bert-base-multilingual-cased \
  --epochs 4 --batch-size 8 --option-shuffle --calibrate \
  --samples-per-question 4 --sigma-start 0.4 --sigma-end 0.1 \
  --ce-weight 1.0 --advantage-norm batch
```

Defaults are the fine-tune recipe: 4 samples per question, sigma annealing from
0.4 to 0.1, a full-weight cross-entropy term, and batch-wide advantage
normalization. `PolicyConfig.base()` holds the base-training recipe instead: 8
samples, sigma 1.0 to 0.3, no cross-entropy.

Keep the policy only if it beats the baseline on held-out ECE or NLL. It often
does not, and that is a result, not a failure. The critical reading of why is at
the end of phase 7 of [exu-guide.md](exu-guide.md).

## 5. Read the training report

`training.json` lands next to the checkpoint, inside `--output`. It records the
config, one entry per epoch, the validation metrics, and the temperature fit when
`--calibrate` was used.

```bash
python -c "import json; d=json.load(open('artifacts/exu/training.json')); print(d['validation'])"
```

What to look at:

- `validation` block: `count`, `nll`, `brier`, `accuracy`, `soft_accuracy`, `ece`,
  plus `rps` and `ordinal_mae` for `score` questions. These are the numbers to
  compare against the baseline run.
- `train_epochs[*].mean_reward`: should rise, or at least not fall apart.
- `calibration.fits`: one entry per bucket, with `samples`, `nll_before`,
  `nll_after`, `fallback` and `at_bound`. A fit with `at_bound: true` means the
  temperature landed on the clamp limit. That is a signal to look at, never a
  result to ship. A fit with `fallback: true` had too few samples to be believed.

## 6. Evaluate on held-out

```bash
uv run exu-evaluate \
  --checkpoint artifacts/exu \
  --data data/tickets.jsonl --split test \
  --order-permutations 4 --latency \
  --output artifacts/report.json
```

The report has: overall metrics, per question kind, per family, the three trivial
baselines (uniform, per-question prior, majority class), selective coverage, order
robustness and latency percentiles. Every field is defined in
[evaluation.md](evaluation.md).

How to read it, in order:

1. **Beat the baselines.** If `accuracy` or `nll` does not beat the majority
   class and the per-question prior, nothing else matters.
2. **Look at `ece` next to `accuracy`.** Accuracy comes from the argmax, ECE from
   the maximum probability. A model that is right 80% of the time with ECE 0.2
   should not be gated on its own confidence.
3. **`order_robustness.stability` close to 1.** Below roughly 0.9 the model is
   reading position, not criteria. Train with `--option-shuffle` and more epochs.
4. **`coverage.top_0.5` above `coverage.all`.** If answering only the most
   confident half does not raise accuracy, confidence is useless for gating.
5. **Latency**, if you plan to serve it.

Beware of evaluating on the split you trained on. The numbers will be excellent
and meaningless.

## 7. Serve it

```python
from exu import DecisionQuestion, DecisionRuntime, Option

runtime = DecisionRuntime.load("artifacts/exu")
decision = runtime.decide(
    "I was charged twice for the same invoice.",
    DecisionQuestion.choice(
        "Where should this ticket go?",
        [Option("billing", "payment, invoice or refund"), Option("support", "access or outage")],
    ),
)
print(decision.label, decision.probabilities, decision.confidence, decision.should_act)
```

- `runtime.decide_many([(state, question), ...])` answers a batch of questions in
  one forward pass. Prefer it over a loop: the cost is re-encoding the state once
  per question either way, and the batch is far faster per question.
- Load once and keep the runtime resident. A cold load is seconds.
- `confidence` and `entropy_confidence` are different scales. `should_act` uses
  the calibrated maximum probability. Read the confidence section of
  [evaluation.md](evaluation.md) before putting a business threshold on either.
- `decision.expected_level` is only meaningful for `score` questions.

### From the command line

`exu-decide` is the same runtime behind a CLI. One question, written with flags:

```bash
exu-decide --checkpoint artifacts/exu \
  --state "I was charged twice for the same invoice." \
  --instruction "Where should this ticket go?" \
  --option "billing=payment, invoice or refund" \
  --option "support=access or outage"
```

`--option NAME[=DESCRIPTION]` repeats, and only the first `=` separates the two.
`--kind score` takes `--level TEXT`, lowest first, and `--kind noul` answers `No` or
`Yes` unless `--false-option` and `--true-option` say otherwise. `--state-file`
reads the state from a file.

Many questions, one JSON record per line, from a file or from stdin with `-`:

```bash
exu-decide --checkpoint artifacts/exu --input questions.jsonl --output decisions.jsonl
```

A record needs `state` and `question`, in the dataset format. `id` is echoed when
it is there and everything else is ignored, so a dataset file works as it is. The
output is one JSON object per question, in input order, with `label`,
`confidence`, `entropy_confidence`, `should_act`, `expected_level`, and
`probabilities` and `logits` keyed by option name.

`--top-only` keeps just the winner: `{"label": "billing", "confidence": 0.87}`, plus
the `id` when there is one. `--metrics` adds a `metrics` object to every answer
(`batch_ms`, `batch_size`, `per_question_ms`) and prints one JSON summary line to
stderr (`questions`, `batches`, `load_ms`, `inference_ms`, `first_batch_ms`,
`questions_per_second`), so stdout keeps one schema. The time covers the whole
answer: building the sequences, the forward pass and reading the result back. The
first batch on a GPU also pays for warm-up, which is why it is reported apart. A
question does not have a latency of its own inside a batch, so `per_question_ms`
is the batch time divided by its size.

The whole input is parsed before the first forward pass. A broken line is named
(`error: line 7: missing field 'question'`), the exit code is 1, and nothing is
answered, so a pipeline never receives half a result. A malformed command line
exits with 2. Every call loads the checkpoint, which costs seconds: a service
should keep a `DecisionRuntime` resident instead.

The checkpoint format, the validation performed before loading and the
device/precision rules are in [checkpoints.md](checkpoints.md).

## 8. When it goes wrong

| Symptom | Cause and what to do |
| --- | --- |
| `too many options for configured header_budget` | The question's options cannot fit the header. With the defaults, roughly 30 options is the ceiling. Raise `--header-budget` (and `--max-length` with it), or decide in two stages. |
| `tokenizer must provide mask_token_id` | The encoder is decoder-only or has no mask token. Pick a bidirectional encoder that has one. |
| `each target distribution must sum to one` | Fix the `target` in the record named in the error. |
| `output path already exists` | `--output` refuses to overwrite. Choose a new directory. |
| Accuracy at chance, validation flat | Options are not distinguishable: texts too similar, too little description, or the budget squeezed them. Check the header budget and `--option-shuffle`, then the encoder. |
| ECE high, accuracy fine | Calibrate on a real held-out split (`--calibration` + `--calibrate`), and check for `at_bound` fits in the report. |
| `stability` low | The model learned position. Turn on `--option-shuffle`. |
| CUDA out of memory | `--device cpu`, or smaller `--batch-size`, or a smaller `--max-length`, or a smaller encoder. |
| `CUDA was requested but is unavailable` | Use `--device auto`. |
| Loss does not move | Check the encoder learning rate against the head's, and that `target` is not all zeros or all ones on every record. |

## 9. What not to commit

Datasets, checkpoints, `artifacts/`, `runs/`, `_site/`, `dist/`. All gitignored;
keep it that way. Datasets and checkpoints carry their own licenses, and a
checkpoint carries the encoder's license with it. Before publishing an artifact,
confirm the encoder, the tokenizer, the data and the resulting weights are
compatible.

Small synthetic fixtures are for plumbing. They are not evidence of quality, and
no number measured on them belongs in a README.
