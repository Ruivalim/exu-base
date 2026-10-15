# Dataset format

One JSONL record is one fully specified typed decision. The file is UTF-8 and one
record per line.

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

## Fields

| Field | Required | Notes |
| --- | --- | --- |
| `id` | yes | Non-empty. Your own bookkeeping: it is not used for dedup, and errors name the line number instead. |
| `state` | yes | String, object or list. Objects and lists become compact JSON without non-ASCII escapes. |
| `context` | no | What the answer depends on besides the state: a policy, a rubric, background. String, object or list, serialized like `state`, never empty. See below. |
| `question.kind` | yes | `choice`, `score` or `noul`. |
| `question.instruction` | yes | Non-empty. The question text the model reads. |
| `question.options` | yes | 2 to 255 entries. Names must be unique. |
| `question.options[].name` | yes | The label returned to the caller. |
| `question.options[].description` | no | The criterion. Renders as `name: description`. |
| `target` | yes | Length equals option count, non-negative, finite, sums to 1. |
| `split` | no | `train`, `validation`, `calibration`, `test`, or your own names. |
| `family` | no | Task family, used for grouped metrics and honest held-out splits. |
| `language` | no | Language variant tag, for provenance. |
| `case_id` | no | Non-empty. Which case the record belongs to: records that share a state share a `case_id`. Never read by the model. |
| `metadata` | no | Flat object of string keys to string values, for grouping reports. Never read by the model. |

`noul` questions must have exactly two options. A blank description is dropped.

## Case and metadata

`case_id` and `metadata` are provenance. The builder never sees them, so changing
either leaves the sequence the model reads exactly as it was, and both survive
`write_jsonl` and option shuffling.

`case_id` says which questions are about the same case. Five questions about one
state are five rows but one case, and they are not independent evidence:
`exu-evaluate` reports `rows_with_case_id` and `distinct_cases` under `cases`, and
leaves `distinct_cases` `null` when no row has a `case_id` instead of guessing the
row count.

`metadata` holds whatever you want a report split by, such as the policy a record
was built from: `{"policy_family_id": "refunds", "generator": "template"}`. Values
must be strings; a number, a list or a nested object is refused at load, so a
group key never depends on how a number was printed. `exu-evaluate --group-by
metadata.policy_family_id` then reports per value. A record without that key is
counted under `missing`.

## Context

`context` is read before the state, in its own slot with its own budget
(`--context-budget`, 256 tokens by default). A context over the budget is refused,
never cut: training checks every record before it loads the model and names the
first one that does not fit. The state is what gets cut when a sequence runs long,
so a long trace cannot push the rules it is judged by out of the sequence.

Use it for what the same state is judged against: the same refund request under a
30-day and a 90-day policy is two records with two contexts and two targets. A
model only learns to use a context it was trained with, and the most useful
training data has pairs like that one, where the context changes the answer.
A record without `context` builds exactly the sequence it built before the field
existed.

## Targets are distributions

If a target were a label, this would be a classifier. A distribution lets several
annotators or a teacher model express disagreement, which is exactly the signal a
calibrated model should learn from.

- Hard label: `[0.0, 1.0]`.
- Annotator split, two of four ambiguous: `[0.5, 0.5]`.
- Teacher distribution: whatever the teacher produced, after review.

Soft targets are not a shortcut. Laya's fine-tune is trained on
teacher-labeled soft distributions, and it is where the fine-tune shines, but a
human set is still needed to measure calibration.

## Splits, and keeping held-out honest

- Split the held-out set by **entire task family**, not by example. Otherwise you
  measure memorization and call it generalization.
- Keep a separate `calibration` split for temperature fitting. Fitting on
  training data makes `T` land near 1 and flatters the calibration.
- Never move a family between splits after seeing results.

`family` exists for this, and `exu-evaluate` reports metrics per family so you
can see the gap between seen and unseen families.

## Augmentation

`exu-train --option-shuffle` permutes a question's options on every pass and
permutes the target with them, so position carries no information. Evaluate with
`exu-evaluate --order-permutations N` to measure whether it worked.

The augmentation module also exposes `permute_example` and `shuffle_example` if
you want to build your own transform; `DecisionDataset` accepts any
`TrainingExample -> TrainingExample` callable.

## Provenance and licenses

The loader only requires the fields above, but a distributable dataset should
record, per record or in an accompanying card: source, license, language variant,
transformation, annotation method, teacher model and prompts for synthetic rows,
PII review, and the split strategy.

Do not commit datasets or checkpoints to this repository. See
[CONTRIBUTING.md](../CONTRIBUTING.md).

## Minimal working example

```bash
exu-train \
  --mode baseline \
  --train data.jsonl --train-split train \
  --validation data.jsonl --validation-split validation \
  --output artifacts/baseline --epochs 2 --batch-size 4
```
