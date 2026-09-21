"""Evaluate a checkpoint on a JSONL split.

Reports decision quality and calibration overall, per question kind and per task
family, next to three trivial baselines: uniform, the per-question prior, and the
majority class. In Laya the base checkpoints scored below the
majority-class baseline on unseen families, so these numbers are not decoration.

The prior and the majority class are fitted on reference labels, the ``train``
split of ``--data`` unless ``--reference`` says otherwise, and never on the split
being scored. Without a usable reference they are reported as unavailable, with
the reason, instead of quietly falling back to the evaluation labels.

Optional passes: order robustness (does the answer survive an option permutation)
and latency (p50/p95 for 1, 5, 10 and 50 questions in one call).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from torch.utils.data import DataLoader

from .checkpoint import load_checkpoint, load_tokenizer
from .data import DecisionDataset, TrainingBatch, load_jsonl
from .devices import resolve_device
from .evaluation import (
    collect,
    grouped_metrics,
    latency_benchmark,
    metrics_for,
    order_robustness,
)
from .metrics import (
    ReferencePrior,
    classification_metrics,
    fit_reference_prior,
    prior_in_sample_baseline,
    selective_coverage,
    stack_examples,
    uniform_baseline,
)
from .sequence import SequenceBuilder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate an Exu checkpoint.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="checkpoint directory")
    parser.add_argument("--data", required=True, type=Path, help="UTF-8 JSONL records")
    parser.add_argument("--split", help="optional split name selected from --data")
    parser.add_argument(
        "--reference",
        type=Path,
        help="JSONL whose labels fit the prior and majority baselines (default: --data)",
    )
    parser.add_argument(
        "--reference-split",
        default="train",
        help="split of the reference file to fit the baselines on",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--order-permutations",
        type=int,
        default=0,
        help="permutations per example for the order-robustness pass (0 disables)",
    )
    parser.add_argument("--latency", action="store_true", help="run the latency benchmark")
    parser.add_argument(
        "--latency-questions", default="1,5,10,50", help="comma-separated question counts"
    )
    parser.add_argument("--output", type=Path, help="write the report to this JSON file")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    device = resolve_device(args.device)

    checkpoint = load_checkpoint(args.checkpoint, device=device)
    tokenizer = load_tokenizer(args.checkpoint)
    builder = SequenceBuilder(tokenizer, checkpoint.sequence_config)
    examples = load_jsonl(args.data, args.split)
    dataset = DecisionDataset(examples, builder)
    loader = _loader(dataset, args.batch_size)
    collected = collect(checkpoint.model, loader, device)
    temperature = checkpoint.temperature

    report: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "data": str(args.data),
        "split": args.split,
        "device": str(device),
        "temperature": temperature.to_dict(),
        "metrics": asdict(metrics_for(collected, temperature)),
        "by_kind": {
            key: asdict(value)
            for key, value in grouped_metrics(collected, temperature, "kind").items()
        },
        "by_family": {
            key: asdict(value)
            for key, value in grouped_metrics(collected, temperature, "family").items()
        },
        "baselines": _baselines(
            examples,
            reference=args.reference or args.data,
            reference_split=args.reference_split,
            data=args.data,
            split=args.split,
        ),
        "coverage": selective_coverage(
            collected.probabilities(temperature), collected.target, collected.option_mask
        ),
    }
    if args.order_permutations > 0:
        report["order_robustness"] = order_robustness(
            checkpoint.model,
            builder,
            examples,
            temperature,
            permutations=args.order_permutations,
            device=device,
        )
    if args.latency:
        questions = [int(value) for value in args.latency_questions.split(",") if value.strip()]
        report["latency"] = latency_benchmark(
            checkpoint.model,
            builder,
            examples,
            temperature,
            questions_per_call=questions,
            device=device,
        )

    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


def _baselines(
    examples: Sequence,
    *,
    reference: Path,
    reference_split: str,
    data: Path,
    split: str | None,
) -> dict[str, object]:
    targets, mask, kinds, _counts = stack_examples(examples)
    ordinal = _ordinal_mask(kinds)

    def score(probabilities, rows=None) -> dict[str, float]:
        if rows is None:
            return asdict(
                classification_metrics(probabilities, targets, mask, ordinal_mask=ordinal)
            )
        kept = None if ordinal is None else ordinal[rows]
        return asdict(
            classification_metrics(
                probabilities[rows], targets[rows], mask[rows], ordinal_mask=kept
            )
        )

    report: dict[str, object] = {
        "uniform": score(uniform_baseline(examples)),
        "prior": None,
        "majority": None,
        "prior_in_sample": score(prior_in_sample_baseline(examples)),
    }
    provenance: dict[str, object] = {
        "path": str(reference),
        "split": reference_split,
        "smoothing": "(count + 1/K) / (n + 1)",
        "unseen_question": "uniform, the n = 0 case of the smoothing",
        "question_key": "kind, instruction and options; choice options in any order",
        "unavailable": None,
    }
    report["reference"] = provenance

    prior = _fit_reference(reference, reference_split, data, split, provenance)
    if prior is None:
        return report

    predictions, seen = prior.predict(examples)
    provenance["seen_rows"] = int(seen.sum().item())
    provenance["unseen_rows"] = int((~seen).sum().item())
    report["prior"] = score(predictions)
    if seen.any() and not seen.all():
        report["prior_seen"] = score(predictions, seen)
        report["prior_unseen"] = score(predictions, ~seen)
    # A one-hot forecast puts an exact zero on every outcome it misses, so its NLL
    # is the error rate times -log(float32 tiny). Only the accuracy means anything.
    report["majority"] = {"accuracy": score(prior.majority(examples))["accuracy"]}
    return report


def _fit_reference(
    reference: Path,
    reference_split: str,
    data: Path,
    split: str | None,
    provenance: dict[str, object],
) -> ReferencePrior | None:
    if reference.resolve() == data.resolve() and split in (None, reference_split):
        provenance["unavailable"] = "the reference rows are among the rows being scored"
        return None
    try:
        rows = load_jsonl(reference, reference_split)
    except (OSError, ValueError) as error:
        provenance["unavailable"] = str(error)
        return None
    provenance["rows"] = len(rows)
    return fit_reference_prior(rows)


def _ordinal_mask(kinds: Sequence[str]):
    import torch

    return torch.tensor([kind == "score" for kind in kinds], dtype=torch.bool)


def _loader(dataset: DecisionDataset, batch_size: int) -> DataLoader[TrainingBatch]:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate)


if __name__ == "__main__":
    raise SystemExit(main())
