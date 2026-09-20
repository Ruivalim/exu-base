"""Train a calibrated decision model.

Two modes share one pipeline:

- ``baseline``: direct optimization of the strictly proper composite score. Fast,
  stable, and the bar the RLCD mode has to beat.
- ``rlcd``: perturbed-logit policy gradient with the same score as reward.

After training, ``--calibrate`` fits temperatures on a held-out split (the
calibration split if given, validation otherwise) and reports the ECE before and
after. The checkpoint stores the model, the sequence configuration, the
temperature map and the tokenizer, so inference never touches the network.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .augmentation import OptionShuffler
from .calibration import TemperatureMap, fit_temperature_map
from .checkpoint import save_checkpoint
from .data import DecisionDataset, TrainingBatch, load_jsonl
from .devices import autocast_context, resolve_device
from .evaluation import collect, metrics_for
from .model import ExuConfig, ExuModel
from .policy import PolicyConfig, policy_gradient_loss, sigma_for
from .scoring import proper_scoring_loss
from .sequence import SequenceBuilder, SequenceConfig

_DEFAULT_POLICY = PolicyConfig()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a calibrated decision model (direct baseline or RLCD)."
    )
    parser.add_argument("--train", required=True, type=Path, help="UTF-8 JSONL training records")
    parser.add_argument("--train-split", help="optional split name selected from --train")
    parser.add_argument(
        "--validation", required=True, type=Path, help="UTF-8 JSONL validation records"
    )
    parser.add_argument("--validation-split", help="optional split name selected from --validation")
    parser.add_argument("--test", type=Path, help="optional UTF-8 JSONL test records")
    parser.add_argument("--test-split", help="optional split name selected from --test")
    parser.add_argument("--calibration", type=Path, help="optional held-out JSONL for temperatures")
    parser.add_argument(
        "--calibration-split", help="optional split name selected from --calibration"
    )
    parser.add_argument("--output", required=True, type=Path, help="new checkpoint directory")
    parser.add_argument("--encoder", default=ExuConfig().encoder_name)
    parser.add_argument("--mode", choices=("baseline", "rlcd"), default="rlcd")
    parser.add_argument("--epochs", type=_positive_int, default=4)
    parser.add_argument("--batch-size", type=_positive_int, default=8)
    parser.add_argument("--grad-accum", type=_positive_int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--encoder-lr", type=_positive_float, default=2.5e-5)
    parser.add_argument("--head-lr", type=_positive_float, default=1e-4)
    parser.add_argument("--weight-decay", type=_non_negative_float, default=0.01)
    parser.add_argument("--max-grad-norm", type=_positive_float, default=1.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--max-length", type=_positive_int, default=512)
    parser.add_argument("--header-budget", type=_positive_int, default=192)
    parser.add_argument(
        "--option-shuffle", action="store_true", help="permute a question's options every pass"
    )
    parser.add_argument(
        "--samples-per-question", type=_positive_int, default=_DEFAULT_POLICY.samples_per_question
    )
    parser.add_argument("--sigma-start", type=_positive_float, default=_DEFAULT_POLICY.sigma_start)
    parser.add_argument("--sigma-end", type=_non_negative_float, default=_DEFAULT_POLICY.sigma_end)
    parser.add_argument(
        "--ce-weight", type=_non_negative_float, default=_DEFAULT_POLICY.cross_entropy_weight
    )
    parser.add_argument(
        "--advantage-norm",
        choices=("batch", "group", "none"),
        default=_DEFAULT_POLICY.advantage_norm,
    )
    parser.add_argument(
        "--spherical-weight",
        type=_non_negative_float,
        default=_DEFAULT_POLICY.spherical_weight,
    )
    parser.add_argument(
        "--rps-weight", type=_non_negative_float, default=_DEFAULT_POLICY.rps_weight
    )
    parser.add_argument("--calibrate", action="store_true", help="fit temperatures after training")
    parser.add_argument("--calibration-min-samples", type=_positive_int, default=64)
    parser.add_argument(
        "--log-every",
        type=_non_negative_int,
        default=0,
        help="0 disables step logs",
    )
    return parser


@dataclass(frozen=True, slots=True)
class EpochStats:
    """Aggregated training diagnostics for one epoch."""

    loss: float
    steps: int
    policy_loss: float | None = None
    cross_entropy: float | None = None
    mean_reward: float | None = None
    mean_advantage: float | None = None
    sigma_end: float | None = None


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"output path already exists: {args.output}")

    _set_seed(args.seed)
    device = resolve_device(args.device)
    policy = _policy_from_args(args) if args.mode == "rlcd" else None
    sequence = SequenceConfig(max_length=args.max_length, header_budget=args.header_budget)

    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    builder = SequenceBuilder(tokenizer, sequence)
    train_examples = load_jsonl(args.train, args.train_split)
    validation_examples = load_jsonl(args.validation, args.validation_split)
    test_examples = load_jsonl(args.test, args.test_split) if args.test else None
    transform = OptionShuffler(args.seed) if args.option_shuffle else None

    train_loader = _loader(
        DecisionDataset(train_examples, builder, transform),
        args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    validation_loader = _loader(DecisionDataset(validation_examples, builder), args.batch_size)
    test_loader = (
        _loader(DecisionDataset(test_examples, builder), args.batch_size) if test_examples else None
    )

    model = ExuModel.from_pretrained(ExuConfig(encoder_name=args.encoder)).to(device)
    optimizer = AdamW(
        [
            {"params": model.encoder.parameters(), "lr": args.encoder_lr},
            {
                "params": [
                    parameter
                    for name, parameter in model.named_parameters()
                    if not name.startswith("encoder.")
                ],
                "lr": args.head_lr,
            },
        ],
        weight_decay=args.weight_decay,
    )

    steps_per_epoch = max(1, (len(train_loader) + args.grad_accum - 1) // args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    step = 0
    history: list[EpochStats] = []
    for epoch in range(1, args.epochs + 1):
        stats, step = _train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            policy,
            args,
            step,
            total_steps,
        )
        history.append(stats)
        print(
            f"epoch {epoch}/{args.epochs} loss={stats.loss:.4f}"
            + (f" reward={stats.mean_reward:.4f}" if stats.mean_reward is not None else "")
        )

    uncalibrated = TemperatureMap()
    validation = metrics_for(collect(model, validation_loader, device), uncalibrated)
    test = metrics_for(collect(model, test_loader, device), uncalibrated) if test_loader else None

    calibration_report = None
    temperature = uncalibrated
    if args.calibrate:
        calibration_examples = (
            load_jsonl(args.calibration, args.calibration_split)
            if args.calibration
            else validation_examples
        )
        calibration_loader = _loader(
            DecisionDataset(calibration_examples, builder), args.batch_size
        )
        collected = collect(model, calibration_loader, device)
        temperature, fits = fit_temperature_map(
            collected.logits,
            collected.target,
            collected.option_mask,
            collected.question_kinds,
            collected.option_counts.tolist(),
            min_samples=args.calibration_min_samples,
        )
        calibration_report = {
            "source": str(args.calibration) if args.calibration else str(args.validation),
            "fits": {
                key: {
                    "temperature": fit.temperature,
                    "samples": fit.samples,
                    "nll_before": fit.nll_before,
                    "nll_after": fit.nll_after,
                    "fallback": fit.fallback,
                    "at_bound": fit.at_bound,
                }
                for key, fit in sorted(fits.items())
            },
            "validation_after": asdict(
                metrics_for(collect(model, validation_loader, device), temperature)
            ),
            "test_after": (
                asdict(metrics_for(collect(model, test_loader, device), temperature))
                if test_loader
                else None
            ),
        }
        print(json.dumps({"temperature_map": temperature.to_dict()}, sort_keys=True))

    model.to("cpu")
    save_checkpoint(
        args.output,
        model,
        sequence_config=sequence,
        temperature=temperature,
        tokenizer=tokenizer,
    )
    summary = {
        "format_version": 1,
        "seed": args.seed,
        "device": str(device),
        "mode": args.mode,
        "encoder": args.encoder,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "option_shuffle": args.option_shuffle,
        "policy": asdict(policy) if policy else None,
        "train_epochs": [asdict(item) for item in history],
        "validation": asdict(validation),
        "test": asdict(test) if test else None,
        "calibration": calibration_report,
    }
    (args.output / "training.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"validation": asdict(validation), "test": asdict(test) if test else None}))
    return 0


def _train_epoch(
    model: ExuModel,
    loader: DataLoader[TrainingBatch],
    optimizer: AdamW,
    device: torch.device,
    policy: PolicyConfig | None,
    args: argparse.Namespace,
    step: int,
    total_steps: int,
) -> tuple[EpochStats, int]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    pending = 0
    losses: list[float] = []
    policy_losses: list[float] = []
    cross_entropies: list[float] = []
    rewards: list[float] = []
    advantages: list[float] = []
    last_sigma: float | None = None

    for index, batch in enumerate(loader):
        moved = batch.to(device)
        # A group that runs past the end of the epoch has fewer micro-batches than
        # `grad_accum`, so scale by the group that actually exists. Dividing every
        # micro-batch by `grad_accum` would shrink the last group's contribution.
        if pending == 0:
            group_scale = _group_scale(args.grad_accum, len(loader) - index)
        with autocast_context(device):
            output = model(**moved.model_inputs())
            if policy is None:
                loss = proper_scoring_loss(
                    output.probabilities,
                    moved.targets,
                    moved.option_mask,
                    moved.ordinal_mask,
                    spherical_weight=args.spherical_weight,
                    rps_weight=args.rps_weight,
                )
            else:
                # `step` counts completed updates, so the last one is total_steps - 1.
                # The schedule spans that index; using total_steps would stop one
                # step short of sigma_end forever.
                span = max(total_steps - 1, 1)
                sigma = sigma_for(min(step, span), span, args.sigma_start, args.sigma_end)
                loss, metrics = policy_gradient_loss(
                    output.logits,
                    moved.targets,
                    moved.option_mask,
                    config=policy,
                    sigma=sigma,
                    ordinal_mask=moved.ordinal_mask,
                )
                last_sigma = sigma
                policy_losses.append(metrics.policy_loss)
                cross_entropies.append(metrics.cross_entropy)
                rewards.append(metrics.mean_reward)
                advantages.append(metrics.mean_advantage)
        (loss * group_scale).backward()
        losses.append(float(loss.detach().item()))
        pending += 1
        if pending >= args.grad_accum:
            _optimizer_step(model, optimizer, args.max_grad_norm)
            pending = 0
            step += 1
        if args.log_every and (index + 1) % args.log_every == 0:
            print(f"  step {step} loss={losses[-1]:.4f}")
    if pending:
        _optimizer_step(model, optimizer, args.max_grad_norm)
        step += 1

    stats = EpochStats(
        loss=sum(losses) / len(losses),
        steps=len(losses),
        policy_loss=sum(policy_losses) / len(policy_losses) if policy_losses else None,
        cross_entropy=sum(cross_entropies) / len(cross_entropies) if cross_entropies else None,
        mean_reward=sum(rewards) / len(rewards) if rewards else None,
        mean_advantage=sum(advantages) / len(advantages) if advantages else None,
        sigma_end=last_sigma,
    )
    return stats, step


def _optimizer_step(model: ExuModel, optimizer: AdamW, max_grad_norm: float) -> None:
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _loader(
    dataset: DecisionDataset, batch_size: int, *, shuffle: bool = False, seed: int | None = None
) -> DataLoader[TrainingBatch]:
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=dataset.collate,
        generator=generator,
    )


def _policy_from_args(args: argparse.Namespace) -> PolicyConfig:
    return PolicyConfig(
        samples_per_question=args.samples_per_question,
        sigma_start=args.sigma_start,
        sigma_end=args.sigma_end,
        cross_entropy_weight=args.ce_weight,
        spherical_weight=args.spherical_weight,
        rps_weight=args.rps_weight,
        advantage_norm=args.advantage_norm,
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _group_scale(grad_accum: int, remaining: int) -> float:
    """The weight that makes every accumulation group count the same.

    A group at the end of an epoch can hold fewer micro-batches than
    ``grad_accum``. Scaling by the group that actually exists keeps its gradient
    from being shrunk, which otherwise underweights the last step of every epoch.
    """
    return 1.0 / min(grad_accum, remaining)


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("cannot be negative")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
