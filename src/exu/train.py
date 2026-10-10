"""Train a calibrated decision model.

Two modes share one pipeline:

- ``baseline``: direct optimization of the strictly proper composite score. Fast,
  stable, and the bar the RLCD mode has to beat.
- ``rlcd``: perturbed-logit policy gradient with the same score as reward.

``--backbone`` picks the architecture. ``encoder`` is the default and reads every
option of a question at its marker in one sequence. ``decoder`` is experimental:
one sequence per option, read at the last token, K times the forward cost. The
choice is stored in the checkpoint, so evaluation and inference never ask again.

After training, ``--calibrate`` fits temperatures on a held-out split (the
calibration split if given, validation otherwise) and reports the ECE before and
after. The checkpoint stores the model, the sequence configuration, the
temperature map and the tokenizer, so inference never touches the network.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .architecture import DecisionModel, builder_for, model_from_pretrained
from .augmentation import OptionShuffler
from .calibration import TemperatureMap, fit_temperature_map
from .checkpoint import save_checkpoint
from .data import DecisionDataset, TrainingBatch, load_jsonl
from .devices import autocast_context, resolve_device, resolve_dtype
from .evaluation import collect, metrics_for
from .lora import apply_lora, merge_lora
from .model import DEFAULT_DECODER, DEFAULT_ENCODER, SCORERS, ExuConfig, masked_log_softmax
from .policy import PolicyConfig, policy_gradient_loss, sigma_for
from .scoring import CONFIDENT_MISS, confident_misses, proper_scoring_loss, reward_definition
from .sequence import SequenceConfig

_DEFAULT_POLICY = PolicyConfig()
# Per backbone. The decoder numbers in BENCHMARKS.md peak at 1e-5 and one epoch;
# the encoder default is the one the encoder runs used.
_DEFAULT_BACKBONE = {"encoder": DEFAULT_ENCODER, "decoder": DEFAULT_DECODER}
_DEFAULT_BACKBONE_LR = {"encoder": 2.5e-5, "decoder": 1e-5}
# The decoder gained from a decaying rate on typed-decisions and the encoder did not
# (BENCHMARKS.md, section 14).
_DEFAULT_SCHEDULE = {"encoder": ("constant", 0), "decoder": ("cosine", 10)}
# Adapters train a small update from zero and want a larger rate than a full
# fine-tune: this is the rate of Unsloth's decision recipe, also rank 16.
_DEFAULT_LORA_LR = 2e-4
LR_SCHEDULES = ("constant", "cosine")


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
    parser.add_argument(
        "--backbone",
        choices=tuple(_DEFAULT_BACKBONE),
        default="encoder",
        help="encoder reads every option at its marker in one sequence. decoder "
        "(experimental) reads one sequence per option at its last token",
    )
    parser.add_argument(
        "--encoder",
        help="Hugging Face name or folder of the pretrained backbone, encoder or decoder "
        f"(default: {DEFAULT_ENCODER}, or {DEFAULT_DECODER} with --backbone decoder)",
    )
    parser.add_argument("--mode", choices=("baseline", "rlcd"), default="rlcd")
    parser.add_argument(
        "--scorer",
        choices=SCORERS,
        default="marker",
        help="marker reads each option at its marker. marker-cls adds a term that "
        "crosses the option's text with [CLS], for answers the option names do not hint at",
    )
    parser.add_argument("--epochs", type=_positive_int, default=4)
    parser.add_argument("--batch-size", type=_positive_int, default=8)
    parser.add_argument("--grad-accum", type=_positive_int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--encoder-lr",
        type=_positive_float,
        help="learning rate of the backbone, or of its adapters with --lora-rank (default: "
        "2.5e-5, or 1e-5 with --backbone decoder, or 2e-4 with --lora-rank)",
    )
    parser.add_argument("--head-lr", type=_positive_float, default=1e-4)
    parser.add_argument(
        "--lora-rank",
        type=_non_negative_int,
        default=0,
        help="freeze the backbone and train a rank-R update on each of its linear layers, "
        "merged into the weights before saving. 0 trains the whole backbone (default: 0)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=_positive_float,
        help="the update is scaled by alpha / rank (default: the rank, a scale of 1)",
    )
    parser.add_argument("--weight-decay", type=_non_negative_float, default=0.01)
    parser.add_argument(
        "--lr-schedule",
        choices=LR_SCHEDULES,
        help="constant keeps both learning rates fixed; cosine decays them to zero by the "
        "last update (default: constant, or cosine with --backbone decoder)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=_non_negative_int,
        help="updates over which both learning rates ramp up linearly from zero "
        "(default: 0, or 10 with --backbone decoder, shortened to fit a shorter run)",
    )
    parser.add_argument("--max-grad-norm", type=_positive_float, default=1.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--max-length", type=_positive_int, default=512)
    parser.add_argument(
        "--header-budget",
        type=_positive_int,
        help="tokens for everything but the state (default: 192, or --max-length if smaller)",
    )
    parser.add_argument(
        "--grad-checkpointing",
        action="store_true",
        help="recompute backbone activations in the backward pass: less memory, slower",
    )
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


def resolve_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Fill the flags whose default depends on the backbone or on another flag."""
    args.encoder = args.encoder or _DEFAULT_BACKBONE[args.backbone]
    default_lr = _DEFAULT_LORA_LR if args.lora_rank else _DEFAULT_BACKBONE_LR[args.backbone]
    args.encoder_lr = args.encoder_lr or default_lr
    args.lora_alpha = args.lora_alpha or (float(args.lora_rank) if args.lora_rank else None)
    schedule, warmup = _DEFAULT_SCHEDULE[args.backbone]
    args.lr_schedule = args.lr_schedule or schedule
    # A default warm-up may be shortened to fit the run; an explicit one may not.
    args.default_warmup = args.warmup_steps is None
    if args.default_warmup:
        args.warmup_steps = warmup
    args.header_budget = args.header_budget or min(192, args.max_length)
    return args


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
    # Share of claimed components the model gave less than CONFIDENT_MISS to, at
    # its own predictions, and over the sampled candidates in rlcd mode.
    confident_miss_rate: float = 0.0
    candidate_confident_miss_rate: float | None = None


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"output path already exists: {args.output}")
    if args.backbone == "decoder" and args.scorer != "marker":
        parser.error("--scorer marker-cls exists for the encoder's markers only")
    if args.lora_alpha is not None and not args.lora_rank:
        parser.error("--lora-alpha needs --lora-rank")
    resolve_defaults(args)

    _set_seed(args.seed)
    device = resolve_device(args.device)
    policy = _policy_from_args(args) if args.mode == "rlcd" else None
    sequence = SequenceConfig(max_length=args.max_length, header_budget=args.header_budget)
    config = ExuConfig(encoder_name=args.encoder, scorer=args.scorer, backbone=args.backbone)

    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    builder = builder_for(tokenizer, sequence, config)
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

    model = model_from_pretrained(config)
    adapted = 0
    if args.lora_rank:
        # Frozen weights in the autocast dtype: the same forward pass, half the memory.
        adapted = apply_lora(
            model.backbone, args.lora_rank, args.lora_alpha, base_dtype=resolve_dtype(device)
        )
        print(f"lora: rank {args.lora_rank} on {adapted} linear layers, backbone frozen")
    if args.grad_checkpointing:
        model.backbone.gradient_checkpointing_enable()
    model = model.to(device)
    backbone_ids = {id(parameter) for parameter in model.backbone.parameters()}
    optimizer = AdamW(
        [
            {
                "params": [p for p in model.backbone.parameters() if p.requires_grad],
                "lr": args.encoder_lr,
            },
            {
                "params": [
                    parameter
                    for parameter in model.parameters()
                    if id(parameter) not in backbone_ids
                ],
                "lr": args.head_lr,
            },
        ],
        weight_decay=args.weight_decay,
        # The fused multi-tensor step allocates a copy of the second moments. On a
        # 12 GB card that copy is what ran a 0.6B decoder out of memory.
        foreach=False if args.backbone == "decoder" else None,
    )

    steps_per_epoch = max(1, (len(train_loader) + args.grad_accum - 1) // args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    if args.default_warmup:
        args.warmup_steps = min(args.warmup_steps, total_steps - 1)
    if args.warmup_steps >= total_steps:
        parser.error(
            f"--warmup-steps {args.warmup_steps} leaves no update after warm-up: "
            f"this run makes {total_steps}"
        )
    scheduler = LambdaLR(
        optimizer,
        lambda update: lr_multiplier(update, total_steps, args.warmup_steps, args.lr_schedule),
    )
    step = 0
    history: list[EpochStats] = []
    for epoch in range(1, args.epochs + 1):
        stats, step = _train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
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

    # Merged before any evaluation, so the temperatures are fitted on the float32 model
    # the checkpoint will hold, not on its lower-precision frozen copy.
    if args.lora_rank:
        # On the CPU, with the optimizer gone: next to the frozen copy, the float32
        # weights of a 2B base do not fit on a 12 GB card.
        del optimizer, scheduler
        model.to("cpu")
        merge_lora(model.backbone)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        model.to(device)
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
        "architecture": config.architecture(),
        "encoder": args.encoder,
        "encoder_lr": args.encoder_lr,
        "head_lr": args.head_lr,
        "lr_schedule": args.lr_schedule,
        "warmup_steps": args.warmup_steps,
        "lora": (
            {"rank": args.lora_rank, "alpha": args.lora_alpha, "layers": adapted}
            if args.lora_rank
            else None
        ),
        "grad_checkpointing": args.grad_checkpointing,
        "scorer": args.scorer,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "option_shuffle": args.option_shuffle,
        "reward": reward_definition(args.spherical_weight, args.rps_weight),
        "confident_miss_threshold": CONFIDENT_MISS,
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
    model: DecisionModel,
    loader: DataLoader[TrainingBatch],
    optimizer: AdamW,
    scheduler: LambdaLR,
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
    missed = claimed = candidate_missed = candidate_claimed = 0

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
                log_probabilities = masked_log_softmax(output.logits, moved.option_mask)
                with torch.no_grad():
                    counts = torch.stack(confident_misses(log_probabilities, moved.targets))
                missed, claimed = missed + counts[0], claimed + counts[1]
                loss = proper_scoring_loss(
                    log_probabilities,
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
                missed += metrics.confident_misses
                claimed += metrics.claimed_components
                candidate_missed += metrics.candidate_confident_misses
                candidate_claimed += metrics.candidate_components
        (loss * group_scale).backward()
        losses.append(float(loss.detach().item()))
        pending += 1
        if pending >= args.grad_accum:
            _optimizer_step(model, optimizer, scheduler, args.max_grad_norm)
            pending = 0
            step += 1
        if args.log_every and (index + 1) % args.log_every == 0:
            print(f"  step {step} loss={losses[-1]:.4f}")
    if pending:
        _optimizer_step(model, optimizer, scheduler, args.max_grad_norm)
        step += 1

    stats = EpochStats(
        loss=sum(losses) / len(losses),
        steps=len(losses),
        policy_loss=sum(policy_losses) / len(policy_losses) if policy_losses else None,
        cross_entropy=sum(cross_entropies) / len(cross_entropies) if cross_entropies else None,
        mean_reward=sum(rewards) / len(rewards) if rewards else None,
        mean_advantage=sum(advantages) / len(advantages) if advantages else None,
        sigma_end=last_sigma,
        confident_miss_rate=float(missed) / max(float(claimed), 1.0),
        candidate_confident_miss_rate=(
            candidate_missed / candidate_claimed if candidate_claimed else None
        ),
    )
    return stats, step


def _optimizer_step(
    model: DecisionModel, optimizer: AdamW, scheduler: LambdaLR, max_grad_norm: float
) -> None:
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)


def lr_multiplier(update: int, total_steps: int, warmup_steps: int, schedule: str) -> float:
    """The factor on both learning rates for the update with this 0-based index.

    Warm-up ramps linearly and reaches the full rate at update ``warmup_steps``, so
    the first update is already a small step rather than a wasted one at zero. Cosine
    then decays over the remaining updates and would reach zero at ``total_steps``,
    one past the last update, so the last update still moves the weights.
    """
    if schedule not in LR_SCHEDULES:
        raise ValueError(f"unknown lr schedule {schedule!r}, expected one of {LR_SCHEDULES}")
    if update < warmup_steps:
        return (update + 1) / (warmup_steps + 1)
    if schedule == "constant":
        return 1.0
    progress = (update - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


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
