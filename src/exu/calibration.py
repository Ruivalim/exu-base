"""Post-training temperature scaling.

Fit a scalar ``T`` that divides the logits before the softmax, using a held-out
set, so the reported confidence matches observed accuracy. This is the cheapest
and highest-return step in the whole pipeline: on Laya's checkpoints mean
ECE fell from 0.466 to 0.081 and from 0.314 to 0.106.

One ``T`` is not enough. Confidence miscalibration depends on how many options a
question has, so temperatures are bucketed by question type and option count,
with a per-type fallback and a global default.

Three traps, all visible in Laya:

- Fitting on a slice of the training set. The model is more confident about what
  it has seen, so ``T`` lands near 1 and looks better than it is. Use held-out.
- A stale bucket map that shadows newly fitted per-type values. Buckets take
  precedence here by design; clear them when they are not wanted.
- A fitted value sitting on a bound. That is an alarm, not a result. Every fit
  reports ``at_bound`` and lower/upper observations are counted.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from .model import masked_softmax

DEFAULT_BOUNDS: tuple[float, float] = (0.1, 10.0)
_OPTION_BUCKETS: tuple[tuple[int, int, str], ...] = (
    (2, 2, "2"),
    (3, 5, "3-5"),
    (6, 10, "6-10"),
    (11, 255, "11+"),
)


def option_count_bucket(option_count: int) -> str:
    """Map an option count to the bucket label used by the temperature map."""
    if option_count < 2:
        raise ValueError("option_count must be at least two")
    for low, high, label in _OPTION_BUCKETS:
        if low <= option_count <= high:
            return label
    raise ValueError("option_count must be at most 255")


def bucket_key(kind: str, option_count: int) -> str:
    """Key of the ``(type, option count bucket)`` cell."""
    return f"{kind}:{option_count_bucket(option_count)}"


@dataclass(frozen=True, slots=True)
class TemperatureFit:
    """Result of fitting one temperature."""

    temperature: float
    samples: int
    nll_before: float
    nll_after: float
    fallback: bool = False
    at_bound: bool = False


def fit_temperature(
    logits: Tensor,
    target: Tensor,
    option_mask: Tensor,
    *,
    min_samples: int = 64,
    bounds: tuple[float, float] = DEFAULT_BOUNDS,
    max_iter: int = 100,
    learning_rate: float = 0.1,
) -> TemperatureFit:
    """Fit ``T`` by minimizing NLL with LBFGS over a bounded sigmoid.

    ``T`` is ``low + (high - low) * sigmoid(raw)``, so a line search can neither
    leave ``bounds`` nor overflow the closure. The sigmoid does not reach a bound
    exactly, so the outer 0.1% of the range is reported as ``at_bound``.

    Below ``min_samples`` rows the fit is skipped and ``T = 1`` is returned with
    ``fallback`` set: a temperature fitted on a handful of points is worse than
    none.
    """
    if min_samples < 1:
        raise ValueError("min_samples must be positive")
    if bounds[0] <= 0 or bounds[1] <= bounds[0]:
        raise ValueError("bounds must be positive and increasing")
    if logits.dim() != 2:
        raise ValueError("logits must be a (rows, options) tensor")
    if target.shape != logits.shape or option_mask.shape != logits.shape:
        raise ValueError("logits, target and option_mask must share a shape")
    rows = logits.size(0)
    if rows < min_samples:
        return TemperatureFit(1.0, rows, float("nan"), float("nan"), fallback=True)

    logits = logits.float().detach()
    target = target.float().detach()
    mask = option_mask.bool()
    tiny = torch.finfo(logits.dtype).tiny
    low, high = bounds

    def negative_log_likelihood(temperature: Tensor) -> Tensor:
        probabilities = masked_softmax(logits, mask, temperature)
        return -(target * probabilities.clamp_min(tiny).log()).sum(dim=-1).mean()

    # Optimize in a bounded space. Working on exp(log T) lets a line search
    # overflow to infinity, which poisons the closure and raises from inside
    # LBFGS. A sigmoid keeps T strictly inside the bounds and keeps gradients
    # finite; a saturated result is reported as at_bound.
    start = min(max(1.0, low + 1e-3 * (high - low)), high - 1e-3 * (high - low))
    raw_start = math.log((start - low) / (high - start))

    def temperature_from(raw: Tensor) -> Tensor:
        return low + (high - low) * torch.sigmoid(raw)

    with torch.enable_grad():
        before = float(negative_log_likelihood(torch.tensor(1.0)).item())
        raw = torch.tensor(raw_start, requires_grad=True)
        optimizer = torch.optim.LBFGS(
            [raw], lr=learning_rate, max_iter=max_iter, line_search_fn="strong_wolfe"
        )

        def closure() -> Tensor:
            optimizer.zero_grad()
            loss = negative_log_likelihood(temperature_from(raw))
            loss.backward()
            return loss

        optimizer.step(closure)
        fitted = float(temperature_from(raw).item())

    after = float(negative_log_likelihood(torch.tensor(fitted)).item())
    # The sigmoid never reaches the bound, so a saturated fit reports a value a
    # hair inside it. Treat the outer 0.1% of the range as at_bound: a fitted
    # value that close to the edge is an alarm, not a result.
    tolerance = 1e-3 * (high - low)
    return TemperatureFit(
        temperature=fitted,
        samples=rows,
        nll_before=before,
        nll_after=after,
        fallback=False,
        at_bound=fitted <= low + tolerance or fitted >= high - tolerance,
    )


@dataclass(slots=True)
class TemperatureMap:
    """Per-type and per-(type, option bucket) temperatures with fallbacks."""

    default: float = 1.0
    by_type: dict[str, float] = field(default_factory=dict)
    by_bucket: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in self._entries():
            if not value > 0:
                raise ValueError(f"temperature for {name} must be positive")

    def temperature(self, kind: str, option_count: int) -> float:
        """Bucket first, then type, then default."""
        cell = self.by_bucket.get(bucket_key(kind, option_count))
        if cell is not None:
            return cell
        return self.by_type.get(kind, self.default)

    def temperatures(self, kinds: Sequence[str], option_counts: Sequence[int]) -> list[float]:
        if len(kinds) != len(option_counts):
            raise ValueError("kinds and option_counts must have the same length")
        return [
            self.temperature(kind, count) for kind, count in zip(kinds, option_counts, strict=True)
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "default": self.default,
            "by_type": dict(self.by_type),
            "by_bucket": dict(self.by_bucket),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TemperatureMap:
        by_type = {str(key): float(value) for key, value in dict(data.get("by_type", {})).items()}
        by_bucket = {
            str(key): float(value) for key, value in dict(data.get("by_bucket", {})).items()
        }
        return cls(default=float(data.get("default", 1.0)), by_type=by_type, by_bucket=by_bucket)

    def _entries(self):
        yield "default", self.default
        for key, value in self.by_type.items():
            yield f"type:{key}", value
        for key, value in self.by_bucket.items():
            yield f"bucket:{key}", value


def fit_temperature_map(
    logits: Tensor,
    target: Tensor,
    option_mask: Tensor,
    kinds: Sequence[str],
    option_counts: Sequence[int],
    *,
    by_type: bool = True,
    by_bucket: bool = True,
    min_samples: int = 64,
    bounds: tuple[float, float] = DEFAULT_BOUNDS,
    max_iter: int = 100,
    learning_rate: float = 0.1,
) -> tuple[TemperatureMap, dict[str, TemperatureFit]]:
    """Fit a whole temperature map, skipping cells without enough held-out rows.

    A cell that falls back is left out of the map, so lookup falls through to the
    per-type value and then to the default. Nothing is invented from too little
    data.
    """
    rows = logits.size(0)
    if len(kinds) != rows or len(option_counts) != rows:
        raise ValueError("kinds and option_counts must have one entry per row")

    def fit(selector: list[int]) -> TemperatureFit:
        index = torch.tensor(selector, dtype=torch.long)
        return fit_temperature(
            logits[index],
            target[index],
            option_mask[index],
            min_samples=min_samples,
            bounds=bounds,
            max_iter=max_iter,
            learning_rate=learning_rate,
        )

    result = TemperatureMap(default=1.0)
    fits: dict[str, TemperatureFit] = {}
    if by_type:
        for kind in sorted(set(kinds)):
            selector = [index for index, value in enumerate(kinds) if value == kind]
            cell = fit(selector)
            fits[f"type:{kind}"] = cell
            if not cell.fallback:
                result.by_type[kind] = cell.temperature
    if by_bucket:
        cells: dict[str, list[int]] = {}
        for index, (kind, count) in enumerate(zip(kinds, option_counts, strict=True)):
            cells.setdefault(bucket_key(kind, count), []).append(index)
        for key, selector in sorted(cells.items()):
            cell = fit(selector)
            fits[f"bucket:{key}"] = cell
            if not cell.fallback:
                result.by_bucket[key] = cell.temperature
    return result, fits
