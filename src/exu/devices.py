"""Device and precision selection.

`auto` takes the best device available, in order CUDA, MPS, CPU. A device asked
for by name that is not there raises rather than falling back silently, because
a caller that pinned a device wants to know. The dtype is picked from the device:
bfloat16 on CUDA capability 8 or more (Ampere and later), float16 below that,
float32 everywhere else. Autocast only wraps CUDA forwards, and the checkpoint
itself always loads in float32: precision here is how a forward pass is computed,
not what is stored on disk.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext

import torch

_DEVICE_CHOICES = ("auto", "cpu", "cuda", "mps")


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve a requested device, failing loudly when it is unavailable."""
    if requested not in _DEVICE_CHOICES:
        raise ValueError(f"device must be one of {_DEVICE_CHOICES}")
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    return torch.device(requested)


def resolve_dtype(device: torch.device | str) -> torch.dtype:
    """Pick a load precision for the device, never bf16 on old or non-CUDA hardware."""
    resolved = torch.device(device)
    if resolved.type != "cuda":
        return torch.float32
    major, _minor = torch.cuda.get_device_capability(resolved)
    return torch.bfloat16 if major >= 8 else torch.float16


@contextmanager
def autocast_context(device: torch.device | str) -> Iterator[None]:
    """Autocast on CUDA only; a no-op everywhere else."""
    resolved = torch.device(device)
    if resolved.type != "cuda":
        with nullcontext():
            yield
        return
    with torch.autocast(device_type="cuda", dtype=resolve_dtype(resolved)):
        yield
