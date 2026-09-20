from __future__ import annotations

import pytest
import torch

from exu.devices import autocast_context, resolve_device, resolve_dtype


def test_resolve_device_returns_cpu_for_cpu() -> None:
    assert resolve_device("cpu").type == "cpu"


def test_resolve_device_rejects_an_unknown_choice() -> None:
    with pytest.raises(ValueError, match="device must be one of"):
        resolve_device("tpu")


def test_resolve_device_fails_when_cuda_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(ValueError, match="CUDA was requested"):
        resolve_device("cuda")


def test_cpu_precision_is_float32() -> None:
    assert resolve_dtype("cpu") == torch.float32


def test_autocast_is_a_no_op_outside_cuda() -> None:
    with autocast_context("cpu"):
        value = torch.ones(1)
    assert value.dtype == torch.float32
