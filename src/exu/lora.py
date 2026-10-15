"""Low-rank adapters on a backbone's linear layers, merged back before saving.

``exu-train --lora-rank R`` freezes the backbone and adds, next to every
``nn.Linear`` in it, a trainable update ``B @ A`` of rank ``R`` scaled by
``alpha / R``. ``B`` starts at zero, so the first forward pass is the base model's.
The decision head stays fully trainable.

After training the update is added into each layer's weight and the wrappers are
removed, so a checkpoint trained with adapters has the same keys and files as one
trained without them, and evaluation and inference do not know the difference.

On a GPU the frozen weights of the adapted layers can be stored in the autocast
dtype (``base_dtype``): autocast runs those matmuls in that dtype anyway, so the
forward pass is the same, and a 2B base then fits on a 12 GB card. The merge
happens in float32 all the same: added in bfloat16, most of a small update would
round away.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class LoRALinear(nn.Module):
    """A frozen ``nn.Linear`` plus a trainable low-rank update."""

    def __init__(
        self, base: nn.Linear, rank: int, alpha: float, base_dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be at least 1")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.base = base if base_dtype is None else base.to(base_dtype)
        self.base.requires_grad_(False)
        self.rank = rank
        self.scaling = alpha / rank
        factory = {"device": base.weight.device, "dtype": torch.float32}
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, **factory))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, **factory))
        # The initialisation of nn.Linear and of the reference LoRA implementation.
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, inputs: Tensor) -> Tensor:
        update = (inputs.to(self.lora_a.dtype) @ self.lora_a.T) @ self.lora_b.T
        stored = self.base.weight.dtype
        if inputs.dtype != stored and not torch.is_autocast_enabled(inputs.device.type):
            # A lower-precision base outside autocast: compute in its dtype, return in ours.
            base = self.base(inputs.to(stored)).to(inputs.dtype)
        else:
            base = self.base(inputs)
        return base + (update * self.scaling).to(base.dtype)

    def merged(self) -> nn.Linear:
        """A plain ``nn.Linear`` whose weight carries the update, in float32."""
        with torch.autocast(device_type=self.base.weight.device.type, enabled=False):
            update = self.lora_b.detach().float() @ self.lora_a.detach().float()
            weight = self.base.weight.detach().float() + update * self.scaling
        bias = self.base.bias is not None
        layer = nn.Linear(
            self.base.in_features,
            self.base.out_features,
            bias=bias,
            device=weight.device,
            dtype=torch.float32,
        )
        with torch.no_grad():
            layer.weight.copy_(weight)
            if bias:
                layer.bias.copy_(self.base.bias.detach().float())
        return layer


def apply_lora(
    module: nn.Module, rank: int, alpha: float, base_dtype: torch.dtype | None = None
) -> int:
    """Freeze ``module`` and wrap each of its ``nn.Linear`` layers. Returns the count.

    Raises when there is nothing to wrap: a backbone with no linear layer would
    otherwise train nothing at all and look like a run that did.
    A root ``nn.Linear`` is refused; wrap it with ``LoRALinear`` directly.
    """
    if isinstance(module, nn.Linear):
        raise ValueError("apply_lora requires a parent module; use LoRALinear for a single layer")
    if any(isinstance(child, LoRALinear) for child in module.modules()):
        raise ValueError("module already has adapters")
    module.requires_grad_(False)
    targets = [
        (name, child) for name, child in module.named_modules() if isinstance(child, nn.Linear)
    ]
    if not targets:
        raise ValueError("no nn.Linear layer to adapt in this backbone")
    for name, child in targets:
        _replace(module, name, LoRALinear(child, rank, alpha, base_dtype))
    return len(targets)


def merge_lora(module: nn.Module) -> int:
    """Fold every adapter in ``module`` into its layer and drop it. Returns the count.

    A root ``LoRALinear`` is refused; use its ``merged()`` method directly.
    """
    if isinstance(module, LoRALinear):
        raise ValueError("merge_lora requires a parent module; use merged() for a single adapter")
    wrapped = [
        (name, child) for name, child in module.named_modules() if isinstance(child, LoRALinear)
    ]
    for name, child in wrapped:
        _replace(module, name, child.merged())
    if wrapped:
        module.float()
        module.requires_grad_(True)
    return len(wrapped)


def _replace(root: nn.Module, name: str, replacement: nn.Module) -> None:
    parent_name, _, attribute = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, attribute, replacement)
