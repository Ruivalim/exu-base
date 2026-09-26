"""Pick the builder and the model that belong to a configuration.

Two architectures share everything after the logits. They differ in how a
question becomes token ids and where the model reads each option:

- ``encoder`` with ``markers``: one sequence per question, read at each option's
  marker. :class:`~exu.model.ExuModel` and :class:`~exu.sequence.SequenceBuilder`.
- ``decoder`` with ``per-option``: one sequence per option, read at its last
  token. :class:`~exu.decoder.DecoderDecisionModel` and
  :class:`~exu.decoder.DecoderSequenceBuilder`. Experimental.

The pair is a property of the configuration, and so of the checkpoint, never of
the caller: the evaluation and inference entry points read it and never take a flag
for it, so a checkpoint cannot be run with the wrong builder.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from transformers import AutoModel

from .decoder import DecoderDecisionModel, DecoderSequenceBuilder
from .model import ExuConfig, ExuModel
from .sequence import DecisionBuilder, SequenceBuilder, SequenceConfig

if TYPE_CHECKING:
    from transformers import PreTrainedModel

DecisionModel = ExuModel | DecoderDecisionModel


def build_model(backbone: PreTrainedModel, config: ExuConfig) -> DecisionModel:
    """Put the decision head of ``config``'s architecture on a loaded backbone."""
    if config.backbone == "decoder":
        return DecoderDecisionModel(backbone, config)
    return ExuModel(backbone, config)


def model_from_pretrained(config: ExuConfig) -> DecisionModel:
    """Download or read the pretrained backbone, then add an untrained head.

    The backbone loads in float32 whatever dtype it was published in. transformers
    5 keeps a checkpoint's own dtype by default, and Qwen3 ships in bfloat16:
    training those as master weights rounds most updates away at a 1e-5 learning
    rate. Precision during a forward pass is autocast's job, not the stored weights'.
    """
    backbone = AutoModel.from_pretrained(config.encoder_name, dtype=torch.float32)
    return build_model(backbone, config)


def builder_for(tokenizer: Any, sequence: SequenceConfig, config: ExuConfig) -> DecisionBuilder:
    """The sequence builder whose batches ``config``'s model reads."""
    if config.backbone == "decoder":
        return DecoderSequenceBuilder(tokenizer, sequence)
    return SequenceBuilder(tokenizer, sequence)
