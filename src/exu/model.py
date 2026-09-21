"""Bidirectional encoder plus a decision head.

The model reads option *text*, it does not have a fixed output layer. That is
what makes a new task a new question instead of a retraining run: the labels
live in the sequence, not in the weights.

Architecture, from the encoder up:

1. a bidirectional encoder, fully fine-tuned with a lower learning rate;
2. a question-type embedding added to every position (three types);
3. two extra transformer layers so markers can talk to each other, already
   conditioned on the type;
4. read the hidden state at each option's ``[MASK]`` marker;
5. a small scorer turns each marker into one logit;
6. softmax over the valid options of each question, divided by the calibration
   temperature.

The optional ``marker-cls`` scorer adds one term to every option's logit: the dot
product of a readout vector built from the option's own *text* and the normalised
``[CLS]`` state. It exists because the marker scorer starts from a symmetric
saddle. The scorer is shared, so at the start every option gets nearly the same
logit, and the gradient that reaches anything the options have in common is
``sum_k (q_k - y_k) * c = 0``. Something has to make the options distinguishable
first. A skewed label prior does it (there is an immediate reward for scoring the
frequent option higher), and so does lexical overlap between an option and the
state. With balanced labels and no such overlap nothing does, and the model stays
at the uniform guess for as long as it is trained. Measured: MultiNLI with
balanced classes stays at 0.32 accuracy and reaches 0.69 when only the training
labels are skewed, and BoolQ learns at its natural 62% of yes and collapses to
``log 2`` when the training split is balanced. ``marker-cls`` breaks the symmetry
by construction, and learns in all four cases. See `BENCHMARKS.md`.

There is also an action head: CLS plus four distribution statistics (max
probability, top-two margin, normalized entropy, option fraction) predicting
whether to act or escalate. It is trained only if you supply action labels;
:meth:`ActionCosts.should_act` is the trivial baseline it has to beat.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from transformers import AutoModel

if TYPE_CHECKING:
    from transformers import PreTrainedModel

DEFAULT_ENCODER = "google-bert/bert-base-multilingual-cased"
SCORERS = ("marker", "marker-cls")


def masked_softmax(
    logits: Tensor, option_mask: Tensor, temperature: float | Tensor = 1.0
) -> Tensor:
    """Softmax over valid options only, then re-zero the padded ones.

    ``temperature`` may be a scalar or one value per row. Padded logits arrive at
    the dtype minimum, so they are replaced with zero *before* dividing by the
    temperature: dividing the sentinel would overflow to ``-inf``, and the
    softmax backward pass would then compute ``0 * inf = NaN`` and poison every
    gradient. Masking to negative infinity after the division severs the graph
    for those positions instead.
    """
    mask = option_mask.bool()
    safe = logits.masked_fill(~mask, 0.0)
    if isinstance(temperature, Tensor):
        scaled = safe / temperature.reshape(-1, 1)
    else:
        scaled = safe / temperature
    scaled = scaled.masked_fill(~mask, float("-inf"))
    probabilities = torch.softmax(scaled, dim=-1)
    return probabilities.masked_fill(~mask, 0.0)


def masked_log_softmax(
    logits: Tensor, option_mask: Tensor, temperature: float | Tensor = 1.0
) -> Tensor:
    """Log-probabilities over valid options only, in float32, zero on padded ones.

    The log score is computed from these and never from ``log(masked_softmax(z))``.
    Softmax underflows to an exact zero on a confident miss, and the log of that is
    infinite unless it is clamped. A clamp makes the reward improper below it and
    switches off the gradient of exactly the rows that most need one.

    The upcast happens here instead of being left to the caller's autocast
    context, because half precision cannot hold a confident miss. Padded positions
    come back as zero, not ``-inf``, so ``target * log_probabilities`` never meets
    ``0 * -inf``. A non-finite logit on a valid option is a bug upstream and is not
    hidden: it comes back non-finite.
    """
    mask = option_mask.bool()
    if not bool(mask.any(dim=-1).all()):
        raise ValueError("every row needs at least one valid option")
    safe = logits.float().masked_fill(~mask, 0.0)
    if isinstance(temperature, Tensor):
        scaled = safe / temperature.reshape(-1, 1)
    else:
        scaled = safe / temperature
    scaled = scaled.masked_fill(~mask, float("-inf"))
    return torch.log_softmax(scaled, dim=-1).masked_fill(~mask, 0.0)


def option_text_embeddings(
    token_embeddings: Tensor, input_ids: Tensor, marker_positions: Tensor
) -> Tensor:
    """Mean embedding of each option's own text tokens, shape ``(batch, options, hidden)``.

    An option is laid out as ``[MASK] text [SEP]``, so its text runs from the token
    after its marker to the separator that closes it. The separator id is read off
    the sequence itself, as the token that closes the first option: a question has
    at least two options, and this keeps the model free of any tokenizer.

    Pass *input* embeddings. They are the same vector in every example, which is
    what makes them an identity for the option. The contextual state at the marker
    changes with the example and was measured not to work for this.

    Rows of padded options come back with whatever their placeholder marker points
    at. The caller masks them.
    """
    steps = torch.arange(input_ids.size(1), device=input_ids.device).view(1, 1, -1)
    after = steps > marker_positions.unsqueeze(-1)
    separator = input_ids.gather(1, (marker_positions[:, 1:2] - 1).clamp_min(0))
    closes = (input_ids.unsqueeze(1) == separator.unsqueeze(-1)) & after
    closing = closes.to(torch.int8).argmax(dim=-1)
    # Prefix sums keep the memory at (batch, length, hidden) however many options
    # there are, where a (batch, options, length, hidden) mask would not.
    running = token_embeddings.float().cumsum(dim=1)
    width = running.size(-1)
    upto_closing = running.gather(1, (closing - 1).clamp_min(0).unsqueeze(-1).expand(-1, -1, width))
    upto_marker = running.gather(1, marker_positions.unsqueeze(-1).expand(-1, -1, width))
    length = (closing - marker_positions - 1).clamp_min(1).unsqueeze(-1)
    return (upto_closing - upto_marker) / length


def centre_options(identity: Tensor, option_mask: Tensor) -> Tensor:
    """Subtract what the valid options of a question share, then rescale.

    Whatever is common to all options of a question produces the same logit
    everywhere and cancels in the softmax. Options usually share a lot (the same
    words in their descriptions, the same average embedding), so without this step
    the readout vectors are nearly equal and the symmetry this scorer exists to
    break is still there. Measured on MultiNLI: 0.32 accuracy without it, 0.74
    with it.
    """
    weight = option_mask.unsqueeze(-1).to(identity.dtype)
    mean = (identity * weight).sum(dim=1, keepdim=True) / weight.sum(dim=1, keepdim=True)
    return torch.nn.functional.layer_norm((identity - mean) * weight, identity.shape[-1:])


@dataclass(frozen=True, slots=True)
class ActionCosts:
    """Cost matrix for act-or-escalate, expressed as a business rule.

    Acting and being right earns ``gain``. Acting and being wrong costs ``loss``.
    Escalating to a human costs ``escalate``. The break-even probability follows
    from the three: acting pays only above ``threshold``.
    """

    gain: float = 1.0
    loss: float = 3.0
    escalate: float = 0.5

    def __post_init__(self) -> None:
        if self.gain <= 0:
            raise ValueError("gain must be positive")
        if self.loss < 0:
            raise ValueError("loss cannot be negative")
        if self.escalate < 0:
            raise ValueError("escalate cost cannot be negative")

    @property
    def threshold(self) -> float:
        """Probability above which acting beats escalating."""
        return (self.loss - self.escalate) / (self.gain + self.loss)

    def should_act(self, probability: float | Tensor) -> bool | Tensor:
        """Recommend acting when the calibrated probability clears the threshold."""
        return probability >= self.threshold

    def to_dict(self) -> dict[str, float]:
        return {"gain": self.gain, "loss": self.loss, "escalate": self.escalate}

    @classmethod
    def from_dict(cls, data: Mapping[str, float]) -> ActionCosts:
        return cls(
            gain=float(data.get("gain", 1.0)),
            loss=float(data.get("loss", 3.0)),
            escalate=float(data.get("escalate", 0.5)),
        )


@dataclass(frozen=True, slots=True)
class ExuConfig:
    """Architecture choices independent of a specific encoder checkpoint."""

    encoder_name: str = DEFAULT_ENCODER
    num_decision_layers: int = 2
    dropout: float = 0.1
    temperature: float = 1.0
    action_costs: ActionCosts = field(default_factory=ActionCosts)
    scorer: str = "marker"

    def __post_init__(self) -> None:
        if self.num_decision_layers < 1:
            raise ValueError("num_decision_layers must be positive")
        if self.scorer not in SCORERS:
            raise ValueError(f"scorer must be one of {SCORERS}")
        if not self.temperature > 0:
            raise ValueError("temperature must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict[str, object]:
        return {
            "encoder_name": self.encoder_name,
            "num_decision_layers": self.num_decision_layers,
            "dropout": self.dropout,
            "temperature": self.temperature,
            "action_costs": self.action_costs.to_dict(),
            "scorer": self.scorer,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ExuConfig:
        raw_costs = data.get("action_costs", {})
        if not isinstance(raw_costs, Mapping):
            raise ValueError("action_costs must be an object")
        return cls(
            encoder_name=str(data.get("encoder_name", DEFAULT_ENCODER)),
            num_decision_layers=int(data.get("num_decision_layers", 2)),  # type: ignore[arg-type]
            dropout=float(data.get("dropout", 0.1)),  # type: ignore[arg-type]
            temperature=float(data.get("temperature", 1.0)),  # type: ignore[arg-type]
            action_costs=ActionCosts.from_dict(raw_costs),  # type: ignore[arg-type]
            # Checkpoints written before the option existed read at the markers only.
            scorer=str(data.get("scorer", "marker")),
        )


@dataclass(frozen=True, slots=True)
class DecisionOutput:
    """Raw logits, normalized probabilities and act-or-escalate logits."""

    logits: Tensor
    probabilities: Tensor
    action_logits: Tensor


class ExuModel(nn.Module):
    """Encoder-only typed-decision model."""

    def __init__(self, encoder: PreTrainedModel, config: ExuConfig | None = None) -> None:
        super().__init__()
        self.config = config or ExuConfig()
        self.encoder = encoder
        hidden_size = encoder.config.hidden_size
        if hidden_size >= 64 and hidden_size % 64:
            raise ValueError(
                "encoder hidden size must be a multiple of 64 for 64-dimensional attention heads"
            )
        heads = max(1, hidden_size // 64)

        self.question_type_embedding = nn.Embedding(3, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=heads,
            dim_feedforward=hidden_size * 4,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decision_layers = nn.TransformerEncoder(
            layer,
            num_layers=self.config.num_decision_layers,
            enable_nested_tensor=False,
        )
        self.scorer = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        if self.config.scorer == "marker-cls":
            self.option_readout = nn.Linear(hidden_size, hidden_size, bias=False)
            self.summary_norm = nn.LayerNorm(hidden_size)
        self.action_head = nn.Sequential(
            nn.LayerNorm(hidden_size + 4),
            nn.Linear(hidden_size + 4, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 2),
        )

    @classmethod
    def from_pretrained(cls, config: ExuConfig | None = None) -> ExuModel:
        """Load base encoder weights. The decision head starts untrained."""
        settings = config or ExuConfig()
        return cls(AutoModel.from_pretrained(settings.encoder_name), settings)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        marker_positions: Tensor,
        option_mask: Tensor,
        question_type_ids: Tensor,
        temperature: float | Tensor | None = None,
    ) -> DecisionOutput:
        """Score a batch emitted by :meth:`exu.sequence.SequenceBuilder.pad`."""
        encoder_inputs: dict[str, Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if getattr(self.encoder.config, "type_vocab_size", 0) > 1:
            encoder_inputs["token_type_ids"] = torch.zeros_like(input_ids)
        hidden = self.encoder(**encoder_inputs).last_hidden_state
        hidden = hidden + self.question_type_embedding(question_type_ids).unsqueeze(1)
        hidden = self.decision_layers(hidden, src_key_padding_mask=attention_mask.eq(0))

        indices = marker_positions.unsqueeze(-1).expand(-1, -1, hidden.size(-1))
        marker_hidden = hidden.gather(1, indices)
        logits = self.scorer(marker_hidden).squeeze(-1)
        if self.config.scorer == "marker-cls":
            words = self.encoder.get_input_embeddings()(input_ids)
            identity = option_text_embeddings(words, input_ids, marker_positions)
            readout = self.option_readout(centre_options(identity, option_mask.bool()))
            summary = self.summary_norm(hidden[:, 0].float()).unsqueeze(1)
            logits = logits + (readout * summary).sum(dim=-1) / hidden.size(-1) ** 0.5
        masked_logits = logits.masked_fill(~option_mask.bool(), torch.finfo(logits.dtype).min)
        chosen = self.config.temperature if temperature is None else temperature
        probabilities = masked_softmax(masked_logits, option_mask, chosen)

        features = _distribution_features(probabilities, option_mask)
        action_logits = self.action_head(torch.cat((hidden[:, 0], features.detach()), dim=-1))
        return DecisionOutput(masked_logits, probabilities, action_logits)


def _distribution_features(probabilities: Tensor, option_mask: Tensor) -> Tensor:
    """Four gradient-free statistics of each distribution, after temperature."""
    valid_options = option_mask.sum(dim=-1).clamp_min(1).to(probabilities.dtype)
    max_probability = probabilities.max(dim=-1).values
    top_two = probabilities.topk(k=min(2, probabilities.size(-1)), dim=-1).values
    margin = top_two[:, 0] - (top_two[:, 1] if top_two.size(-1) == 2 else 0.0)
    safe = probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny)
    entropy = -(safe * safe.log()).sum(dim=-1)
    # Divide by log(K), not by 1 for every small K: with the old floor a noul
    # question capped this feature at 0.69 while the runtime's entropy_confidence
    # normalised it properly, so the two disagreed for two-option questions.
    normalized_entropy = entropy / valid_options.clamp_min(2.0).log()
    option_fraction = valid_options / 255.0
    return torch.stack((max_probability, margin, normalized_entropy, option_fraction), dim=-1)
