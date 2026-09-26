"""A decoder backbone that reads one sequence per option. Experimental.

The encoder path puts every option in one sequence and reads each at its marker.
A causal decoder cannot do that: a marker does not see the option text after it.
So this path builds one sequence per option and reads the hidden state of the
last token::

    State: <state, truncated from the right>
    <type prefix>
    Question: <instruction>
    Candidate: <option k>
    Decision:<EOS>

The state and the question come first on purpose. Under causal attention that
prefix is identical in the K sequences of a question, which is what makes it
shareable later. A shared head (``LayerNorm`` then ``Linear(hidden, 1)``) turns
each read into one logit, and the softmax runs across the options of a question
exactly as in the encoder path. Everything after the logits (the reward, the
temperatures, the metrics) is shared.

Two properties follow from the layout:

- Options never see each other, so the answer cannot depend on option order. A
  permutation of the options permutes the distribution, exactly.
- The cost is K sequences per question, in training and at inference.

The question part of a sequence (everything but the state) has to fit
``header_budget``; a question that does not fit raises instead of being cut.
Literal special-token strings are stripped from every untrusted text, so input
cannot plant an end-of-sequence token. The read position is the builder's own
last token either way, never a token found by its id.

The layout, the head and its initialisation are those of the probe the numbers
in ``BENCHMARKS.md`` were measured with, so the probe's weights load into this
model and give the same answers. See ``docs/checkpoints.md``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch
from torch import Tensor, nn
from transformers import AutoModel

from .model import (
    DEFAULT_DECODER,
    DecisionOutput,
    ExuConfig,
    _distribution_features,
    masked_softmax,
)
from .sequence import SequenceConfig
from .types import DecisionQuestion

if TYPE_CHECKING:
    from transformers import PreTrainedModel


class DecoderTokenizer(Protocol):
    """Minimum Hugging Face tokenizer surface needed by :class:`DecoderSequenceBuilder`."""

    eos_token_id: int | None
    pad_token_id: int | None

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...


@dataclass(frozen=True, slots=True)
class EncodedOptions:
    """One question encoded as one decoder sequence per option."""

    sequences: tuple[tuple[int, ...], ...]
    question_type_id: int

    @property
    def option_count(self) -> int:
        return len(self.sequences)


class DecoderSequenceBuilder:
    """Build the K sequences of a question for :class:`DecoderDecisionModel`."""

    def __init__(self, tokenizer: DecoderTokenizer, config: SequenceConfig | None = None) -> None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer must provide eos_token_id")
        self.tokenizer = tokenizer
        self.config = config or SequenceConfig()
        self._special = tuple(
            token for token in getattr(tokenizer, "all_special_tokens", ()) if token
        )
        self._lead = self._encode("State: ")

    def build(self, state: str, question: DecisionQuestion) -> EncodedOptions:
        kind = self.config.type_prefixes[question.kind.value]
        instruction = self._clean(question.instruction)
        tails = [
            self._encode(
                f"\n{kind}\nQuestion: {instruction}\n"
                f"Candidate: {self._clean(option.render())}\nDecision:"
            )
            for option in question.options
        ]
        if not tails:
            raise ValueError("a question needs at least one option")
        # The lead, the longest tail and the closing EOS: everything but the state.
        longest = len(self._lead) + max(len(tail) for tail in tails) + 1
        if longest > self.config.header_budget:
            raise ValueError(
                f"question takes {longest} tokens without the state, more than the "
                f"configured header_budget of {self.config.header_budget}"
            )
        # One cut for all K sequences, set by the longest tail: every option is
        # judged on the same state, and the prefix stays identical across them.
        room = self.config.max_length - longest
        prefix = (*self._lead, *self._encode(self._clean(state))[:room])
        eos = self.tokenizer.eos_token_id
        return EncodedOptions(
            sequences=tuple((*prefix, *tail, eos) for tail in tails),
            question_type_id=self.config.type_id(question.kind.value),
        )

    def tensors(
        self,
        encoded: Sequence[EncodedOptions],
        device: torch.device | str | None = None,
    ) -> dict[str, Tensor]:
        """The keyword arguments of :meth:`DecoderDecisionModel.forward` for a batch.

        Sequences are padded on the right, so the last real token of each sits at
        its own length minus one and padding never reaches it through causal
        attention. ``sequence_rows`` and ``sequence_slots`` say which question and
        which option every sequence is.
        """
        if not encoded:
            raise ValueError("cannot pad an empty batch")
        sequences = [ids for item in encoded for ids in item.sequences]
        length = max(len(ids) for ids in sequences)
        width = max(item.option_count for item in encoded)
        pad = self.tokenizer.pad_token_id
        pad_id = pad if pad is not None else self.tokenizer.eos_token_id
        input_ids = torch.full((len(sequences), length), pad_id, dtype=torch.long)
        attention = torch.zeros((len(sequences), length), dtype=torch.long)
        for index, ids in enumerate(sequences):
            input_ids[index, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention[index, : len(ids)] = 1
        rows = [row for row, item in enumerate(encoded) for _ in item.sequences]
        slots = [slot for item in encoded for slot in range(item.option_count)]
        option_mask = torch.zeros((len(encoded), width), dtype=torch.bool)
        for row, item in enumerate(encoded):
            option_mask[row, : item.option_count] = True
        return {
            "input_ids": input_ids.to(device),
            "attention_mask": attention.to(device),
            "read_positions": torch.tensor(
                [len(ids) - 1 for ids in sequences], dtype=torch.long, device=device
            ),
            "sequence_rows": torch.tensor(rows, dtype=torch.long, device=device),
            "sequence_slots": torch.tensor(slots, dtype=torch.long, device=device),
            "option_mask": option_mask.to(device),
        }

    def _clean(self, value: str) -> str:
        """Strip literal special-token strings, repeatedly, as the encoder path does."""
        if not isinstance(value, str):
            raise TypeError("state and question text must be strings")
        cleaned = value
        while any(token in cleaned for token in self._special):
            for token in self._special:
                cleaned = cleaned.replace(token, "")
        return cleaned

    def _encode(self, value: str) -> list[int]:
        return self.tokenizer.encode(value, add_special_tokens=False)


class DecoderDecisionModel(nn.Module):
    """A causal decoder read at the last token of each option's sequence."""

    def __init__(self, backbone: PreTrainedModel, config: ExuConfig) -> None:
        super().__init__()
        if config.backbone != "decoder":
            raise ValueError("DecoderDecisionModel needs a config with backbone='decoder'")
        self.config = config
        self.backbone = backbone
        hidden_size = backbone.config.hidden_size
        self.norm = nn.LayerNorm(hidden_size)
        self.score = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.score.weight, std=0.02)
        nn.init.zeros_(self.score.bias)
        self.action_head = nn.Sequential(
            nn.LayerNorm(hidden_size + 4),
            nn.Linear(hidden_size + 4, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 2),
        )

    @classmethod
    def from_pretrained(cls, config: ExuConfig | None = None) -> DecoderDecisionModel:
        """Load base decoder weights, without its language-model head."""
        settings = config or ExuConfig(encoder_name=DEFAULT_DECODER, backbone="decoder")
        backbone = AutoModel.from_pretrained(settings.encoder_name, dtype=torch.float32)
        return cls(backbone, settings)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        read_positions: Tensor,
        sequence_rows: Tensor,
        sequence_slots: Tensor,
        option_mask: Tensor,
        temperature: float | Tensor | None = None,
    ) -> DecisionOutput:
        """Score a batch emitted by :meth:`DecoderSequenceBuilder.tensors`."""
        hidden = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).last_hidden_state
        read = hidden[torch.arange(hidden.size(0), device=hidden.device), read_positions].float()
        scalar = self.score(self.norm(read)).squeeze(-1)
        rows, width = option_mask.shape
        empty = scalar.new_full((rows, width), torch.finfo(scalar.dtype).min)
        logits = empty.index_put((sequence_rows, sequence_slots), scalar)
        mask = option_mask.bool()
        masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        chosen = self.config.temperature if temperature is None else temperature
        probabilities = masked_softmax(masked_logits, mask, chosen)

        # The action head reads the mean of a question's K reads, where the encoder
        # path reads its single [CLS]. It is trained only when action labels exist.
        pooled = read.new_zeros((rows, read.size(-1))).index_add(0, sequence_rows, read)
        pooled = pooled / mask.sum(dim=-1, keepdim=True).clamp_min(1).to(pooled.dtype)
        features = _distribution_features(probabilities, mask)
        action_logits = self.action_head(torch.cat((pooled, features.detach().float()), dim=-1))
        return DecisionOutput(masked_logits, probabilities, action_logits)
