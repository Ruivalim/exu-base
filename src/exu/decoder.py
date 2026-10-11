"""A decoder backbone that reads one sequence per option. Experimental.

The encoder path puts every option in one sequence and reads each at its marker.
A causal decoder cannot do that: a marker does not see the option text after it.
So this path builds one sequence per option and reads the hidden state of the
last token::

    [Context: <context>
    ]State: <state, truncated from the right>
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
- The cost is K sequences per question in training. Without gradients (evaluation,
  calibration, the runtime) each distinct prefix runs once, so the questions about
  one state share one run of it, and the tails run on its cache. Under causal
  attention that is the same computation; on Qwen3.5-2B the probabilities moved by
  at most 0.0006 over 2,000 questions, from where the linear-attention kernels cut
  their chunks, and no answer changed.

The question part of a sequence (everything but the context and the state) has to
fit ``header_budget``, and the optional context has to fit ``context_budget``;
either one that does not fit raises instead of being cut. Without a context the
sequence is the same as it always was.
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
from .sequence import SequenceConfig, context_ids
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
    prefix_length: int
    """Leading tokens every sequence shares: the lead and the state."""

    def __post_init__(self) -> None:
        if not self.sequences:
            raise ValueError("a question needs at least one option")
        prefix = self.sequences[0][: self.prefix_length]
        if self.prefix_length < 1 or any(
            len(ids) <= self.prefix_length or ids[: self.prefix_length] != prefix
            for ids in self.sequences
        ):
            raise ValueError("every sequence must start with the same prefix and extend past it")

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
        self._context_lead = self._encode("Context: ")
        self._context_end = self._encode("\nState: ")

    def build(
        self, state: str, question: DecisionQuestion, context: str | None = None
    ) -> EncodedOptions:
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
        lead = self._lead
        ids = context_ids(
            self._encode(self._clean(context)) if context is not None else None,
            self.config.context_budget,
        )
        if ids:
            lead = [*self._context_lead, *ids, *self._context_end]
            longest += len(lead) - len(self._lead)
        room = self.config.max_length - longest
        if room < 0:
            raise ValueError("the context and the question leave no room in max_length")
        prefix = (*lead, *self._encode(self._clean(state))[:room])
        eos = self.tokenizer.eos_token_id
        return EncodedOptions(
            sequences=tuple((*prefix, *tail, eos) for tail in tails),
            question_type_id=self.config.type_id(question.kind.value),
            prefix_length=len(prefix),
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
        which option every sequence is, and ``prefix_lengths`` how many leading
        tokens the sequences of each question share.
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
            "prefix_lengths": torch.tensor(
                [item.prefix_length for item in encoded], dtype=torch.long, device=device
            ),
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
        # A vision-language base (Qwen3.5) keeps its text sizes in `text_config`.
        hidden_size = backbone.config.get_text_config().hidden_size
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
        prefix_lengths: Tensor | None = None,
    ) -> DecisionOutput:
        """Score a batch emitted by :meth:`DecoderSequenceBuilder.tensors`.

        Without gradients and with ``prefix_lengths``, each distinct prefix runs
        once and the tails of every question that shares it run on its cache. With
        gradients, or on a backbone in training mode with gradient checkpointing
        (which drops any cache it is given), the K full sequences run.
        """
        if prefix_lengths is not None and self._can_share_prefix():
            read = self._read_shared(
                input_ids, attention_mask, read_positions, sequence_rows, prefix_lengths
            )
        else:
            hidden = self.backbone(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            ).last_hidden_state
            rows_index = torch.arange(hidden.size(0), device=hidden.device)
            read = hidden[rows_index, read_positions].float()
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

    def _can_share_prefix(self) -> bool:
        # Training runs the full sequences: without gradient checkpointing, which
        # drops the cache, the shared prefix needed more memory than K checkpointed
        # sequences and ran slower on Qwen3.5-2B.
        if torch.is_grad_enabled():
            return False
        checkpointing = getattr(self.backbone, "is_gradient_checkpointing", False)
        return not (self.backbone.training and checkpointing)

    def _read_shared(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        read_positions: Tensor,
        sequence_rows: Tensor,
        prefix_lengths: Tensor,
    ) -> Tensor:
        """The read of every sequence, running each distinct prefix once.

        Questions about the same state usually share their prefix token for token,
        so they share one run of it. The prefix runs alone, its cache is copied once
        per sequence that continues it, and those tails, padded on the right, run
        on it in one call. The cache's own ``reorder_cache`` does the copy, so the
        linear-attention state of a hybrid base such as Qwen3.5 is carried as well
        as the keys and values of its attention layers.
        """
        groups: dict[tuple[int, ...], list[int]] = {}
        for row, length in enumerate(prefix_lengths.tolist()):
            members = (sequence_rows == row).nonzero(as_tuple=True)[0]
            if members.numel():
                key = tuple(input_ids[members[0], :length].tolist())
                groups.setdefault(key, []).extend(members.tolist())
        if not groups:
            raise ValueError("cannot score an empty batch")
        read = None
        for prefix, sequences in groups.items():
            members = torch.tensor(sequences, dtype=torch.long, device=input_ids.device)
            length = len(prefix)
            cache = self.backbone(input_ids=input_ids[members[:1], :length], use_cache=True)
            cache = cache.past_key_values
            cache.reorder_cache(members.new_zeros(members.numel()))
            end = int(read_positions[members].max()) + 1
            hidden = self.backbone(
                input_ids=input_ids[members, length:end],
                attention_mask=attention_mask[members, :end],
                past_key_values=cache,
                use_cache=True,
            ).last_hidden_state
            local = read_positions[members] - length
            reads = hidden[torch.arange(members.numel(), device=hidden.device), local].float()
            if read is None:
                read = reads.new_empty((input_ids.size(0), reads.size(-1)))
            read[members] = reads
        return read
