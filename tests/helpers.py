"""Shared helpers for the test suite.

The tokenizer here is a real Hugging Face ``BertTokenizer`` over a 100-token
vocabulary, and the encoder is a real ``BertModel`` with one tiny layer. Tests
stay fully offline while still exercising the real code paths.
"""

from __future__ import annotations

from pathlib import Path

from transformers import BertConfig, BertModel, BertTokenizer

FIXTURES = Path(__file__).parent / "fixtures"
SMOKE = FIXTURES / "smoke.jsonl"
VOCAB = FIXTURES / "vocab.txt"


class FakeTokenizer:
    """Deterministic tokenizer for pure sequence and data tests."""

    cls_token_id = 101
    sep_token_id = 102
    mask_token_id = 103
    pad_token_id = 0
    mask_token = "[MASK]"

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return [1000 + index for index, _ in enumerate(text.split(), start=1)]


def _load_bert_tokenizer(vocab_path: Path) -> BertTokenizer:
    """Construct a slow BERT tokenizer across transformers versions.

    transformers 5 renamed the ``vocab_file`` argument to ``vocab`` and swallows
    the old name through ``**kwargs``, silently building a tokenizer that only
    knows the five special tokens. Try the new name first, then the old one, and
    let callers assert the resulting size.
    """
    try:
        return BertTokenizer(vocab=str(vocab_path), do_lower_case=True)
    except TypeError:
        return BertTokenizer(vocab_file=str(vocab_path), do_lower_case=True)


def tiny_tokenizer() -> BertTokenizer:
    tokenizer = _load_bert_tokenizer(VOCAB)
    expected = len([line for line in VOCAB.read_text(encoding="utf-8").splitlines() if line])
    if tokenizer.vocab_size != expected:
        raise RuntimeError(
            f"test tokenizer loaded {tokenizer.vocab_size} tokens, expected {expected}; "
            "the BertTokenizer constructor changed again"
        )
    return tokenizer


def tiny_encoder(vocab_size: int = 100, hidden_size: int = 64) -> BertModel:
    return BertModel(
        BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=1,
            num_attention_heads=1,
            intermediate_size=128,
            max_position_embeddings=512,
            type_vocab_size=2,
        )
    )


def smoke_examples(split: str | None = None):
    from exu import load_jsonl

    return load_jsonl(SMOKE, split)


def tiny_model(hidden_size: int = 64):
    from exu import ExuConfig, ExuModel

    return ExuModel(
        tiny_encoder(hidden_size=hidden_size),
        ExuConfig(encoder_name="tiny-bert", num_decision_layers=1, dropout=0.0),
    )


def tiny_dataset(builder, split: str | None = None):
    from exu import DecisionDataset

    return DecisionDataset(smoke_examples(split), builder)


def save_tiny_checkpoint(path, *, sequence=None, temperature=None):
    from exu import (
        ActionCosts,
        ExuConfig,
        ExuModel,
        SequenceConfig,
        TemperatureMap,
        save_checkpoint,
    )

    model = ExuModel(
        tiny_encoder(),
        ExuConfig(
            encoder_name="tiny-bert",
            num_decision_layers=1,
            dropout=0.0,
            action_costs=ActionCosts(),
        ),
    )
    return save_checkpoint(
        path,
        model,
        sequence_config=sequence or SequenceConfig(),
        temperature=temperature or TemperatureMap(),
        tokenizer=tiny_tokenizer(),
    )


# Words a decoder test may need, on top of the fixture's own. The builder's
# scaffolding ("State:", "Question:", ...) is here so it tokenizes to real ids.
_DECODER_WORDS = [
    "state",
    "question",
    "candidate",
    "decision",
    "type",
    "choice",
    "ordinal",
    "scale",
    "yes",
    "or",
    "no",
    "red",
    "green",
    "blue",
    "gold",
    "alpha",
    "beta",
    "do",
    "the",
    "two",
    "colours",
    "match",
]


def tiny_decoder_tokenizer():
    """A real fast tokenizer, word level, built offline from the fixture's words.

    It has an end-of-sequence and a padding token, as a decoder's does, and it
    saves and loads like any Hugging Face tokenizer, so checkpoints round-trip.
    """
    import json

    from tokenizers import Tokenizer, normalizers, pre_tokenizers
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    normalizer = normalizers.Lowercase()
    splitter = pre_tokenizers.Whitespace()
    words = set(_DECODER_WORDS)
    for line in SMOKE.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        question = record["question"]
        texts = [record["state"], question["instruction"]]
        texts += [
            option["name"] + " " + option.get("description", "") for option in question["options"]
        ]
        for text in texts:
            words.update(
                piece for piece, _ in splitter.pre_tokenize_str(normalizer.normalize_str(text))
            )
    vocab = {
        token: index for index, token in enumerate(["<pad>", "<eos>", "<unk>", *sorted(words)])
    }
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.normalizer = normalizer
    backend.pre_tokenizer = splitter
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, eos_token="<eos>", pad_token="<pad>", unk_token="<unk>"
    )


def tiny_decoder(vocab_size: int):
    from transformers import Qwen3Config, Qwen3Model

    return Qwen3Model(
        Qwen3Config(
            vocab_size=vocab_size,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            max_position_embeddings=512,
        )
    )


def tiny_composite_decoder(vocab_size: int):
    """A Qwen3.5-shaped backbone: the text sizes live in `text_config`, next to a vision tower."""
    from transformers import Qwen3_5Config, Qwen3_5Model

    return Qwen3_5Model(
        Qwen3_5Config(
            text_config={
                "vocab_size": vocab_size,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 16,
                "max_position_embeddings": 512,
                "layer_types": ["linear_attention", "full_attention"],
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 2,
                "linear_key_head_dim": 8,
                "linear_value_head_dim": 8,
            },
            vision_config={
                "depth": 1,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_heads": 2,
                "out_hidden_size": 32,
                "num_position_embeddings": 16,
            },
        )
    )


def tiny_decoder_model(tokenizer=None):
    from exu import DecoderDecisionModel, ExuConfig

    tokenizer = tokenizer or tiny_decoder_tokenizer()
    return DecoderDecisionModel(
        tiny_decoder(len(tokenizer)), ExuConfig(encoder_name="tiny-qwen", backbone="decoder")
    )
