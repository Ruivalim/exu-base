"""Write a tiny offline encoder so the smoke run needs no network.

It reuses the test vocabulary and builds a one-layer BERT with a hidden size of
64. Small enough to train on CPU in seconds, real enough to exercise the whole
pipeline including checkpoint saving.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from transformers import BertConfig, BertModel, BertTokenizer

VOCAB = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "vocab.txt"


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: make_tiny_encoder.py <output-directory>")
    output = Path(sys.argv[1])
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir(parents=True)

    # Loading from the folder works across transformers versions, where the
    # BertTokenizer keyword changed from vocab_file to vocab.
    shutil.copy(VOCAB, output / "vocab.txt")
    tokenizer = BertTokenizer.from_pretrained(output)
    expected = len([line for line in VOCAB.read_text(encoding="utf-8").splitlines() if line])
    if tokenizer.vocab_size != expected:
        raise SystemExit(f"tokenizer has {tokenizer.vocab_size} tokens, expected {expected}")
    tokenizer.save_pretrained(output)

    BertModel(
        BertConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=1,
            intermediate_size=128,
            max_position_embeddings=512,
            type_vocab_size=2,
        )
    ).save_pretrained(output)
    print(f"wrote tiny encoder with {tokenizer.vocab_size} tokens to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
