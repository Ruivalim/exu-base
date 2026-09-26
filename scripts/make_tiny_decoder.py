"""Write a tiny offline decoder so the smoke run needs no network.

A two-layer Qwen3 with a hidden size of 32 and a word-level tokenizer built from
the fixture's words, the same pair the tests use. Small enough to train on CPU
in seconds, real enough to exercise the decoder path through exu-train,
exu-evaluate and the checkpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parent.parent / "tests"


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: make_tiny_decoder.py <output-directory>")
    output = Path(sys.argv[1])
    if output.exists():
        raise SystemExit(f"output already exists: {output}")

    sys.path.insert(0, str(TESTS))
    from helpers import tiny_decoder, tiny_decoder_tokenizer

    tokenizer = tiny_decoder_tokenizer()
    tokenizer.save_pretrained(output)
    tiny_decoder(len(tokenizer)).save_pretrained(output)
    print(f"wrote tiny decoder with {len(tokenizer)} tokens to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
