"""End-to-end RLCD run, fully offline, in a few seconds.

Builds a tiny encoder, writes a small labeled file, trains with the RLCD policy,
saves a checkpoint, reloads it and answers a question. Run it with:

    uv run python examples/end_to_end.py

Nothing here is a quality claim. It exercises the real code paths: sequence
builder, policy gradient, checkpoint round trip, inference runtime.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from transformers import BertConfig, BertModel, BertTokenizer

from rlcd import (
    DecisionQuestion,
    DecisionRuntime,
    PolicyConfig,
    RLCDConfig,
    RLCDModel,
    SequenceBuilder,
    TrainingExample,
    load_jsonl,
    policy_gradient_loss,
    save_checkpoint,
    write_jsonl,
)

LABELS = [{"name": "No"}, {"name": "Yes"}]
TRAIN = [
    ("the payment failed", "No", "Yes"),
    ("please update my invoice address", "No", "Yes"),
    ("i cannot log in to my account", "No", "Yes"),
    ("the card was charged twice", "Yes", "Yes"),
]


def build_tokenizer(directory: Path) -> BertTokenizer:
    words = set()
    for state, _false, _true in TRAIN:
        words.update(state.lower().split())
    words.update({"no", "yes", "type:", "or", "failed", "payment", "describe", "message", "this"})
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "no", "yes", ":"] + sorted(words)
    entries = list(dict.fromkeys(token for token in vocab if token))
    (directory / "vocab.txt").write_text("\n".join(entries) + "\n", encoding="utf-8")
    # Loading from the folder is stable across transformers versions, where the
    # BertTokenizer keyword changed from vocab_file to vocab.
    tokenizer = BertTokenizer.from_pretrained(directory)
    if tokenizer.vocab_size != len(entries):
        raise RuntimeError(f"tokenizer has {tokenizer.vocab_size} tokens, expected {len(entries)}")
    return tokenizer


def build_examples() -> list[TrainingExample]:
    records = []
    for index, (state, _false, true_label) in enumerate(TRAIN):
        target = [1.0, 0.0] if true_label == "No" else [0.0, 1.0]
        records.append(
            TrainingExample.from_record(
                {
                    "id": f"pay-{index}",
                    "state": state,
                    "question": {
                        "kind": "noul",
                        "instruction": "does this message describe a failed payment",
                        "options": LABELS,
                    },
                    "target": target,
                    "family": "payment",
                }
            )
        )
    return records


def main() -> None:
    with tempfile.TemporaryDirectory() as workspace:
        directory = Path(workspace)
        tokenizer = build_tokenizer(directory)
        encoder = BertModel(
            BertConfig(
                vocab_size=tokenizer.vocab_size,
                hidden_size=64,
                num_hidden_layers=1,
                num_attention_heads=1,
                intermediate_size=128,
                max_position_embeddings=512,
                type_vocab_size=2,
            )
        )
        model = RLCDModel(
            encoder, RLCDConfig(encoder_name="tiny", num_decision_layers=1, dropout=0.0)
        )
        builder = SequenceBuilder(tokenizer)
        examples = build_examples()
        dataset = build_training_file(directory, examples, builder)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        policy = PolicyConfig(
            samples_per_question=8, sigma_start=0.5, sigma_end=0.2, cross_entropy_weight=1.0
        )
        generator = torch.Generator().manual_seed(7)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=2, shuffle=True, collate_fn=dataset.collate
        )

        for step in range(40):
            sigma = max(policy.sigma_end, policy.sigma_start * (1 - step / 40))
            for batch in loader:
                optimizer.zero_grad()
                output = model(**batch.model_inputs())
                loss, metrics = policy_gradient_loss(
                    output.logits,
                    batch.targets,
                    batch.option_mask,
                    config=policy,
                    sigma=sigma,
                    ordinal_mask=batch.ordinal_mask,
                    generator=generator,
                )
                loss.backward()
                optimizer.step()
            if step % 10 == 0:
                print(f"step {step:02d} reward {metrics.mean_reward:+.4f} sigma {sigma:.2f}")

        checkpoint = directory / "checkpoint"
        save_checkpoint(checkpoint, model, tokenizer=tokenizer, temperature=None)
        print(f"wrote {checkpoint / 'rlcd.json'}")

        runtime = DecisionRuntime.load(checkpoint, device="cpu")
        question = DecisionQuestion.noul("does this message describe a failed payment")
        decision = runtime.decide("the payment failed and the card was declined", question)
        print(
            f"label={decision.label} confidence={decision.confidence:.3f} act={decision.should_act}"
        )


def build_training_file(directory: Path, examples, builder):
    """Write the records out and read them back, proving the JSONL path too."""
    path = directory / "train.jsonl"
    write_jsonl(path, examples)
    reloaded = load_jsonl(path)
    from rlcd import DecisionDataset

    return DecisionDataset(reloaded, builder)


if __name__ == "__main__":
    main()
