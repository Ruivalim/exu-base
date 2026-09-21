from __future__ import annotations

import random
from dataclasses import replace

import pytest
import torch
from helpers import smoke_examples, tiny_dataset, tiny_model, tiny_tokenizer
from torch.utils.data import DataLoader

from exu import DecisionDataset, TemperatureMap, evaluation
from exu.augmentation import random_order
from exu.evaluation import (
    collect,
    grouped_metrics,
    latency_benchmark,
    metrics_for,
    order_robustness,
)
from exu.sequence import SequenceBuilder


def _setup():
    builder = SequenceBuilder(tiny_tokenizer())
    dataset = tiny_dataset(builder)
    loader = DataLoader(dataset, batch_size=6, shuffle=False, collate_fn=dataset.collate)
    return builder, dataset, loader, tiny_model()


def test_collect_and_metrics_cover_the_whole_split() -> None:
    _builder, _dataset, loader, model = _setup()

    collected = collect(model, loader, "cpu")
    metrics = metrics_for(collected, TemperatureMap())

    assert len(collected) == 24
    assert metrics.count == 24
    assert 0.0 <= metrics.ece <= 1.0
    assert 0.0 <= metrics.accuracy <= 1.0


def test_temperature_changes_the_probabilities() -> None:
    _builder, _dataset, loader, model = _setup()
    collected = collect(model, loader, "cpu")

    warm = collected.probabilities(TemperatureMap(default=4.0))
    cold = collected.probabilities(TemperatureMap(default=0.25))

    assert cold.max(dim=-1).values.mean() > warm.max(dim=-1).values.mean()


def test_grouped_metrics_cover_every_kind_and_family() -> None:
    _builder, _dataset, loader, model = _setup()
    collected = collect(model, loader, "cpu")

    by_kind = grouped_metrics(collected, TemperatureMap(), "kind")
    by_family = grouped_metrics(collected, TemperatureMap(), "family")

    assert set(by_kind) == {"choice", "noul", "score"}
    assert set(by_family) == {"access", "payment", "risk", "routing", "urgency"}
    assert sum(item.count for item in by_kind.values()) == 24


def test_one_record_without_a_family_keeps_the_other_families() -> None:
    """Regression: a single `family`-less record emptied the family report.

    `family` is optional in the data contract, so a dataset with one unlabeled
    record is valid, and it used to return `{}` for every family.
    """
    builder = SequenceBuilder(tiny_tokenizer())
    examples = list(smoke_examples())
    expected = {example.family for example in examples if example.family is not None}
    examples[0] = replace(examples[0], family=None)
    dataset = DecisionDataset(examples, builder)
    loader = DataLoader(dataset, batch_size=6, shuffle=False, collate_fn=dataset.collate)

    collected = collect(tiny_model(), loader, "cpu")
    by_family = grouped_metrics(collected, TemperatureMap(), "family")

    assert set(by_family) == expected
    assert sum(item.count for item in by_family.values()) == len(examples) - 1


class _QuestionBuilder:
    """Hands the question itself to the forward pass, so a stub can read its options."""

    def build(self, state, question):
        return question


def _stub_forward(pick):
    def forward(model, builder, encoded, temperatures, device, batch_size):
        rows = []
        for question in encoded:
            row = torch.full((len(question.options),), 0.3 / max(len(question.options) - 1, 1))
            row[pick(question)] = 0.7
            rows.append(row)
        return rows

    return forward


def test_order_robustness_is_one_for_a_model_that_reads_the_options(monkeypatch) -> None:
    # This stub answers by option name, whatever the order. Any mistake in mapping a
    # permuted answer back to the original positions makes the stability drop.
    def by_name(question):
        names = [option.name for option in question.options]
        return names.index(min(names))

    monkeypatch.setattr(evaluation, "_forward_probabilities", _stub_forward(by_name))
    examples = smoke_examples()

    report = order_robustness(None, _QuestionBuilder(), examples, TemperatureMap(), permutations=5)

    assert report["examples"] == 24
    assert report["permutations"] == 5
    assert report["stability"] == 1.0
    assert report["mean_winner_probability"] == pytest.approx(0.7)


def test_order_robustness_catches_a_model_that_answers_by_position(monkeypatch) -> None:
    # This stub always answers the first slot. It agrees with its unpermuted answer
    # only when the permutation leaves the first option in place.
    monkeypatch.setattr(evaluation, "_forward_probabilities", _stub_forward(lambda question: 0))
    examples = smoke_examples()
    generator = random.Random(17)
    kept = [
        random_order(example.option_count, generator)[0] == 0
        for example in examples
        for _ in range(5)
    ]

    report = order_robustness(None, _QuestionBuilder(), examples, TemperatureMap(), permutations=5)

    assert report["stability"] == pytest.approx(sum(kept) / len(kept))
    assert report["stability"] < 0.7


def test_order_robustness_runs_on_the_real_model() -> None:
    builder, dataset, _loader, model = _setup()

    report = order_robustness(model, builder, dataset.examples, TemperatureMap(), permutations=3)

    assert report["examples"] == 24
    assert report["permutations"] == 3


def test_latency_benchmark_reports_percentiles() -> None:
    builder, dataset, _loader, model = _setup()

    report = latency_benchmark(
        model, builder, dataset.examples, TemperatureMap(), questions_per_call=(1, 3), repeats=2
    )

    assert set(report) == {"1", "3"}
    assert report["1"]["p50_ms"] > 0
    assert report["3"]["p95_ms"] >= report["3"]["p50_ms"]
