from __future__ import annotations

from helpers import tiny_dataset, tiny_model, tiny_tokenizer
from torch.utils.data import DataLoader

from rlcd import TemperatureMap
from rlcd.evaluation import (
    collect,
    grouped_metrics,
    latency_benchmark,
    metrics_for,
    order_robustness,
)
from rlcd.sequence import SequenceBuilder


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


def test_order_robustness_reports_a_stability_in_range() -> None:
    builder, dataset, _loader, model = _setup()

    report = order_robustness(model, builder, dataset.examples, TemperatureMap(), permutations=3)

    assert report["examples"] == 24
    assert report["permutations"] == 3
    assert 0.0 <= report["stability"] <= 1.0
    assert 0.0 <= report["mean_winner_probability"] <= 1.0


def test_latency_benchmark_reports_percentiles() -> None:
    builder, dataset, _loader, model = _setup()

    report = latency_benchmark(
        model, builder, dataset.examples, TemperatureMap(), questions_per_call=(1, 3), repeats=2
    )

    assert set(report) == {"1", "3"}
    assert report["1"]["p50_ms"] > 0
    assert report["3"]["p95_ms"] >= report["3"]["p50_ms"]
