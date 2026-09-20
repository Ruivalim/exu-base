from __future__ import annotations

import pytest
import torch

from rlcd.calibration import (
    TemperatureMap,
    bucket_key,
    fit_temperature,
    fit_temperature_map,
    option_count_bucket,
)


def test_option_count_buckets_match_the_documented_ranges() -> None:
    assert option_count_bucket(2) == "2"
    assert option_count_bucket(3) == "3-5"
    assert option_count_bucket(5) == "3-5"
    assert option_count_bucket(6) == "6-10"
    assert option_count_bucket(10) == "6-10"
    assert option_count_bucket(11) == "11+"
    assert option_count_bucket(255) == "11+"
    assert bucket_key("choice", 4) == "choice:3-5"


def test_option_count_bucket_rejects_impossible_counts() -> None:
    with pytest.raises(ValueError, match="at least two"):
        option_count_bucket(1)
    with pytest.raises(ValueError, match="at most 255"):
        option_count_bucket(256)


def test_fit_temperature_recovers_a_known_rescaling() -> None:
    torch.manual_seed(0)
    logits = torch.randn(256, 3)
    mask = torch.ones(256, 3, dtype=torch.bool)
    target = torch.softmax(logits / 2.0, dim=-1)

    fit = fit_temperature(logits, target, mask, min_samples=32)

    assert fit.temperature == pytest.approx(2.0, rel=0.1)
    assert fit.nll_after <= fit.nll_before
    assert not fit.at_bound
    assert not fit.fallback


def test_fit_temperature_falls_back_below_min_samples() -> None:
    logits = torch.randn(3, 3)
    mask = torch.ones(3, 3, dtype=torch.bool)
    target = torch.softmax(logits, dim=-1)

    fit = fit_temperature(logits, target, mask, min_samples=32)

    assert fit.fallback is True
    assert fit.temperature == 1.0


def test_fit_temperature_flags_a_value_at_the_bound() -> None:
    torch.manual_seed(1)
    logits = torch.randn(256, 3)
    mask = torch.ones(256, 3, dtype=torch.bool)
    target = torch.softmax(logits * 20.0, dim=-1)

    fit = fit_temperature(logits, target, mask, min_samples=32, bounds=(0.1, 10.0))

    assert fit.temperature == pytest.approx(0.1, abs=1e-4)
    assert fit.at_bound is True


def test_fit_temperature_validates_its_inputs() -> None:
    logits = torch.zeros(4, 2)
    mask = torch.ones(4, 2, dtype=torch.bool)
    target = torch.softmax(logits, dim=-1)

    with pytest.raises(ValueError, match="bounds"):
        fit_temperature(logits, target, mask, min_samples=1, bounds=(1.0, 1.0))
    with pytest.raises(ValueError, match="shape"):
        fit_temperature(logits, target, mask[:2], min_samples=1)


def test_map_lookup_prefers_bucket_then_type_then_default() -> None:
    temperature = TemperatureMap(
        default=1.5, by_type={"choice": 2.0}, by_bucket={"choice:3-5": 3.0}
    )

    assert temperature.temperature("choice", 4) == 3.0
    assert temperature.temperature("choice", 2) == 2.0
    assert temperature.temperature("score", 2) == 1.5
    assert temperature.temperatures(["choice", "score"], [4, 2]) == [3.0, 1.5]


def test_map_rejects_a_non_positive_temperature() -> None:
    with pytest.raises(ValueError, match="positive"):
        TemperatureMap(by_type={"choice": 0.0})


def test_map_round_trips_through_a_dict() -> None:
    temperature = TemperatureMap(default=1.2, by_type={"score": 0.8}, by_bucket={"noul:2": 1.4})

    restored = TemperatureMap.from_dict(temperature.to_dict())

    assert restored.temperature("noul", 2) == 1.4
    assert restored.temperature("score", 5) == 0.8
    assert restored.temperature("choice", 5) == 1.2


def test_map_skips_cells_without_enough_rows() -> None:
    torch.manual_seed(2)
    logits = torch.randn(80, 3)
    mask = torch.ones(80, 3, dtype=torch.bool)
    target = torch.softmax(logits / 1.5, dim=-1)
    kinds = ["choice"] * 40 + ["score"] * 40
    counts = [2] * 20 + [3] * 20 + [3] * 40

    temperature, fits = fit_temperature_map(
        logits,
        target,
        mask,
        kinds,
        counts,
        min_samples=30,
        by_bucket=True,
    )

    assert set(temperature.by_type) == {"choice", "score"}
    assert set(temperature.by_bucket) == {"score:3-5"}
    assert fits["bucket:choice:2"].fallback is True
    assert fits["type:choice"].fallback is False
