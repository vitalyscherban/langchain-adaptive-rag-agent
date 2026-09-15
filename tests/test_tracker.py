"""Tests for the JSON-lines usage tracker."""

from __future__ import annotations

import json

from src.config import Settings
from src.tracking.usage_tracker import UsageTracker


def _settings(tmp_path) -> Settings:
    return Settings(
        tracking_log_path=str(tmp_path / "usage.jsonl"),
        cheap_model="cheap-x",
        strong_model="strong-y",
        cheap_model_prompt_price_per_1k=0.001,
        cheap_model_completion_price_per_1k=0.002,
        strong_model_prompt_price_per_1k=0.01,
        strong_model_completion_price_per_1k=0.02,
    )


def test_record_writes_one_json_line(tmp_path) -> None:
    tracker = UsageTracker(_settings(tmp_path))

    usage = tracker.record(
        query="hello",
        model="cheap-x",
        complexity="simple",
        prompt_tokens=100,
        completion_tokens=50,
        latency_seconds=0.5,
    )

    assert usage.total_tokens == 150
    lines = tracker.log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["query"] == "hello"
    assert record["total_tokens"] == 150


def test_estimate_cost_uses_cheap_pricing_for_cheap_model(tmp_path) -> None:
    settings = _settings(tmp_path)
    tracker = UsageTracker(settings)

    cost = tracker.estimate_cost("cheap-x", prompt_tokens=1000, completion_tokens=1000)

    expected = 1 * settings.cheap_model_prompt_price_per_1k + 1 * settings.cheap_model_completion_price_per_1k
    assert cost == expected


def test_estimate_cost_uses_strong_pricing_for_strong_model(tmp_path) -> None:
    settings = _settings(tmp_path)
    tracker = UsageTracker(settings)

    cost = tracker.estimate_cost("strong-y", prompt_tokens=1000, completion_tokens=1000)

    expected = 1 * settings.strong_model_prompt_price_per_1k + 1 * settings.strong_model_completion_price_per_1k
    assert cost == expected


def test_multiple_records_append_multiple_lines(tmp_path) -> None:
    tracker = UsageTracker(_settings(tmp_path))

    for i in range(3):
        tracker.record(
            query=f"q{i}",
            model="cheap-x",
            complexity="simple",
            prompt_tokens=10,
            completion_tokens=5,
            latency_seconds=0.1,
        )

    lines = tracker.log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3


def test_timed_context_manager_records_latency() -> None:
    import time

    tracker = UsageTracker.__new__(UsageTracker)  # avoid needing a log path
    with tracker.timed() as timing:
        time.sleep(0.01)

    assert timing["latency_seconds"] >= 0.01
