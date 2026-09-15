"""Token/cost/latency tracking with structured JSON-lines logging.

Every request is logged as one JSON object per line to
``settings.tracking_log_path`` (directory created automatically), making it
trivial to tail, grep, or load into pandas for analysis. This module has no
LangChain dependency so it can also be reused by the eval script.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

from src.config import Settings, get_settings


@dataclass
class RequestUsage:
    """A single request's token/cost/latency record."""

    query: str
    model: str
    complexity: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    latency_seconds: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


class UsageTracker:
    """Logs per-request usage records as JSON lines to a local file."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.log_path = Path(self.settings.tracking_log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def estimate_cost(
        self, model: str, prompt_tokens: int, completion_tokens: int
    ) -> float:
        """Estimate USD cost for a request based on configured per-1K prices."""

        is_cheap = model == self.settings.cheap_model
        prompt_price = (
            self.settings.cheap_model_prompt_price_per_1k
            if is_cheap
            else self.settings.strong_model_prompt_price_per_1k
        )
        completion_price = (
            self.settings.cheap_model_completion_price_per_1k
            if is_cheap
            else self.settings.strong_model_completion_price_per_1k
        )
        return (prompt_tokens / 1000) * prompt_price + (completion_tokens / 1000) * completion_price

    def log(self, usage: RequestUsage) -> None:
        """Append ``usage`` as one JSON line to the log file."""

        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(usage.to_dict()) + "\n")

    def record(
        self,
        query: str,
        model: str,
        complexity: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_seconds: float,
    ) -> RequestUsage:
        """Build a :class:`RequestUsage`, log it, and return it."""

        cost = self.estimate_cost(model, prompt_tokens, completion_tokens)
        usage = RequestUsage(
            query=query,
            model=model,
            complexity=complexity,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            estimated_cost_usd=cost,
            latency_seconds=latency_seconds,
        )
        self.log(usage)
        return usage

    @contextmanager
    def timed(self) -> Iterator[dict]:
        """Context manager yielding a dict to stash timing results in.

        Usage::

            with tracker.timed() as timing:
                ...do work...
            print(timing["latency_seconds"])
        """

        timing: dict = {}
        start = time.perf_counter()
        try:
            yield timing
        finally:
            timing["latency_seconds"] = time.perf_counter() - start
