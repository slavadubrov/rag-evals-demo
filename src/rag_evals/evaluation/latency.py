"""Latency / cost telemetry.

A small ``Tracer`` context manager records per-stage durations. Aggregations
emit p50/p95/p99. Cost tracking accepts a token-count callback so users
can plug in litellm's cost calculator when running live.
"""

from __future__ import annotations

import statistics
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class Tracer:
    events: list[tuple[str, float]] = field(default_factory=list)
    tokens: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield self
        finally:
            self.events.append((name, (time.perf_counter() - t0) * 1000))

    def add_tokens(self, model: str, n: int) -> None:
        self.tokens[model] += n

    def by_stage(self) -> dict[str, list[float]]:
        out: dict[str, list[float]] = defaultdict(list)
        for name, ms in self.events:
            out[name].append(ms)
        return out


@dataclass
class StageStats:
    p50: float
    p95: float
    p99: float
    n: int


def quantiles(values: list[float], qs: tuple[float, ...] = (0.5, 0.95, 0.99)) -> StageStats:
    if not values:
        return StageStats(0.0, 0.0, 0.0, 0)
    s = sorted(values)
    n = len(s)

    def _q(p: float) -> float:
        if n == 1:
            return s[0]
        rank = p * (n - 1)
        lo = int(rank)
        hi = min(lo + 1, n - 1)
        frac = rank - lo
        return s[lo] + frac * (s[hi] - s[lo])

    return StageStats(p50=_q(qs[0]), p95=_q(qs[1]), p99=_q(qs[2]), n=n)


def summarise(tracers: list[Tracer]) -> dict[str, StageStats]:
    merged: dict[str, list[float]] = defaultdict(list)
    for t in tracers:
        for name, ms in t.events:
            merged[name].append(ms)
    return {k: quantiles(v) for k, v in merged.items()}


def mean_total(tracers: list[Tracer]) -> float:
    totals = [sum(ms for _, ms in t.events) for t in tracers]
    return statistics.fmean(totals) if totals else 0.0
