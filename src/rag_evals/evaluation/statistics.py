"""Seeded query-level percentile bootstrap; descriptive, not a release guarantee."""

from __future__ import annotations

import random
from statistics import fmean


def bootstrap_mean(
    values: list[float], *, seed: int = 7, samples: int = 1000
) -> tuple[float, float]:
    if not values:
        raise ValueError("Cannot bootstrap an empty population")
    if samples < 2:
        raise ValueError("Need at least two bootstrap samples")
    rng = random.Random(seed)
    means = sorted(fmean(rng.choices(values, k=len(values))) for _ in range(samples))
    return means[int(0.025 * (samples - 1))], means[int(0.975 * (samples - 1))]
