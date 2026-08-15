"""Wilson score interval for a binomial proportion.

Normal-approximation intervals are wrong at the rates we report (near 0 and 1,
n≈30). Wilson is well-behaved there and is what the tables use.
"""

from __future__ import annotations

import math


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Return (point, low, high) for a proportion at ~95% (z=1.96)."""
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


def fmt(successes: int, n: int) -> str:
    p, lo, hi = wilson(successes, n)
    if n == 0:
        return "n/a"
    return f"{p:5.1%} [{lo:4.0%},{hi:4.0%}] {successes}/{n}"
