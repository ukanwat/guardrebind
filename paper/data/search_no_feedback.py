"""Search versus no-search rates in the round-2 adaptive runs, for the paper.

Reads only the per-item JSONL records (no model). `queries` is the 1-based index of the first
candidate scoring below tau. The first candidate of each item is a random draw made before any score
is seen, so the share of items with queries == 1 is the clear rate of one candidate chosen without
score feedback. Candidates 1-20 are the initial random population (adaptive.search_attack, pop=20).

Run from the repo root:  .venv/bin/python paper/data/search_no_feedback.py > paper/data/search_no_feedback.md
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ["base_search_null", "t_seed0_search_null", "base_search", "t_seed0_search", "bt_seed0_search"]


def load(run: str) -> list[dict]:
    return [json.loads(line) for line in (ROOT / "results" / "fix" / f"r2_adaptive_{run}.json.jsonl").open()]


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main() -> None:
    print("# Search versus no-search clear rates (round-2 adaptive runs)\n")
    print("| run | n | cleared by candidate 1 (no feedback) | cleared within the 20 random candidates | cleared within budget |")
    print("|---|---|---|---|---|")
    for run in RUNS:
        rs = load(run)
        n = len(rs)
        q = [r["queries"] for r in rs if r["bypass"]]
        k1, k20, kall = sum(x == 1 for x in q), sum(x <= 20 for x in q), len(q)
        lo, hi = wilson(k1, n)
        print(f"| {run} | {n} | {k1}/{n} ({100 * k1 / n:.0f}%, Wilson [{100 * lo:.0f}%, {100 * hi:.0f}%]) "
              f"| {k20}/{n} | {kall}/{n} |")
    print()
    for guard in ("base", "t_seed0"):
        null = {r["id"] for r in load(f"{guard}_search_null") if r["bypass"]}
        search = [r for r in load(f"{guard}_search") if r["bypass"]]
        ids = {r["id"] for r in search}
        pos: dict[str, int] = {}
        for r in search:
            pos[r["best"]["position"]] = pos.get(r["best"]["position"], 0) + 1
        print(f"- {guard}: {len(search)} search wins; plain-text-control wins {len(null)}, "
              f"of which {len(null & ids)} are also search wins; search-win positions {dict(sorted(pos.items()))}")


if __name__ == "__main__":
    main()
