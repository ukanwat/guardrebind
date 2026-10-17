"""Decoys used by winning score-guided and plain-text-control candidates, for the paper.

For each round-2 adaptive search run it lists, per winning item, whether the best candidate's decoy is a
decoy-pool sentence (held-out half of attacks_grammar decoy pools) or one of the development-split user prompts,
and scores every distinct winning decoy on its own with the base Shieldstral guard (default policy, CPU).

Run from the repo root:  .venv/bin/python paper/data/search_decoys.py > paper/data/search_decoys.md
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.bench.attacks_grammar import split_pool  # noqa: E402
from src.bench.guards import ShieldstralFixed  # noqa: E402

RUNS = ["base_search", "base_search_null", "t_seed0_search", "t_seed0_search_null", "bt_seed0_search"]


def wins(run: str) -> dict[str, dict]:
    path = ROOT / "results" / "fix" / f"r2_adaptive_{run}.json.jsonl"
    return {r["id"]: r["best"] for r in map(json.loads, path.open()) if r["bypass"]}


def main() -> None:
    pool = {d for cat in ("benign", "borderline", "harmful") for d in split_pool("decoy", cat, "heldout")}
    by_run = {run: wins(run) for run in RUNS}
    decoys = sorted({b["decoy"] for w in by_run.values() for b in w.values()})
    guard = ShieldstralFixed(adapter=None, bounded=False, device="cpu")
    scores = dict(zip(decoys, guard.score(decoys, kind="prompt")))

    print("# Decoys of winning adaptive-search candidates\n")
    print("| run | wins | decoy-pool decoy | dev-prompt decoy | distinct decoys | max standalone score |")
    print("|---|---|---|---|---|---|")
    for run, w in by_run.items():
        ds = [b["decoy"] for b in w.values()]
        n_pool = sum(d in pool for d in ds)
        print(f"| {run} | {len(ds)} | {n_pool} | {len(ds) - n_pool} | {len(set(ds))} | {max(scores[d] for d in ds):.3f} |")
    t, t_null = by_run["t_seed0_search"], by_run["t_seed0_search_null"]
    print(f"\nT plain-text-control wins that are also T search wins: {len(set(t_null) & set(t))} of {len(t_null)}")
    s = list(scores.values())
    print(f"\nStandalone base-guard score of the {len(s)} distinct winning decoys: median {statistics.median(s):.4f}, "
          f"max {max(s):.3f}, {sum(v >= 0.5 for v in s)} at or above 0.5, {sum(v < 0.05 for v in s)} below 0.05\n")
    for d, v in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"- {v:.4f}  {d[:100]!r}")


if __name__ == "__main__":
    main()
