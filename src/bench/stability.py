"""Numeric stability of the benchmark scores on MPS (PROTOCOL_v2 deviation D1(f)).

    uv run --frozen python -m src.bench.stability --inp results/bench/v2 --n 200

Draws --n random cells per guard (seeded) from the run_bench_v2 results and
re-scores them three times with the run's own code path:
  run1, run2   batched exactly as run_bench_v2 does (token-length sorted, token budget)
  single       one cell per forward pass
and compares run1 vs stored, run2 vs run1 (repeatability), single vs run1
(batched vs unbatched). For each comparison: max and 99th-percentile |diff| and
the number of decisions that flip at tau=0.5. Writes stability.json/.csv to --inp.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path

import numpy as np
import torch

from .attacks_grammar import GUARD_SPECS, families, grammar
from .data import load_all
from .guards import load_guard
from .run_bench_v2 import Cell, score_cells

TAU = 0.5


def sample_rows(inp: Path, n: int, seed: int) -> dict[str, list[dict]]:
    """n random rows per guard, across that guard's policies."""
    by_guard: dict[str, list[dict]] = {}
    for path in sorted(inp.glob("*__*.jsonl")):
        if path.name.endswith("__split.jsonl"):
            continue
        with path.open() as f:
            for line in f:
                r = json.loads(line)
                by_guard.setdefault(r["guard"], []).append(r)
    rng = random.Random(seed)
    return {g: rng.sample(rows, min(n, len(rows))) for g, rows in sorted(by_guard.items())}


def rebuild(rows: list[dict]) -> list[Cell]:
    """The exact Cell each row was scored from."""
    fams = {f.family: f for f in families(grammar(core=True))}
    items = {it.id: it for it in load_all()}
    cells = []
    for r in rows:
        f, it = fams[r["attack_family"]], items[r["item"]]
        insts = f.instantiate(it.id, r["inst_id"] + 1)
        a = insts[r["inst_id"]] if len(insts) > 1 else insts[0]
        cells.append(Cell(r["attack_family"], a, r["inst_id"], it))
    return cells


def compare(guard: str, name: str, a, b) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = np.abs(a - b)
    return {"guard": guard, "comparison": name, "n": len(d), "max_abs_diff": float(d.max()) if len(d) else 0.0,
            "p99_abs_diff": float(np.percentile(d, 99)) if len(d) else 0.0,
            "flips": int(((a >= TAU) != (b >= TAU)).sum())}


def check_guard(guard, guard_name: str, rows: list[dict], max_tokens: int, max_batch: int) -> list[dict]:
    cells = rebuild(rows)
    stored = np.array([r["score"] for r in rows])
    run1, run2, single = (np.zeros(len(rows)) for _ in range(3))
    policies = {p.id: p for p in GUARD_SPECS[guard_name].policies}
    for pol in sorted({r["policy_id"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["policy_id"] == pol]
        guard.set_policy(policies[pol])
        sub = [cells[i] for i in idx]
        run1[idx] = score_cells(guard, sub, max_tokens, max_batch)
        run2[idx] = score_cells(guard, sub, max_tokens, max_batch)
        single[idx] = [score_cells(guard, [c], max_tokens, 1)[0] for c in sub]
    # Stored scores are rounded to 6 decimals in the JSONL.
    return [compare(guard_name, "run1 vs stored", run1.round(6), stored),
            compare(guard_name, "run2 vs run1", run2, run1),
            compare(guard_name, "single vs batched", single, run1)]


def main(argv: list[str] | None = None, loader=load_guard) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", type=Path, default=Path("results/bench/v2"))
    ap.add_argument("--n", type=int, default=200, help="cells per guard")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--max-batch", type=int, default=32)
    a = ap.parse_args(argv)
    out = []
    for guard_name, rows in sample_rows(a.inp, a.n, a.seed).items():
        guard = loader(guard_name)
        try:
            out += check_guard(guard, guard_name, rows, a.max_tokens, a.max_batch)
        finally:
            guard.close()
            del guard
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        for r in out[-3:]:
            print(f"{r['guard']:18s} {r['comparison']:18s} n={r['n']} max={r['max_abs_diff']:.2e} "
                  f"p99={r['p99_abs_diff']:.2e} flips={r['flips']}", flush=True)
    (a.inp / "stability.json").write_text(json.dumps({"n_per_guard": a.n, "seed": a.seed, "comparisons": out},
                                                     indent=1))
    import pandas as pd

    pd.DataFrame(out).to_csv(a.inp / "stability.csv", index=False)


if __name__ == "__main__":
    main()
