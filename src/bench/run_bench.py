"""GuardRebind runner: score every (guard, attack, item) cell.

    uv run python -m src.bench.run_bench --guards shieldstral --attacks core --items core --out results/bench/core

One JSONL per guard under --out, one row per cell, appended as it goes. Rerun
the same command to resume: cells already present are skipped. Base scores
(no attack) are the `none` attack, which is always in the panel.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from .attacks_grammar import grammar, split_attacks
from .data import core_set, load_all
from .guards import load_guard


def _done(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    return {(r["attack"], r["item"]) for r in map(json.loads, path.open())}


def _items(which: str, n: int):
    if which == "core":
        return core_set(n=n)
    if which == "test":
        return [it for it in load_all() if it.split == "test"]
    raise ValueError(which)


def _attacks(which: str):
    if which == "core":
        return grammar(core=True)
    attacks = grammar(core=False)
    if which == "full":
        return attacks
    train, heldout = split_attacks(attacks)
    return {"train": train, "heldout": heldout}[which]


def run(guard_name: str, attacks, items, out: Path, batch: int, parse_special: bool) -> None:
    path = out / f"{guard_name}{'' if parse_special else '__split'}.jsonl"
    done = _done(path)
    todo = [(a, it) for a in attacks for it in items if (a.name, it.id) not in done]
    print(f"{guard_name}: {len(done)} done, {len(todo)} to go", flush=True)
    if not todo:
        return
    guard = load_guard(guard_name, parse_special_in_content=parse_special)
    t0 = time.time()
    with path.open("a") as f:
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            texts = [a.apply(it.text, guard.spec) for a, it in chunk]
            scores = guard.score(texts, kind=chunk[0][1].kind) if len({it.kind for _, it in chunk}) == 1 else [
                guard.score([t], kind=it.kind)[0] for t, (_, it) in zip(texts, chunk)]
            for (a, it), s in zip(chunk, scores):
                f.write(json.dumps({
                    "guard": guard_name, "attack": a.name, "item": it.id, "label": it.label,
                    "category": it.category, "source": it.source, "score": round(float(s), 6),
                }) + "\n")
            f.flush()
            if (i // batch) % 50 == 0:
                rate = (i + len(chunk)) / (time.time() - t0)
                print(f"  {i + len(chunk)}/{len(todo)} {rate:.1f}/s eta {(len(todo) - i) / rate / 60:.0f}m", flush=True)
    del guard
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--guards", required=True)
    ap.add_argument("--attacks", default="core", choices=["core", "full", "train", "heldout"])
    ap.add_argument("--items", default="core", choices=["core", "test"])
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--split-special", action="store_true", help="tokenize content with special parsing off")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    items, attacks = _items(a.items, a.n), _attacks(a.attacks)
    print(f"{len(items)} items x {len(attacks)} attacks", flush=True)
    for g in a.guards.split(","):
        run(g, attacks, items, a.out, a.batch, not a.split_special)


if __name__ == "__main__":
    main()
