"""Checks for the benchmark data layer. Uses the cached data/*.jsonl (built on first use).

    uv run python -m src.bench.test_data
"""

from __future__ import annotations

import csv
import io
import urllib.request
from collections import Counter

from .data import (ADVBENCH_CSV, CATEGORIES, Item, core_set, load_all, normalise, split_of,
                   structured_benign)

FIELDS = {"id": str, "text": str, "kind": str, "label": int, "category": str, "source": str, "lang": str, "split": str}


def check_schema(items: list[Item]) -> None:
    ids = Counter(it.id for it in items)
    assert not [i for i, n in ids.items() if n > 1], "duplicate ids"
    for it in items:
        for name, typ in FIELDS.items():
            assert isinstance(getattr(it, name), typ), (it.id, name)
        assert it.id.startswith(it.source + ":"), it.id
        assert it.text.strip(), it.id
        assert it.kind in ("prompt", "response"), it.id
        assert it.label in (0, 1), it.id
        assert it.split in ("train", "dev", "test"), it.id
        if it.label:
            assert it.category in CATEGORIES, (it.id, it.category)
        else:
            assert it.category == "benign", (it.id, it.category)
        if it.kind == "response":
            assert "\n\n" in it.text, it.id


def check_split_determinism(items: list[Item]) -> None:
    for it in items:
        if it.source == "advbench" or it.source == "structured_benign":
            continue
        key = it.text.partition("\n\n")[0] if it.kind == "response" else it.text
        assert split_of(key) == split_of(key) == it.split, it.id
    share = Counter(split_of(f"probe {i}") for i in range(20000))
    for split, want in (("train", 0.6), ("dev", 0.1), ("test", 0.3)):
        assert abs(share[split] / 20000 - want) < 0.02, share


def check_advbench_forced(items: list[Item]) -> None:
    with urllib.request.urlopen(ADVBENCH_CSV, timeout=60) as response:
        goals = [r["goal"] for r in csv.DictReader(io.StringIO(response.read().decode()))]
    by_id = {it.id: it for it in items}
    for row in range(200, 300):
        it = by_id.get(f"advbench:{row}")
        assert it is not None, f"advbench:{row} missing"
        assert it.split == "test", it.id
        assert it.text == goals[row], it.id


def check_no_overlap(items: list[Item]) -> None:
    by_split = {s: {normalise(it.text) for it in items if it.split == s} for s in ("train", "dev", "test")}
    assert not by_split["train"] & by_split["test"], "train/test text overlap"
    assert not by_split["train"] & by_split["dev"], "train/dev text overlap"

    def prompt(it: Item) -> str:
        return normalise(it.text.partition("\n\n")[0] if it.kind == "response" else it.text)

    prompts = {s: {prompt(it) for it in items if it.split == s and it.source != "structured_benign"}
               for s in ("train", "test")}
    shared = prompts["train"] & prompts["test"]
    assert not shared, f"{len(shared)} prompts in both train and test"


def check_core_set(items: list[Item]) -> None:
    a, b = core_set(items=items), core_set(items=items)
    assert [it.id for it in a] == [it.id for it in b]
    assert sum(it.label for it in a) == 2000 and sum(1 - it.label for it in a) == 2000
    assert all(it.split == "test" for it in a)


def main() -> None:
    items = load_all()
    structured = structured_benign()
    for name, check, arg in (
        ("schema", check_schema, items + structured),
        ("split determinism", check_split_determinism, items),
        ("advbench 200-299 in test", check_advbench_forced, items),
        ("no train/test overlap", check_no_overlap, items + structured),
        ("core_set", check_core_set, items),
    ):
        check(arg)
        print(f"ok  {name}")
    again = structured_benign(refresh=True, items=items)
    assert [it.text for it in again] == [it.text for it in structured], "structured_benign not deterministic"
    print("ok  structured_benign determinism")


if __name__ == "__main__":
    main()
