"""Sanity check of data/fix/full_r2.jsonl before training: decoy/query diversity, train-split pools only,
train-split hard negatives only, example counts and target spread. Run from the repo root."""
import json
import sys
from collections import Counter

from src.bench.attacks_grammar import _INST_DECOY_POOLS, QUERY_POOLS_V1, split_indices
from src.bench.data import structured_benign

path = sys.argv[1] if len(sys.argv) > 1 else "data/fix/full_r2.jsonl"
rows = [json.loads(l) for l in open(path)]
kind = Counter("hard_negative" if r["variant"] == "hard_negative" else "clean" if r["variant"] == "clean"
               else "attacked" for r in rows)
print("total", len(rows), dict(kind))

decoy_of = {}
for cat, pool in _INST_DECOY_POOLS.items():
    for i, t in enumerate(pool):
        decoy_of.setdefault(t, []).append((cat, i))
att = [r for r in rows if r["variant"] not in ("clean", "hard_negative")]
decoys = Counter(r["segments"][2] for r in att if len(r["segments"]) > 2)
print("attacked with a decoy", sum(decoys.values()), "distinct decoys", len(decoys))
bad = unknown = 0
by_split = Counter()
for t, n in decoys.items():
    hits = decoy_of.get(t)
    if not hits:
        unknown += n
        continue
    ok = any(i in split_indices("decoy", c, "train") for c, i in hits)
    by_split["train" if ok else "heldout"] += n
    bad += 0 if ok else n
print("decoy uses by split", dict(by_split), "not in any pool", unknown)
print("top decoy share", round(decoys.most_common(1)[0][1] / sum(decoys.values()), 3))
print("decoys by category", dict(Counter(c for t in decoys for c, _ in decoy_of.get(t, [("?", 0)])[:1])))

qtrain = qheld = 0
qseen = Counter()
for cat, pool in QUERY_POOLS_V1.items():
    tr = set(split_indices("query", cat, "train"))
    for i, q in enumerate(pool):
        n = sum(q in r["document"] for r in att)
        if n:
            qseen[(cat, i)] = n
            if i in tr:
                qtrain += n
            else:
                qheld += n
print("distinct pool queries used", len(qseen), "uses train", qtrain, "uses heldout", qheld)
print("query uses per (category, idx)", dict(sorted(qseen.items())))

sb = structured_benign(5000)
split_of = {}
for s in sb:
    split_of.setdefault(s.text, set()).add(s.split)
hn = [r["document"] for r in rows if r["variant"] == "hard_negative"]
print("hard negatives", len(hn), "splits", dict(Counter(tuple(sorted(split_of.get(t, {"MISSING"}))) for t in hn)))

tg = [r["target"] for r in rows]
print("targets: min", round(min(tg), 4), "max", round(max(tg), 4), "mean", round(sum(tg) / len(tg), 4),
      "unset(-1)", sum(t < 0 for t in tg), ">=0.5", sum(t >= 0.5 for t in tg))
print("distinct attack families", len(Counter(r["variant"].rsplit("|", 1)[0] for r in att)))
print("DEGENERATE" if bad or unknown or qheld or len(decoys) < 10 or any(t < 0 for t in tg) else "SANITY_OK")
