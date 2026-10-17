"""CPU-only end-to-end test of run_bench_v2 -> stability -> analyze, with a mock
guard (no model is loaded). Train split only.

    uv run --frozen --with matplotlib python -m src.bench.test_bench_v2
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from . import analyze, run_bench_v2, stability
from .attacks_grammar import DECOY_POOL_V1, GUARD_SPECS, families, grammar

GUARDS = "shieldstral,llama_guard_3_1b,shieldgemma_2b"


class MockGuard:
    """Deterministic stand-in: the score is a hash of (policy, text), pushed
    high for clean text and low when a benign decoy from the pool is present,
    so the analysis sees screened items and non-trivial bypass."""

    def __init__(self, name: str, parse_special_in_content: bool = True, **_):
        self.name, self.spec = name, GUARD_SPECS[name]
        self.policies = self.spec.policies
        self.policy = self.policies[0]
        self.backend = SimpleNamespace(batch_size=16)
        self.calls: list[int] = []

    def set_policy(self, policy):
        self.policy = policy

    def build_ids(self, texts, kind="prompt"):
        return [list(t.encode()) for t in texts]

    def score_ids(self, ids):
        self.calls.append(len(ids))
        return [mock_score(bytes(x).decode(), self.policy.id) for x in ids]

    def close(self):
        pass


def mock_score(text: str, policy_id: str) -> float:
    u = int(hashlib.sha256(f"{policy_id}|{text}".encode()).hexdigest()[:8], 16) / 16**8
    return u ** 3 if any(d in text for d in DECOY_POOL_V1) else u ** 0.25


def test_score_cells_order_and_batching():
    items = run_bench_v2.items_for("train", 6)
    cells = run_bench_v2.plan(items, families(grammar(core=True)), 3, True, False)[:300]
    g = MockGuard("shieldgemma_2b")
    direct = [mock_score(c.attack.apply(c.item.text, g.spec), "default") for c in cells]
    for max_tokens, max_batch in ((10**9, 1), (4096, 32), (10**9, 64)):
        assert run_bench_v2.score_cells(g, cells, max_tokens, max_batch) == direct
    g.calls.clear()
    run_bench_v2.score_cells(g, cells, 4096, 32)
    assert max(g.calls) <= 32 and len(g.calls) > 1


def test_plan_counts():
    items = run_bench_v2.items_for("train", 4)
    fams = families(grammar(core=True))
    d = run_bench_v2.plan(items, fams, 3, True, False)
    p = run_bench_v2.plan(items, fams, 3, False, False)
    n_h = sum(it.label for it in items)
    n_b = len(items) - n_h
    multi = [f for f in fams if f.family != run_bench_v2.CLEAN]
    per_item = sum(len(f.instantiate("x:1", 3)) for f in multi)
    six = [f for f in multi if f.family in run_bench_v2.POLICY_FAMILIES]
    assert len(six) == 6
    assert len(d) == len(items) + n_h * per_item + n_b * len(six)
    assert len(p) == len(items) + n_h * sum(len(f.instantiate("x:1", 3)) for f in six)
    # screen: only listed harmful items are attacked
    keep = {items[0].id}
    s = run_bench_v2.plan(items, fams, 3, True, False, screened=keep)
    assert {c.item.id for c in s if c.family != run_bench_v2.CLEAN and c.item.label == 1} == keep


def _rows(out: Path) -> pd.DataFrame:
    return analyze.load(out)


def test_end_to_end_max_cells():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        args = ["--guards", GUARDS, "--split", "train", "--n", "10", "--max-cells", "50", "--out", str(out)]
        run_bench_v2.main(args, loader=MockGuard)
        df = _rows(out)
        sizes = df.groupby(["guard", "policy_id"]).size()
        assert len(sizes) == 9 and (sizes == 50).all(), sizes
        assert set(df.columns) >= {"guard", "policy_id", "attack_family", "inst_id", "item", "label", "category",
                                   "source", "score"}
        assert (df[df.attack_family == "control|none"].groupby(["guard", "policy_id"]).size() == 20).all()
        # attacked harmful items all pass the screen under their own policy
        clean, att = analyze.split_frames(df)
        assert (att.loc[att.label == 1, "base"] >= 0.5).all()
        # resumable: a rerun scores nothing new
        run_bench_v2.main(args, loader=MockGuard)
        assert len(_rows(out)) == len(df)
        assert json.loads((out / "run_config.json").read_text())["split"] == "train"
        stability.main(["--inp", str(out), "--n", "20"], loader=MockGuard)
        s = json.loads((out / "stability.json").read_text())["comparisons"]
        assert len(s) == 9 and all(r["max_abs_diff"] < 1e-6 and r["flips"] == 0 for r in s), s
        _analyze(out)
        assert (out / "summary.md").exists() and (out / "bypass.csv").exists()


def test_end_to_end_full_small():
    """Every cell (no cap) on 10+10 train items: the analysis tables fill in."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        run_bench_v2.main(["--guards", GUARDS, "--split", "train", "--n", "10", "--out", str(out)], loader=MockGuard)
        _analyze(out)
        byp = pd.read_csv(out / "bypass.csv")
        assert set(byp.policy_id) == {"default", "para1", "para2"}
        assert byp[byp.policy_id == "default"].attack_family.nunique() == 56  # all but the clean control
        assert byp[byp.policy_id == "para1"].attack_family.nunique() == 6
        assert ((byp.ci_lo <= byp.bypass + 1e-12) & (byp.bypass <= byp.ci_hi + 1e-12)).all()
        assert (byp.worst + 1e-12 >= byp.bypass).all()
        pr = pd.read_csv(out / "policy_range.csv")
        assert len(pr) == 18 and (pr["range"] >= 0).all()
        tau = pd.read_csv(out / "bypass_tau.csv")
        assert sorted(tau.tau.unique())[0] == 0.05 and sorted(tau.tau.unique())[-1] == 0.95
        auc = pd.read_csv(out / "auc.csv")
        assert auc.auroc.between(0, 1).all() and (auc.family == "clean").sum() == 9
        cat = pd.read_csv(out / "by_category.csv")
        assert cat.small_n.all()  # 10 items: every cell is small
        for name in ("by_source", "over_refusal"):
            assert len(pd.read_csv(out / f"{name}.csv"))
        figs = sorted(p.name for p in out.glob("*.pdf"))
        try:
            import matplotlib  # noqa: F401
            assert len(figs) == 5, figs
        except ImportError:
            pass
        print((out / "summary.md").read_text()[:1500])


def _analyze(out: Path) -> None:
    analyze.main(["--inp", str(out)])


def test_two_level_bootstrap():
    import numpy as np

    rng = np.random.default_rng(0)
    m = np.array([[1, 1, 1], [0, 0, 0], [1, 0, np.nan]], float)
    point, lo, hi = analyze.two_level_bootstrap(m, rng)
    assert abs(point - (1 + 0 + 0.5) / 3) < 1e-12 and lo <= point <= hi
    const = np.ones((5, 3))
    assert analyze.two_level_bootstrap(const, rng) == (1.0, 1.0, 1.0)
    assert analyze.wilson(0, 10)[0] == 0.0


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
