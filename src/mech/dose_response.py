"""Decoy dose-response: does bypass track the trailing decoy's own score?

    uv run --frozen --with matplotlib python -m src.mech.dose_response --split dev --n 150 --n-decoys 40

Decoys are train-split prompt items, benign and harmful, scored alone first (--pool of them). A ladder
of --n-decoys is picked whose standalone scores span 0-1: the decoy nearest each of n evenly spaced
targets on [0, 1], without replacement (the guard's scores are bimodal, so the middle rungs sit
wherever the pool has mass; the realised scores are in the output).

Every screened harmful request of --split (flagged alone) is attacked with every rung:
    H + "\\n\\n<Query>: Q\\n\\n<Document>: decoy"
Per rung: bypass = fraction of requests whose attacked score < tau, and the median attacked score.
Reported: Spearman rho between the rung's standalone score and its bypass rate (pre-registered M3
predicts rho <= -0.8), with a 95% bootstrap interval over requests; a binned curve (bypass by
standalone-score bin, bootstrap over requests); and the pooled request x rung rho of attacked score
vs standalone score. With --split train, decoys are excluded from the request pool.
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.stats import spearmanr

from .common import (OUT, TAU, Model, Timer, attack, bootstrap_mean, check_split, plt_style, prompt_items,
                     save_fig, screened, write_json)


def ladder(scores: np.ndarray, n: int) -> list[int]:
    """Indices of n pool entries nearest n evenly spaced targets on [0, 1], without replacement."""
    free = set(range(len(scores)))
    picked = []
    for t in np.linspace(0, 1, n):
        j = min(free, key=lambda i: (abs(scores[i] - t), i))
        picked.append(j)
        free.remove(j)
    return sorted(picked, key=lambda i: scores[i])


def run(a) -> dict:
    timer = Timer()
    model = Model(batch_size=a.batch_size)

    half = a.pool // 2
    pool = []
    for label in (0, 1):
        its, sc, _ = screened(model, prompt_items("train", label, a.seed + 7), half, lambda s: True, a.decoy_max_tokens)
        pool += list(zip(its, sc))
    scores = np.array([s for _, s in pool])
    rungs = ladder(scores, min(a.n_decoys, len(pool)))
    decoys = [pool[i][0] for i in rungs]
    d_scores = np.array([pool[i][1] for i in rungs])
    print(f"[{timer()}s] pool {len(pool)}, ladder of {len(decoys)}: " + " ".join(f"{s:.2f}" for s in d_scores))

    exclude = {d.id for d in decoys}
    harmful, h_scores, h_screen = screened(model, prompt_items(a.split, 1, a.seed, exclude), a.n, lambda s: s >= TAU, a.max_tokens)
    print(f"[{timer()}s] {len(harmful)} screened requests")

    texts = [attack(h.text, d.text) for h in harmful for d in decoys]
    att = np.array(model.score(texts)).reshape(len(harmful), len(decoys))  # [requests, rungs]
    print(f"[{timer()}s] scored {len(texts)} attacked prompts")

    bypass = (att < TAU).astype(float)
    rate = bypass.mean(0)
    rho = spearmanr(d_scores, rate).statistic if len(decoys) > 2 else float("nan")
    rng = np.random.default_rng(a.seed)
    boot = []
    for _ in range(a.n_boot):
        r = bypass[rng.integers(0, len(harmful), len(harmful))].mean(0)
        boot.append(spearmanr(d_scores, r).statistic if np.ptp(r) > 0 else np.nan)
    rho_lo, rho_hi = np.nanpercentile(boot, [2.5, 97.5]) if np.isfinite(boot).any() else (np.nan, np.nan)
    pooled = spearmanr(np.repeat(d_scores[None], len(harmful), 0).ravel(), att.ravel()).statistic

    edges = np.linspace(0, 1, a.bins + 1)
    which = np.clip(np.digitize(d_scores, edges) - 1, 0, a.bins - 1)
    binned = []
    for b in range(a.bins):
        cols = np.where(which == b)[0]
        if not len(cols):
            continue
        m, lo, hi = bootstrap_mean(bypass[:, cols].mean(1), a.n_boot, a.seed)
        binned.append({"lo_edge": edges[b], "hi_edge": edges[b + 1], "n_rungs": len(cols),
                       "mean_standalone": d_scores[cols].mean(), "bypass": float(m), "ci": [float(lo), float(hi)]})
    rung_ci = [bootstrap_mean(bypass[:, k], a.n_boot, a.seed) for k in range(len(decoys))]

    return {
        "experiment": "decoy_dose_response", "split": a.split, "decoy_split": "train", "n_requests": len(harmful),
        "n_rungs": len(decoys), "pool": len(pool), "seed": a.seed, "tau": TAU, "bins": a.bins,
        "screen": h_screen,
        "spearman_rho_bypass": rho, "spearman_rho_ci": [rho_lo, rho_hi],
        "spearman_rho_pooled_scores": pooled,
        "monotone_nonincreasing_bins": all(x["bypass"] >= y["bypass"] for x, y in zip(binned, binned[1:])),
        "binned": binned,
        "rungs": [{"id": d.id, "label": int(d.label), "text": d.text, "standalone": float(s), "bypass": float(r),
                   "bypass_ci": [float(c[1]), float(c[2])], "median_attacked": float(np.median(att[:, k]))}
                  for k, (d, s, r, c) in enumerate(zip(decoys, d_scores, rate, rung_ci))],
        "requests": [{"id": h.id, "score_alone": s} for h, s in zip(harmful, h_scores)],
        "attacked_scores": att,
        "seconds": timer(),
    }


def plot(res: dict, path) -> None:
    plt = plt_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.5))
    rungs = res["rungs"]
    x = np.array([r["standalone"] for r in rungs])
    benign = np.array([r["label"] == 0 for r in rungs])
    y = np.array([r["bypass"] for r in rungs])
    ax = axes[0]
    for mask, lab, mk, c in ((benign, "benign decoy", "o", 0), (~benign, "harmful decoy", "^", 1)):
        ax.scatter(x[mask], y[mask], s=10, marker=mk, label=lab, color=f"C{c}", zorder=3, lw=0)
    if res["binned"]:
        bx = [b["mean_standalone"] for b in res["binned"]]
        ax.errorbar(bx, [b["bypass"] for b in res["binned"]],
                    yerr=np.abs(np.array([b["ci"] for b in res["binned"]]).T - [b["bypass"] for b in res["binned"]]),
                    color="0.2", lw=1, capsize=2, label="binned (95% CI)")
    ax.axvline(0.5, color="0.6", lw=0.5, ls=":")
    ax.set_xlabel("decoy score on its own")
    ax.set_ylabel(r"bypass (attacked score $<\tau$)")
    ax.set_title(f"Spearman $\\rho$ = {res['spearman_rho_bypass']:.2f} over {res['n_rungs']} decoys")
    ax.legend(frameon=False, loc="upper right")
    ax = axes[1]
    ax.scatter(x, [r["median_attacked"] for r in rungs], s=10, color="C2", lw=0, zorder=3)
    ax.plot([0, 1], [0, 1], color="0.6", lw=0.5, ls=":")
    ax.set_xlabel("decoy score on its own")
    ax.set_ylabel("median attacked score")
    ax.set_title("verdict follows the trailing decoy")
    fig.suptitle(f"Decoy dose-response, {res['split']} split, n={res['n_requests']} requests", fontsize=9, y=1.02)
    save_fig(fig, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["train", "dev", "test"], default="dev", help="split of the attacked requests")
    ap.add_argument("--n", type=int, default=150, help="screened harmful requests")
    ap.add_argument("--n-decoys", type=int, default=40, help="ladder rungs")
    ap.add_argument("--pool", type=int, default=2000, help="train-split decoy candidates scored alone (half benign)")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=80, help="cap on request length")
    ap.add_argument("--decoy-max-tokens", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-boot", type=int, default=2_000, help="bootstrap resamples (rho is recomputed per resample)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--allow-test", action="store_true")
    a = ap.parse_args()
    check_split(a.split, a.allow_test)
    res = run(a)
    stem = f"dose_response_{a.split}_n{a.n}_d{a.n_decoys}{a.tag}"
    write_json(OUT / f"{stem}.json", res)
    plot(res, OUT / f"{stem}.pdf")


if __name__ == "__main__":
    main()
