"""Paper figures, drawn from the raw result CSVs.

    uv run --frozen --with matplotlib python paper/figures/make_figures.py

Writes vector PDFs next to this file. Fonts are STIX (a Times clone, matching
the newtx text font); colours are Okabe-Ito (colour-blind safe).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
# results/ may be untracked in a worktree; GUARD_RESULTS points at another checkout's copy.
RESULTS = Path(os.environ.get("GUARD_RESULTS", HERE.parent.parent / "results"))

OKABE = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
}
TAU = 0.5

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def _strip(ax, x: float, y: np.ndarray, colour: str, rng: np.random.Generator, width: float = 0.28) -> None:
    jitter = rng.uniform(-width / 2, width / 2, size=len(y))
    ax.scatter(x + jitter, y, s=9, color=colour, alpha=0.75, linewidths=0, zorder=3)
    ax.hlines(np.median(y), x - width / 2 - 0.05, x + width / 2 + 0.05, color=OKABE["black"], lw=1.2, zorder=4)


def _threshold(ax) -> None:
    ax.axhline(TAU, color="0.45", lw=0.7, ls=(0, (4, 3)), zorder=1)
    ax.text(ax.get_xlim()[1], TAU + 0.015, r"$\tau = 0.5$", ha="right", va="bottom", fontsize=7.5, color="0.35")


def fig_scores() -> None:
    """Per-document scores: (a) no attack vs forged pair; (b) the decoy ladder."""
    rng = np.random.default_rng(0)
    # Pre-registered panel: 75 screened held-out rows, undefended.
    s = pd.read_csv(RESULTS / "confirm_shieldstral.csv")
    abl = s[(s.exp == "ablation") & (s.defended == 0)]
    none = abl[abl.condition == "none"].score.to_numpy()
    pair = abl[abl.condition == "query_doc"].score.to_numpy()
    # Decoy ladder: 39 screened rows, three trailing documents.
    g = pd.read_csv(RESULTS / "confirm_groundtruth.csv")
    rungs = [("benign", "benign"), ("mild (old decoy)", "borderline"), ("flagging", "flagged")]

    fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, 2.35), gridspec_kw={"width_ratios": [2, 3], "wspace": 0.32})

    _strip(a, 0, none, OKABE["vermillion"], rng)
    _strip(a, 1, pair, OKABE["blue"], rng)
    a.set_xlim(-0.55, 1.55)
    a.set_xticks([0, 1], ["no attack", "forged pair"])
    a.set_ylim(-0.03, 1.03)
    a.set_ylabel(r"guard score $P(\mathrm{yes})$")
    _threshold(a)
    for x, y in [(0, none), (1, pair)]:
        cleared = int((y < TAU).sum())
        a.text(x, 1.07, f"{cleared}/{len(y)} cleared", ha="center", va="bottom", fontsize=7.5)
    a.set_title("(a) forged pair, $n=75$", loc="left", pad=14)

    colours = [OKABE["blue"], OKABE["orange"], OKABE["vermillion"]]
    for i, ((key, label), colour) in enumerate(zip(rungs, colours)):
        rows = g[g.decoy == key]
        _strip(b, i, rows.attack_score.to_numpy(), colour, rng)
        standalone = rows.decoy_standalone.iloc[0]
        b.scatter([i + 0.27], [standalone], marker="D", s=16, facecolor="white", edgecolor=OKABE["black"], lw=0.8, zorder=5)
        b.text(i, 1.07, f"{int(rows.cleared.sum())}/{len(rows)} cleared", ha="center", va="bottom", fontsize=7.5)
    b.set_xlim(-0.55, 2.55)
    b.set_xticks(range(3), [f"{label} decoy" for _, label in rungs])
    b.set_ylim(-0.03, 1.03)
    _threshold(b)
    b.scatter([], [], marker="D", s=16, facecolor="white", edgecolor=OKABE["black"], lw=0.8, label="decoy scored alone")
    b.legend(loc="center left", bbox_to_anchor=(0.0, 0.62), frameon=False, handletextpad=0.3)
    b.set_title("(b) trailing decoy varied, forged pair fixed, $n=39$", loc="left", pad=14)

    fig.savefig(HERE / "fig_scores.pdf")
    plt.close(fig)


def fig_mech() -> None:
    """(a) Activation patching on the test split; (b) decoy dose-response on the test split."""
    p = json.loads((RESULTS / "mech" / "patching_test_n100.json").read_text())
    d = json.loads((RESULTS / "mech" / "dose_response_test_n150_d40.json").read_text())

    fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, 2.35), gridspec_kw={"wspace": 0.3})
    layers = np.arange(p["n_layers"])
    series = [("final", "into_attacked", "final position, flag restored", OKABE["vermillion"], "-"),
              ("final", "into_swapped", "final position, bypass transferred", OKABE["blue"], "-"),
              ("forged_markers", "into_attacked", "forged labels, flag restored", OKABE["orange"], "--"),
              ("forged_frame", "into_attacked", "forged query + labels, flag restored", OKABE["green"], ":")]
    for key, direction, label, colour, ls in series:
        e = p["summary"][key][direction]
        a.plot(layers, e["flip_rate"], color=colour, ls=ls, lw=1.2, label=label)
        a.fill_between(layers, e["flip_lo"], e["flip_hi"], color=colour, alpha=0.15, lw=0)
    a.set_xlabel("layer patched")
    a.set_ylabel("share of verdicts flipped")
    a.set_ylim(-0.03, 1.03)
    a.legend(loc="upper left", frameon=False, handlelength=1.6, fontsize=7)
    a.set_title(f"(a) activation patching, $n={p['n_items']}$", loc="left")

    rungs = d["rungs"]
    x = np.array([r["standalone"] for r in rungs])
    y = np.array([r["bypass"] for r in rungs])
    b.scatter(x, y, s=12, color=OKABE["blue"], linewidths=0, zorder=3, label="one decoy")
    bins = d["binned"]
    bx = [r["mean_standalone"] for r in bins]
    by = [r["bypass"] for r in bins]
    err = np.array([[r["bypass"] - r["ci"][0] for r in bins], [r["ci"][1] - r["bypass"] for r in bins]])
    b.errorbar(bx, by, yerr=err, color=OKABE["black"], lw=0.9, capsize=2, zorder=4, label="binned (95% CI)")
    b.set_xlabel("decoy's standalone score")
    b.set_ylabel("bypass")
    b.set_xlim(-0.03, 1.03)
    b.set_ylim(-0.03, 1.03)
    b.legend(loc="upper right", frameon=False, fontsize=7)
    b.set_title(f"(b) {d['n_rungs']} decoys, $n={d['n_requests']}$, " + r"$\rho=" + f"{d['spearman_rho_bypass']:.2f}$",
                loc="left")

    fig.savefig(HERE / "fig_mech.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_scores()
    print("wrote", HERE / "fig_scores.pdf")
    fig_mech()
    print("wrote", HERE / "fig_mech.pdf")
