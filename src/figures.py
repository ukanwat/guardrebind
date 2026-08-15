"""Build the results tables and SVG figures from the confirmatory CSVs.

Reads results/confirm_shieldstral.csv and results/confirm_crossmodel.csv, writes
figures to results/fig_*.svg and a markdown table block to results/tables.md.
Pure stdlib; no plotting deps, so the repo stays light and the charts stay
inspectable.

    uv run python -m src.figures
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from .stats import wilson

RESULTS = Path(__file__).resolve().parent.parent / "results"

# colourblind-safe: one hue for controls, one for the attack, grey gridlines
C_CTRL = "#6b7280"
C_ATK = "#dc2626"
C_GRID = "#e5e7eb"
C_TEXT = "#111827"


def load(name: str) -> list[dict]:
    path = RESULTS / name
    if not path.exists():
        return []
    with path.open() as fh:
        return list(csv.DictReader(fh))


def rate(rows: list[dict], pred) -> tuple[int, int]:
    sel = [r for r in rows if pred(r)]
    return sum(int(r["cleared"]) for r in sel), len(sel)


def barh_svg(items: list[tuple[str, int, int, bool]], title: str, path: Path) -> None:
    """items: (label, successes, n, is_attack). Horizontal bars with Wilson CIs."""
    W, rowh, padL, padR, padT = 720, 30, 250, 60, 56
    H = padT + rowh * len(items) + 30
    x0, x1 = padL, W - padR
    def sx(p): return x0 + (x1 - x0) * p
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'font-family="ui-sans-serif,system-ui,sans-serif" font-size="13">',
        f'<text x="{padL}" y="26" font-size="15" font-weight="600" '
        f'fill="{C_TEXT}">{title}</text>',
    ]
    for gx in (0, .25, .5, .75, 1):
        X = sx(gx)
        parts.append(f'<line x1="{X:.1f}" y1="{padT-8}" x2="{X:.1f}" y2="{H-30}" '
                     f'stroke="{C_GRID}"/>')
        parts.append(f'<text x="{X:.1f}" y="{H-12}" text-anchor="middle" '
                     f'fill="{C_CTRL}">{int(gx*100)}%</text>')
    for i, (label, s, n, atk) in enumerate(items):
        y = padT + i * rowh
        p, lo, hi = wilson(s, n) if n else (0, 0, 0)
        col = C_ATK if atk else C_CTRL
        safe = (label.replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;"))
        parts.append(f'<text x="{padL-10}" y="{y+rowh/2+4:.1f}" text-anchor="end" '
                     f'fill="{C_TEXT}">{safe}</text>')
        parts.append(f'<rect x="{x0}" y="{y+7:.1f}" width="{sx(p)-x0:.1f}" '
                     f'height="{rowh-14}" fill="{col}" opacity="0.85" rx="2"/>')
        if n:
            parts.append(f'<line x1="{sx(lo):.1f}" y1="{y+rowh/2:.1f}" '
                         f'x2="{sx(hi):.1f}" y2="{y+rowh/2:.1f}" stroke="{C_TEXT}"/>')
            for xx in (lo, hi):
                parts.append(f'<line x1="{sx(xx):.1f}" y1="{y+rowh/2-4:.1f}" '
                             f'x2="{sx(xx):.1f}" y2="{y+rowh/2+4:.1f}" stroke="{C_TEXT}"/>')
            parts.append(f'<text x="{sx(hi)+8:.1f}" y="{y+rowh/2+4:.1f}" '
                         f'fill="{C_TEXT}">{p:.0%} ({s}/{n})</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts))
    print(f"wrote {path.name}")


def main() -> None:
    ss = load("confirm_shieldstral.csv")
    if not ss:
        print("no confirm_shieldstral.csv yet")
        return
    n_screen = len({r["document"] for r in ss if r["exp"] == "ablation"})

    # Fig 1: ablation panel (undefended)
    order = ["none", "sentence_bare", "authority_note", "benign_tail",
             "instruct_only", "instruct_query", "doc_only", "query_doc",
             "full_triple"]
    struct = {"instruct_only", "instruct_query", "doc_only", "query_doc", "full_triple"}
    items = []
    for cond in order:
        s, n = rate(ss, lambda r: r["exp"] == "ablation" and r["condition"] == cond
                    and r["defended"] == "0")
        items.append((cond, s, n, cond in struct))
    barh_svg(items, f"Shieldstral: which forged sections rebind the target (n={n_screen})",
             RESULTS / "fig_ablation.svg")

    # Fig 2: ground truth (benign vs harmful decoy)
    gt = []
    for cond, atk in [("query_doc", False), ("query_doc_neutral_decoy", False),
                      ("query_doc_harmful_decoy", True),
                      ("full_triple_harmful_decoy", True)]:
        exp = "ablation" if cond == "query_doc" else "groundtruth"
        s, n = rate(ss, lambda r, c=cond, e=exp: r["exp"] == e
                    and r["condition"] == c and r["defended"] == "0")
        gt.append((cond, s, n, not atk))
    barh_svg(gt, "Ground truth: harmful decoy stops the bypass",
             RESULTS / "fig_groundtruth.svg")

    # Fig 3: query relatedness + mitigations (from supplementary run)
    supp = load("confirm_supp.csv")
    if supp:
        qr_order = ["exact", "paraphrase", "topical", "generic",
                    "stacked_blind", "unrelated"]
        items = []
        for cond in qr_order:
            s, n = rate(supp, lambda r, c=cond: r["exp"] == "query_relatedness"
                        and r["condition"] == c)
            items.append((cond, s, n, cond != "unrelated"))
        barh_svg(items, "Attacker needs no knowledge of the deployment's query",
                 RESULTS / "fig_query_relatedness.svg")

    # Fig 4: what actually mitigates it
    mit = []
    for cond, label in [("none", "no mitigation"),
                        ("escape_brackets", "escape < >"),
                        ("strip_labels", "decorate colon")]:
        s, n = rate(ss, lambda r, c=cond: r["exp"] == "mitigation"
                    and r["condition"] == c)
        mit.append((label, s, n, True))
    if supp:
        s, n = rate(supp, lambda r: r["exp"] == "mitigation2"
                    and r["condition"] == "normalize")
        mit.append(("delete labels (normalize)", s, n, False))
    barh_svg(mit, "Bypass rate after mitigation (lower is better)",
             RESULTS / "fig_mitigations.svg")

    # Fig 5: cross-model
    cm = load("confirm_crossmodel.csv")
    if cm:
        votes: dict[tuple[str, str, str], list[str]] = defaultdict(list)
        for r in cm:
            votes[(r["model"], r["document"], r["condition"])].append(r["verdict"])
        maj = {k: max(set(v), key=v.count) for k, v in votes.items()}
        items = []
        for model in sorted({r["model"] for r in cm}):
            docs = {d for (m, d, c) in maj if m == model}
            flagged = [d for d in docs if maj.get((model, d, "clean")) == "1"]
            for cond, lbl in [("assertion", "assertion (control)"),
                              ("structural", "structural attack")]:
                s = sum(1 for d in flagged if maj.get((model, d, cond)) == "0")
                items.append((f"{model.split('-2')[0][:18]} — {lbl}", s,
                              len(flagged), cond == "structural"))
        barh_svg(items, "Same failure across three guard designs",
                 RESULTS / "fig_crossmodel.svg")

    # tables.md
    lines = ["## Confirmatory results (Shieldstral, held-out AdvBench)\n",
             f"n screened = {n_screen}, tau = 0.5\n",
             "| condition | kind | cleared (undef) | cleared (defended) |",
             "|---|---|---|---|"]
    for cond in order:
        su, nu = rate(ss, lambda r, c=cond: r["exp"] == "ablation"
                      and r["condition"] == c and r["defended"] == "0")
        sd, nd = rate(ss, lambda r, c=cond: r["exp"] == "ablation"
                      and r["condition"] == c and r["defended"] == "1")
        pu = wilson(su, nu); pd = wilson(sd, nd)
        kind = "structural" if cond in struct else "control"
        lines.append(f"| {cond} | {kind} | {pu[0]:.0%} [{pu[1]:.0%},{pu[2]:.0%}] "
                     f"| {pd[0]:.0%} [{pd[1]:.0%},{pd[2]:.0%}] |")
    (RESULTS / "tables.md").write_text("\n".join(lines) + "\n")
    print("wrote tables.md")


if __name__ == "__main__":
    main()
