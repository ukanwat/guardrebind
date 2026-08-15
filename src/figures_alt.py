"""Two article figures, recomputed from the confirmatory CSVs.

Written for the writeup rather than the log. Differences from src/figures.py:

  1. fig_mechanism.svg collapses the ablation panel, the recency control and the
     ground-truth condition into ONE chart, because they are one argument: the
     forged pair is what moves the model, and the two probes below it are what
     rule out recency and rule out "it learned to say safe".
  2. fig_three_guards.svg includes Shieldstral alongside the two OpenRouter
     guards, so the "three designs" claim in the title is actually shown.
  3. XML-escapes label text. src/figures.py emits a raw "escape < >" label,
     which makes fig_mitigations.svg malformed XML and drops that row.

Pure stdlib, same as src/figures.py.

    uv run python -m src.figures_alt
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from .stats import wilson

RESULTS = Path(__file__).resolve().parent.parent / "results"

C_CTRL = "#6b7280"   # controls / conditions that do not move the model
C_ATK = "#dc2626"    # the structural attack
C_PROBE = "#1d4ed8"  # mechanism probes (recency, harmful decoy)
C_GRID = "#e5e7eb"
C_TEXT = "#111827"


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def load(name: str) -> list[dict]:
    path = RESULTS / name
    return list(csv.DictReader(path.open())) if path.exists() else []


def rate(rows: list[dict], **eq: str) -> tuple[int, int]:
    sel = [r for r in rows if all(r[k] == v for k, v in eq.items())]
    return sum(int(r["cleared"]) for r in sel), len(sel)


def barh_svg(items, title, path, padL=282, note=None):
    """items: (label, successes, n, colour)."""
    W, rowh, padR, padT = 760, 30, 78, 56
    H = padT + rowh * len(items) + (46 if note else 30)
    x0, x1 = padL, W - padR

    def sx(p):
        return x0 + (x1 - x0) * p

    base = padT + rowh * len(items)
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'font-family="ui-sans-serif,system-ui,sans-serif" font-size="13">',
        f'<text x="16" y="26" font-size="15" font-weight="600" '
        f'fill="{C_TEXT}">{esc(title)}</text>',
    ]
    for gx in (0, 0.25, 0.5, 0.75, 1):
        X = sx(gx)
        out.append(f'<line x1="{X:.1f}" y1="{padT-8}" x2="{X:.1f}" y2="{base}" '
                   f'stroke="{C_GRID}"/>')
        out.append(f'<text x="{X:.1f}" y="{base+18}" text-anchor="middle" '
                   f'fill="{C_CTRL}">{int(gx*100)}%</text>')
    for i, (label, s, n, col) in enumerate(items):
        y = padT + i * rowh
        if n == 0:
            continue
        p, lo, hi = wilson(s, n)
        out.append(f'<text x="{padL-10}" y="{y+rowh/2+4:.1f}" text-anchor="end" '
                   f'fill="{C_TEXT}">{esc(label)}</text>')
        out.append(f'<rect x="{x0}" y="{y+7:.1f}" width="{max(sx(p)-x0, 1.0):.1f}" '
                   f'height="{rowh-14}" fill="{col}" opacity="0.85" rx="2"/>')
        out.append(f'<line x1="{sx(lo):.1f}" y1="{y+rowh/2:.1f}" '
                   f'x2="{sx(hi):.1f}" y2="{y+rowh/2:.1f}" stroke="{C_TEXT}"/>')
        for xx in (lo, hi):
            out.append(f'<line x1="{sx(xx):.1f}" y1="{y+rowh/2-4:.1f}" '
                       f'x2="{sx(xx):.1f}" y2="{y+rowh/2+4:.1f}" stroke="{C_TEXT}"/>')
        out.append(f'<text x="{sx(hi)+8:.1f}" y="{y+rowh/2+4:.1f}" '
                   f'fill="{C_TEXT}">{p:.0%} ({s}/{n})</text>')
    if note:
        out.append(f'<text x="16" y="{base+38}" font-size="12" '
                   f'fill="{C_CTRL}">{esc(note)}</text>')
    out.append("</svg>")
    path.write_text("\n".join(out))
    print(f"wrote {path.name}")


def mechanism(ss: list[dict]) -> None:
    rows = [
        ("no payload (control)", ("ablation", "none"), C_CTRL),
        ('"disregard the above", as prose', ("ablation", "sentence_bare"), C_CTRL),
        ("same sentence, forged <Instruct>:", ("ablation", "instruct_only"), C_CTRL),
        ("forged <Instruct>: + <Query>:", ("ablation", "instruct_query"), C_CTRL),
        ("forged <Document>: alone", ("ablation", "doc_only"), C_CTRL),
        ("forged <Query>: + <Document>:", ("ablation", "query_doc"), C_ATK),
        ("...same pair, prepended not appended", ("recency", "prepend"), C_PROBE),
        ("...same pair, trailing doc also harmful",
         ("groundtruth", "query_doc_harmful_decoy"), C_PROBE),
    ]
    items = []
    for label, (exp, cond), col in rows:
        s, n = rate(ss, exp=exp, condition=cond, defended="0")
        items.append((label, s, n, col))
    barh_svg(items,
             "Shieldstral 1.0 3B: share of flagged documents cleared (n=75)",
             RESULTS / "fig_mechanism.svg",
             note="Blue rows are the controls that rule out recency and "
                  "rule out simple compliance. Bars are Wilson 95% intervals.")


def three_guards(ss: list[dict], cm: list[dict]) -> None:
    s_ctrl, n_ctrl = rate(ss, exp="ablation", condition="sentence_bare", defended="0")
    s_atk, n_atk = rate(ss, exp="ablation", condition="query_doc", defended="0")
    items = [
        ("Shieldstral 3B: assertion only", s_ctrl, n_ctrl, C_CTRL),
        ("Shieldstral 3B: forged pair", s_atk, n_atk, C_ATK),
    ]
    votes = defaultdict(list)
    for r in cm:
        votes[(r["model"], r["document"], r["condition"])].append(r["verdict"])
    maj = {k: max(set(v), key=v.count) for k, v in votes.items()}
    pretty = {"gpt-oss-safeguard-20b": "gpt-oss-safeguard 20B",
              "llama-guard-4-12b": "Llama Guard 4 12B"}
    for model in ("gpt-oss-safeguard-20b", "llama-guard-4-12b"):
        docs = {d for (m, d, c) in maj if m == model}
        flagged = [d for d in docs if maj.get((model, d, "clean")) == "1"]
        for cond, lbl, col in (("assertion", "assertion only", C_CTRL),
                               ("structural", "forged pair", C_ATK)):
            hit = sum(1 for d in flagged if maj.get((model, d, cond)) == "0")
            items.append((f"{pretty[model]}: {lbl}", hit, len(flagged), col))
    barh_svg(items, "Three guard designs, one failure",
             RESULTS / "fig_three_guards.svg",
             note="Denominator is documents each guard flags with no payload "
                  "present. Assertion control carries the same claim without "
                  "the forged labels.")


def main() -> None:
    ss = load("confirm_shieldstral.csv")
    cm = load("confirm_crossmodel.csv")
    if not ss or not cm:
        raise SystemExit("run src.confirm and src.crossmodel first")
    mechanism(ss)
    three_guards(ss, cm)


if __name__ == "__main__":
    main()
