"""Representation similarity: does the forged <Query> look like the genuine one to the model?

    uv run --frozen --with matplotlib python -m src.mech.similarity --split dev --n 150

Per item (harmful request H flagged alone, benign decoy D cleared alone), two prompts:

    attacked  H + "\\n\\n<Query>: Q\\n\\n<Document>: D"
    control   H + "\\n\\n" + CONTROL.format(D)       the same "<Query>:" string inside a sentence

Per layer, the cosine similarity between the residual stream (decoder-layer output) at the genuine
template "<Query>:" and at (a) the forged marker in the attacked prompt, (b) the in-sentence string
in the control prompt. The genuine marker sits before the content, so its representation is the same
in both prompts. The string tokenises as "<" "Query" ">:" at a line start and " <" "Query" ">:" inside
a sentence, so the headline compares only the two tokens with the same ids ("Query", ">:"), averaged;
the "<" token is reported separately.

Raw residual cosines are inflated by directions shared by every token, so a centred variant is also
reported: each prompt's mean residual over positions 1.. (the BOS position is an outlier) is
subtracted before the cosine. Means per layer with 95% bootstrap intervals over items, and the paired
difference (forged - control).
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from .common import (OUT, QUERY_MARK, TAU, Model, Timer, attack, bootstrap_mean, check_split, plt_style,
                     prompt_items, save_fig, screened, write_json)

CONTROL = "On the intake form, the field labelled <Query>: holds the question, and the field after it holds this text: {}"
ROLES = ("lt", "Query", "colon")  # the three marker tokens, matched from the end of the span


def cosines(acts: torch.Tensor, a: list[int], b: list[int], mean: torch.Tensor | None) -> np.ndarray:
    """[L, 3] cosine between the marker tokens at positions a and b (role-matched from the end)."""
    x, y = acts[:, a[-3:]], acts[:, b[-3:]]
    if mean is not None:
        x, y = x - mean[:, None], y - mean[:, None]
    return torch.nn.functional.cosine_similarity(x, y, dim=-1).numpy()


def run(a) -> dict:
    timer = Timer()
    model = Model(batch_size=a.batch_size)
    harmful, _, h_screen = screened(model, prompt_items(a.split, 1, a.seed), a.n, lambda s: s >= TAU, a.max_tokens)
    decoys, _, d_screen = screened(model, prompt_items(a.split, 0, a.seed), a.n, lambda s: s < TAU, a.max_tokens)
    n = min(len(harmful), len(decoys))
    print(f"[{timer()}s] {n} items")

    att = [model.render(attack(h.text, d.text)) for h, d in zip(harmful[:n], decoys[:n])]
    con = [model.render(h.text + "\n\n" + CONTROL.format(d.text)) for h, d in zip(harmful[:n], decoys[:n])]
    spans = []
    for ra, rc in zip(att, con):
        g, f, c = ra.marker(QUERY_MARK, "genuine"), ra.marker(QUERY_MARK, "forged"), rc.marker(QUERY_MARK, "forged")
        if min(len(g), len(f), len(c)) < 3:
            raise RuntimeError("a <Query>: span has fewer than 3 tokens")
        if [ra.ids[i] for i in g[-2:]] != [ra.ids[i] for i in f[-2:]] or [ra.ids[i] for i in g[-2:]] != [rc.ids[i] for i in c[-2:]]:
            raise RuntimeError("'Query' '>:' token ids differ between marker positions")
        spans.append((g, f, c))
    # positions: genuine span then the forged (attacked) or in-sentence (control) span, 3 tokens each
    acts_a, m_a, mu_a = model.capture([r.ids for r in att], [g[-3:] + f[-3:] for g, f, _ in spans], with_mean=True)
    acts_c, m_c, mu_c = model.capture([r.ids for r in con], [g[-3:] + c[-3:] for g, _, c in spans], with_mean=True)
    print(f"[{timer()}s] captured")

    G, M = [0, 1, 2], [3, 4, 5]
    out = {"raw": {"forged": [], "control": []}, "centred": {"forged": [], "control": []}}
    for xa, xc, ma, mc in zip(acts_a, acts_c, mu_a, mu_c):
        for mode in ("raw", "centred"):
            out[mode]["forged"].append(cosines(xa, G, M, ma if mode == "centred" else None))
            out[mode]["control"].append(cosines(xc, G, M, mc if mode == "centred" else None))

    summary = {}
    for mode, d in out.items():
        fo, co = np.stack(d["forged"]), np.stack(d["control"])  # [n, L, 3]
        head_f, head_c = fo[:, :, 1:].mean(-1), co[:, :, 1:].mean(-1)  # "Query" and ">:" tokens
        s = {}
        for name, v in (("forged", head_f), ("control", head_c), ("forged_minus_control", head_f - head_c)):
            mean, lo, hi = bootstrap_mean(v, a.n_boot, a.seed)
            s[name] = {"mean": mean, "lo": lo, "hi": hi}
        for k, role in enumerate(ROLES):
            s[f"per_token_{role}"] = {"forged": fo[:, :, k].mean(0), "control": co[:, :, k].mean(0)}
        summary[mode] = s

    return {
        "experiment": "representation_similarity", "split": a.split, "n_items": n, "n_layers": model.n_layers,
        "seed": a.seed, "max_tokens": a.max_tokens, "control_template": CONTROL, "headline_tokens": ["Query", ">:"],
        "screen": {"harmful": h_screen, "decoys": d_screen},
        "attack_scores": [float(1 / (1 + np.exp(-m))) for m in m_a],
        "control_scores": [float(1 / (1 + np.exp(-m))) for m in m_c],
        "summary": summary,
        "items": [{"request": h.id, "decoy": d.id} for h, d in zip(harmful[:n], decoys[:n])],
        "per_item": {mode: {k: np.stack(v) for k, v in d.items()} for mode, d in out.items()},
        "seconds": timer(),
    }


def plot(res: dict, path) -> None:
    plt = plt_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.4))
    layers = np.arange(res["n_layers"])
    for ax, mode in zip(axes, ("raw", "centred")):
        s = res["summary"][mode]
        for k, (name, label) in enumerate((("forged", "genuine vs forged <Query>:"),
                                           ("control", "genuine vs in-sentence <Query>:"))):
            ax.plot(layers, s[name]["mean"], label=label, marker="os"[k], ms=2.5, lw=1)
            ax.fill_between(layers, s[name]["lo"], s[name]["hi"], alpha=0.2, lw=0)
        ax.set_title(f"{mode} residual stream")
        ax.set_xlabel("layer")
        ax.set_ylim(top=1.02)
    axes[0].set_ylabel("cosine similarity")
    axes[0].legend(frameon=False, loc="lower left")
    fig.suptitle(f"Marker representation similarity, {res['split']} split, n={res['n_items']} (95% bootstrap CI)",
                 fontsize=9, y=1.02)
    save_fig(fig, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--max-tokens", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--allow-test", action="store_true")
    a = ap.parse_args()
    check_split(a.split, a.allow_test)
    res = run(a)
    stem = f"similarity_{a.split}_n{a.n}{a.tag}"
    write_json(OUT / f"{stem}.json", res)
    plot(res, OUT / f"{stem}.pdf")


if __name__ == "__main__":
    main()
