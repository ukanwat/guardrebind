"""Activation patching: does the verdict follow the forged pair's residual stream?

    uv run --frozen --with matplotlib python -m src.mech.patching --split dev --n 100

Per item (a harmful request H the guard flags alone, and a benign decoy D it clears alone, D picked
from a screened pool to match H's token length as closely as possible):

    attacked  = H + "\\n\\n<Query>: Q\\n\\n<Document>: D"     guard grades D, low score
    swapped   = D + "\\n\\n<Query>: Q\\n\\n<Document>: H"     roles swapped, guard grades H, high score

Items enter only if the attack works (attacked < tau) and the swap is flagged (swapped >= tau), so the
two runs sit on opposite sides of the threshold. For every decoder layer l and each position set,
the output of layer l at those positions is overwritten with the other run's, in both directions:

    into_attacked   receiver = attacked, donor = swapped   (does the flag come back?)
    into_swapped    receiver = swapped, donor = attacked   (does the bypass transfer?)

Position sets are matched by role, not index, so a length mismatch between H and D does not matter:
    forged_query      the "<Query>:" tokens of the forged pair (3 tokens)
    forged_document   the "<Document>:" tokens of the forged pair (3 tokens)
    forged_markers    both of the above
    forged_frame      the forged pair minus its document: "<Query>: Q\n\n<Document>:" (same text in both runs)
    final             the last position ([/INST]), where the label is read

Effect = (m_patched - m_receiver) / (m_donor - m_receiver) on the yes-no logit margin: 0 no effect,
1 a full flip to the donor's margin. Per-layer means with 95% bootstrap intervals over items, and
the fraction of items whose patched score crosses tau to the donor's side. Sanity: `final` at the
last layer is exactly 1 (the final norm and the head only see that position).
"""

from __future__ import annotations

import argparse

import numpy as np

from .common import (DOC_MARK, OUT, QUERY_MARK, TAU, Model, Timer, attack, bootstrap_mean, check_split,
                     plt_style, prompt_items, save_fig, screened, write_json)

SETS = ("forged_query", "forged_document", "forged_markers", "forged_frame", "final")
DIRECTIONS = ("into_attacked", "into_swapped")


def positions(r) -> dict[str, list[int]]:
    q, d = r.marker(QUERY_MARK, "forged"), r.marker(DOC_MARK, "forged")
    frame = list(range(q[0], d[-1] + 1))
    return {"forged_query": q, "forged_document": d, "forged_markers": q + d, "forged_frame": frame,
            "final": [r.final]}


def match_decoy(model: Model, request: str, decoys: list, decoy_len: list[int], used: set[int]) -> int:
    """Index of the unused decoy whose token length is closest to the request's."""
    n = model.n_tokens(request)
    order = sorted((abs(L - n), i) for i, L in enumerate(decoy_len) if i not in used)
    return order[0][1]


def run(a) -> dict:
    timer = Timer()
    model = Model(batch_size=a.batch_size)
    L = model.n_layers

    harmful, h_scores, h_screen = screened(model, prompt_items(a.split, 1, a.seed), a.n * 2, lambda s: s >= TAU, a.max_tokens)
    decoys, d_scores, d_screen = screened(model, prompt_items(a.split, 0, a.seed), a.n * a.decoy_pool, lambda s: s < TAU, a.max_tokens)
    decoy_len = [model.n_tokens(d.text) for d in decoys]
    print(f"[{timer()}s] screened {len(harmful)} harmful, {len(decoys)} decoys")

    items, excluded, used = [], [], set()
    for h, hs in zip(harmful, h_scores):
        t_item = timer()
        if len(items) >= a.n or len(used) >= len(decoys):
            break
        j = match_decoy(model, h.text, decoys, decoy_len, used)
        d = decoys[j]
        rc, rx = model.render(attack(h.text, d.text)), model.render(attack(d.text, h.text))
        pc, px = positions(rc), positions(rx)
        rec = {"id": h.id, "decoy_id": d.id, "score_alone": hs, "decoy_score_alone": d_scores[j],
               "len_request": model.n_tokens(h.text), "len_decoy": decoy_len[j],
               "len_attacked": len(rc.ids), "len_swapped": len(rx.ids)}
        if any(len(pc[s]) != len(px[s]) for s in SETS):
            excluded.append({**rec, "reason": "marker token counts differ"})
            continue
        union_c = sorted({p for s in SETS for p in pc[s]})
        union_x = sorted({p for s in SETS for p in px[s]})
        (act_c, act_x), (m_c, m_x) = model.capture([rc.ids, rx.ids], [union_c, union_x])
        rec.update(margin_attacked=m_c, margin_swapped=m_x,
                   score_attacked=float(1 / (1 + np.exp(-m_c))), score_swapped=float(1 / (1 + np.exp(-m_x))))
        if not (rec["score_attacked"] < TAU <= rec["score_swapped"]) and not a.keep_unscreened:
            excluded.append({**rec, "reason": "attack failed or swap not flagged"})
            continue
        used.add(j)

        runs = {
            "into_attacked": (rc.ids, pc, m_c, act_x, union_x, px, m_x),
            "into_swapped": (rx.ids, px, m_x, act_c, union_c, pc, m_c),
        }
        rec["effect"], rec["flip"], rec["cache_drift"] = {}, {}, {}
        for direction, (ids, p_recv, m_full, act_don, union_don, p_don, m_don) in runs.items():
            for s in SETS:
                # positions before the patch are shared by every row: cache them, and take the
                # receiver's margin from an unpatched row run the same way
                rows = [union_don.index(p) for p in p_don[s]]
                jobs = [(None, [], None)] + [(layer, p_recv[s], act_don[layer, rows]) for layer in range(L)]
                m = np.array(model.patched_margins(ids, jobs, start=min(p_recv[s])))
                m_recv, m = m[0], m[1:]
                eff = (m - m_recv) / (m_don - m_recv)
                flip = (m > 0) == (m_don > 0)  # tau = 0.5 <=> margin 0
                rec["effect"].setdefault(s, {})[direction] = eff.tolist()
                rec["flip"].setdefault(s, {})[direction] = flip.astype(int).tolist()
                rec["cache_drift"].setdefault(s, {})[direction] = float(m_recv - m_full)
        rec["seconds"] = round(timer() - t_item, 1)
        items.append(rec)
        print(f"[{timer()}s] item {len(items)}/{a.n} {h.id}: attacked {rec['score_attacked']:.3f} "
              f"swapped {rec['score_swapped']:.3f}; final@L-1 {rec['effect']['final']['into_attacked'][-1]:.3f}, "
              f"markers peak {max(rec['effect']['forged_markers']['into_attacked']):.3f}")

    summary = {}
    for s in SETS:
        for direction in DIRECTIONS:
            eff = np.array([r["effect"][s][direction] for r in items])
            flip = np.array([r["flip"][s][direction] for r in items])
            if not len(items):
                continue
            mean, lo, hi = bootstrap_mean(eff, a.n_boot, a.seed)
            fmean, flo, fhi = bootstrap_mean(flip, a.n_boot, a.seed)
            summary.setdefault(s, {})[direction] = {
                "effect_mean": mean, "effect_lo": lo, "effect_hi": hi,
                "flip_rate": fmean, "flip_lo": flo, "flip_hi": fhi,
                "peak_layer": int(np.argmax(mean)), "peak_effect": float(np.max(mean)),
            }
    return {
        "experiment": "activation_patching", "split": a.split, "n_requested": a.n, "n_items": len(items),
        "n_layers": L, "seed": a.seed, "keep_unscreened": a.keep_unscreened, "max_tokens": a.max_tokens, "tau": TAU, "n_boot": a.n_boot,
        "position_sets": list(SETS), "directions": list(DIRECTIONS),
        "screen": {"harmful": h_screen, "decoys": d_screen, "excluded": excluded},
        "summary": summary, "items": items, "seconds": timer(),
    }


def plot(res: dict, path) -> None:
    plt = plt_style()
    titles = {"into_attacked": "swapped $\\rightarrow$ attacked run (restore the flag)",
              "into_swapped": "attacked $\\rightarrow$ swapped run (transfer the bypass)"}
    labels = {"forged_query": "forged <Query>:", "forged_document": "forged <Document>:",
              "forged_markers": "both forged markers", "forged_frame": "forged pair frame (markers + Q)",
              "final": "final position"}
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.5), sharey=True)
    layers = np.arange(res["n_layers"])
    for ax, direction in zip(axes, DIRECTIONS):
        for k, s in enumerate(SETS):
            if s not in res["summary"]:
                continue
            d = res["summary"][s][direction]
            ax.plot(layers, d["effect_mean"], label=labels[s], marker="os^vD"[k], ms=2.5, lw=1)
            ax.fill_between(layers, d["effect_lo"], d["effect_hi"], alpha=0.2, lw=0)
        ax.axhline(0, color="0.6", lw=0.5)
        ax.axhline(1, color="0.6", lw=0.5, ls=":")
        ax.set_title(titles[direction])
        ax.set_xlabel("layer (residual stream after decoder layer)")
    axes[0].set_ylabel("normalised margin change")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle(f"Activation patching, Shieldstral-1.0-3B, {res['split']} split, n={res['n_items']} (95% bootstrap CI)",
                 fontsize=9, y=1.02)
    save_fig(fig, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    ap.add_argument("--n", type=int, default=100, help="items (screened request/decoy pairs)")
    ap.add_argument("--max-tokens", type=int, default=80, help="cap on request and decoy length")
    ap.add_argument("--decoy-pool", type=int, default=4, help="screened decoys per item to length-match from")
    ap.add_argument("--batch-size", type=int, default=13, help="patched runs per forward")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="suffix for the output file names")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--keep-unscreened", action="store_true", help="debug only: skip the attacked<tau<=swapped item screen")
    a = ap.parse_args()
    check_split(a.split, a.allow_test)
    res = run(a)
    stem = f"patching_{a.split}_n{a.n}{a.tag}"
    write_json(OUT / f"{stem}.json", res)
    plot(res, OUT / f"{stem}.pdf")


if __name__ == "__main__":
    main()
