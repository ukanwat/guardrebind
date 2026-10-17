"""Linear probe: which document is being graded, read from the final-position residual stream.

    uv run --frozen --with matplotlib python -m src.mech.probe --split dev --n 150 --n-train 600

Examples: a harmful request H the guard flags alone and a benign decoy D it clears alone, in the
forged-pair layout both ways round:

    benign-last   H + "\\n\\n<Query>: Q\\n\\n<Document>: D"   label 0 (the attack: D is graded)
    harmful-last  D + "\\n\\n<Query>: Q\\n\\n<Document>: H"   label 1 (H is graded)

Each request is paired with --decoys-per-item decoys. Features: the output of every decoder layer at
the final position ([/INST], where the label is read). Per layer, an L2 logistic regression
(standardised features) is fit on train-split examples and scored on --split. Targets:

    graded_doc   harmful-last vs benign-last (the label known by construction)
    verdict      the guard's own decision (score >= tau) on that prompt
    random       a random label per prompt (control: probe capacity / memorisation)

If --split is train, the evaluation requests and decoys are disjoint from the training ones.
Accuracy per layer with a 95% bootstrap interval over requests (each request's examples resampled
together).
"""

from __future__ import annotations

import argparse
import random

import numpy as np

from .common import (OUT, TAU, Model, Timer, attack, bootstrap_mean, check_split, plt_style, prompt_items,
                     save_fig, screened, write_json)

TARGETS = ("graded_doc", "verdict", "random")


def build(model: Model, harmful, decoys, k: int, seed: int, timer) -> dict:
    """Prompts, labels and final-position features for every (request, decoy) pair, both orders."""
    rng = random.Random(seed)
    texts, graded, groups, pairs = [], [], [], []
    order = list(range(len(decoys)))
    for g, h in enumerate(harmful):
        for j in rng.sample(order, min(k, len(order))):
            d = decoys[j]
            texts += [attack(h.text, d.text), attack(d.text, h.text)]
            graded += [0, 1]
            groups += [g, g]
            pairs += [(h.id, d.id)] * 2
    seqs = [model.render(t).ids for t in texts]
    acts, margins = model.capture(seqs, [[len(s) - 1] for s in seqs])
    print(f"[{timer()}s] captured {len(seqs)} prompts")
    feats = np.stack([a[:, 0].numpy() for a in acts])  # [n, L, d]
    margins = np.array(margins)
    return {"X": feats, "graded_doc": np.array(graded), "verdict": (margins > 0).astype(int),
            "random": np.random.default_rng(seed).integers(0, 2, len(texts)), "groups": np.array(groups),
            "margins": margins, "pairs": pairs, "lengths": [len(s) for s in seqs]}


def fit_eval(train: dict, test: dict, target: str, C: float, n_boot: int, seed: int) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y_tr, y_te = train[target], test[target]
    L = train["X"].shape[1]
    correct = np.zeros((len(y_te), L))
    train_acc = []
    if len(set(y_tr)) < 2:  # e.g. the verdict on a tiny run where every attack works
        return {"skipped": f"train labels for {target} are constant"}
    for layer in range(L):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
        clf.fit(train["X"][:, layer], y_tr)
        train_acc.append(clf.score(train["X"][:, layer], y_tr))
        correct[:, layer] = clf.predict(test["X"][:, layer]) == y_te
    groups = test["groups"]
    per_group = np.stack([correct[groups == g].mean(0) for g in np.unique(groups)])
    mean, lo, hi = bootstrap_mean(per_group, n_boot, seed)
    return {"acc": correct.mean(0), "acc_lo": lo, "acc_hi": hi, "train_acc": train_acc,
            "majority_baseline": max(np.mean(y_te), 1 - np.mean(y_te)),
            "first_layer_ge_95": next((l for l, v in enumerate(correct.mean(0)) if v >= 0.95), None)}


def run(a) -> dict:
    timer = Timer()
    model = Model(batch_size=a.batch_size)
    same = a.split == "train"
    n_h_train = a.n_train + (a.n if same else 0)
    n_d_train = a.n_decoys_train + (a.n_decoys if same else 0)

    h_tr, _, h_tr_screen = screened(model, prompt_items("train", 1, a.seed), n_h_train, lambda s: s >= TAU, a.max_tokens)
    d_tr, _, d_tr_screen = screened(model, prompt_items("train", 0, a.seed), n_d_train, lambda s: s < TAU, a.max_tokens)
    if same:
        h_te, h_tr = h_tr[a.n_train :], h_tr[: a.n_train]
        d_te, d_tr = d_tr[a.n_decoys_train :], d_tr[: a.n_decoys_train]
        h_te_screen, d_te_screen = h_tr_screen, d_tr_screen
    else:
        h_te, _, h_te_screen = screened(model, prompt_items(a.split, 1, a.seed), a.n, lambda s: s >= TAU, a.max_tokens)
        d_te, _, d_te_screen = screened(model, prompt_items(a.split, 0, a.seed), a.n_decoys, lambda s: s < TAU, a.max_tokens)
    print(f"[{timer()}s] train {len(h_tr)} requests x {len(d_tr)} decoys; eval {len(h_te)} x {len(d_te)}")

    train = build(model, h_tr, d_tr, a.decoys_per_item, a.seed, timer)
    test = build(model, h_te, d_te, a.decoys_per_item, a.seed + 1, timer)
    probes = {t: fit_eval(train, test, t, a.C, a.n_boot, a.seed) for t in TARGETS}
    for t, r in probes.items():
        if "acc" in r:
            print(f"{t:>10}: eval acc by layer " + " ".join(f"{v:.2f}" for v in r["acc"]))

    def behaviour(d):
        g, v = d["graded_doc"], d["verdict"]
        return {"n_prompts": len(g), "flagged_harmful_last": float(v[g == 1].mean()),
                "flagged_benign_last": float(v[g == 0].mean()), "verdict_agrees_with_graded_doc": float((g == v).mean())}

    return {
        "experiment": "linear_probe", "split": a.split, "train_split": "train", "n_layers": model.n_layers,
        "n_train_requests": len(h_tr), "n_eval_requests": len(h_te), "n_train_decoys": len(d_tr), "n_eval_decoys": len(d_te),
        "decoys_per_item": a.decoys_per_item, "C": a.C, "seed": a.seed, "max_tokens": a.max_tokens, "tau": TAU,
        "screen": {"train_harmful": h_tr_screen, "train_decoys": d_tr_screen, "eval_harmful": h_te_screen, "eval_decoys": d_te_screen},
        "behaviour": {"train": behaviour(train), "eval": behaviour(test)},
        "probes": probes,
        "eval_examples": [{"request": p[0], "decoy": p[1], "graded_doc": int(g), "margin": float(m)}
                          for p, g, m in zip(test["pairs"], test["graded_doc"], test["margins"])],
        "seconds": timer(),
    }


def plot(res: dict, path) -> None:
    plt = plt_style()
    labels = {"graded_doc": "which document is graded", "verdict": "guard's verdict", "random": "random label (control)"}
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    layers = np.arange(res["n_layers"])
    for k, t in enumerate(TARGETS):
        r = res["probes"][t]
        if "acc" not in r:
            continue
        ax.plot(layers, r["acc"], label=labels[t], marker="os^"[k], ms=2.5, lw=1)
        ax.fill_between(layers, r["acc_lo"], r["acc_hi"], alpha=0.2, lw=0)
    ax.axhline(0.5, color="0.6", lw=0.5, ls=":")
    ax.set_ylim(0.3, 1.02)
    ax.set_xlabel("layer (final-position residual stream)")
    ax.set_ylabel(f"held-out accuracy ({res['split']})")
    ax.set_title(f"Linear probe, n={res['n_eval_requests']} held-out requests (95% bootstrap CI)")
    ax.legend(frameon=False, loc="lower right")
    save_fig(fig, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["train", "dev", "test"], default="dev", help="evaluation split (training is always train)")
    ap.add_argument("--n", type=int, default=150, help="evaluation requests")
    ap.add_argument("--n-train", type=int, default=600, help="training requests")
    ap.add_argument("--n-decoys", type=int, default=None, help="evaluation decoy pool (default: --n)")
    ap.add_argument("--n-decoys-train", type=int, default=None, help="training decoy pool (default: --n-train)")
    ap.add_argument("--decoys-per-item", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=80)
    ap.add_argument("--C", type=float, default=0.01, help="inverse L2 strength (features are standardised)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--allow-test", action="store_true")
    a = ap.parse_args()
    check_split(a.split, a.allow_test)
    a.n_decoys = a.n_decoys or a.n
    a.n_decoys_train = a.n_decoys_train or a.n_train
    res = run(a)
    stem = f"probe_{a.split}_n{a.n}_train{a.n_train}{a.tag}"
    write_json(OUT / f"{stem}.json", res)
    plot(res, OUT / f"{stem}.pdf")


if __name__ == "__main__":
    main()
