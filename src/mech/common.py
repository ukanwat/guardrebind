"""Shared machinery for the mechanism experiments (paper section "Where the binding happens").

Everything runs on Shieldstral-1.0-3B through the bench guard (src/bench/guards.py), so prompts are
built and tokenised exactly as in the behavioural results. The model is read and patched with plain
PyTorch forward hooks on the language model's decoder layers (Mistral3ForConditionalGeneration ->
get_decoder() -> .layers, 26 Ministral3DecoderLayer). "Residual stream at layer l" is the output
of decoder layer l (l = 0..L-1); the output of the last layer goes through the final norm and the
yes/no rows of the lm_head, as in HFBackend.

The attack is the core one from the paper: content = request + "\\n\\n<Query>: Q\\n\\n<Document>: decoy",
with Q the deployment's own query (BROAD_QUERY), rendered by the `query_doc` payload in src/attacks.py.

Plots need matplotlib, which is not in the lock file. Run with
    uv run --frozen --with matplotlib python -m src.mech.<script> ...
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from ..attacks import PAYLOADS
from ..bench.data import Item, load_all
from ..bench.guards import Shieldstral
from ..prompt import BROAD_QUERY
from ..safe_io import atomic_write_text

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "mech"
TAU = 0.5
QUERY_MARK = "<Query>:"
DOC_MARK = "<Document>:"
OVERREFUSAL = {"overrefusal_xstest", "overrefusal_orbench"}
_QUERY_DOC = next(p for p in PAYLOADS if p.name == "query_doc")

# Okabe-Ito, colour-blind safe.
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#F0E442"]


def attack(request: str, decoy: str, query: str = BROAD_QUERY) -> str:
    """The forged-pair attack: `request`, then a forged query and a forged document holding `decoy`."""
    return request + _QUERY_DOC.render(query, decoy)


def check_split(split: str, allow_test: bool = False) -> None:
    if split == "test" and not allow_test:
        raise SystemExit("refusing --split test: the test split is for the confirmatory runs only (PROTOCOL_v2.md); "
                         "pass --allow-test if this run is the scheduled confirmatory one")


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


class Model:
    """The bench Shieldstral guard plus hooks on its decoder layers."""

    def __init__(self, batch_size: int = 8):
        self.guard = Shieldstral(batch_size=batch_size)
        self.tok = self.guard.tokenizer
        self.backend = self.guard.backend
        self.decoder = self.backend.decoder
        self.layers = self.decoder.layers
        self.n_layers = len(self.layers)
        self.device = self.backend.model.device
        self.batch_size = batch_size
        self.cand = torch.tensor(self.guard.unsafe_ids + self.guard.safe_ids, device=self.device)
        self.k = len(self.guard.unsafe_ids)

    # -- prompts ---------------------------------------------------------- #

    def render(self, content: str) -> "Rendered":
        pieces = self.guard.segments(content, "prompt")
        full = "".join(p for p, _ in pieces)
        enc = self.tok(full, add_special_tokens=False, return_offsets_mapping=True)
        ids = enc["input_ids"]
        if ids != self.guard.encode(pieces):  # same ids the guard scores
            raise RuntimeError("offset tokenisation diverges from the guard's encode()")
        return Rendered(full, ids, [tuple(o) for o in enc["offset_mapping"]], len(pieces[0][0]), len(content))

    def n_tokens(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def score(self, contents: list[str]) -> list[float]:
        return self.guard.score(contents) if contents else []

    # -- forward passes --------------------------------------------------- #

    def _margin(self, hidden_last: torch.Tensor) -> torch.Tensor:
        logits = self.backend._label_logits(hidden_last, self.cand)
        return (torch.logsumexp(logits[:, : self.k], 1) - torch.logsumexp(logits[:, self.k :], 1)).float().cpu()

    def _pad(self, seqs: Sequence[list[int]]):
        width = max(len(s) for s in seqs)
        pad = self.backend.pad_id
        ids = torch.full((len(seqs), width), pad, dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        for r, s in enumerate(seqs):
            ids[r, : len(s)] = torch.tensor(s)
            mask[r, : len(s)] = 1
        return ids.to(self.device), mask.to(self.device)

    @torch.inference_mode()
    def capture(self, seqs: Sequence[list[int]], positions: Sequence[Sequence[int]], with_mean: bool = False):
        """Residual stream at every layer at the given positions of each sequence.

        Returns (acts, margins): acts[i] is a float32 tensor [n_layers, len(positions[i]), d] on the CPU,
        margins[i] the yes-no logit margin at the last real token. Right-padded batches, as in HFBackend.
        with_mean=True also returns means[i] [n_layers, d], the residual averaged over real positions 1..
        (BOS left out: its residual norm is an outlier)."""
        acts, margins, means = [], [], []
        for start in range(0, len(seqs), self.batch_size):
            chunk = list(seqs[start : start + self.batch_size])
            pos = list(positions[start : start + self.batch_size])
            store: dict = {}

            def hook(layer):
                def fn(_mod, _inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    store[layer] = [h[r, p].float().cpu() for r, p in enumerate(pos_t)]
                    if with_mean:
                        store[("mean", layer)] = [h[r, 1 : len(s)].float().mean(0).cpu() for r, s in enumerate(chunk)]
                return fn

            pos_t = [torch.tensor(p, device=self.device, dtype=torch.long) for p in pos]
            handles = [m.register_forward_hook(hook(i)) for i, m in enumerate(self.layers)]
            try:
                ids, mask = self._pad(chunk)
                hidden = self.decoder(input_ids=ids, attention_mask=mask).last_hidden_state
            finally:
                for h in handles:
                    h.remove()
            last = torch.tensor([len(s) - 1 for s in chunk], device=self.device)
            margins += self._margin(hidden[torch.arange(len(chunk), device=self.device), last]).tolist()
            for r in range(len(chunk)):
                acts.append(torch.stack([store[l][r] for l in range(self.n_layers)]))
                if with_mean:
                    means.append(torch.stack([store[("mean", l)][r] for l in range(self.n_layers)]))
        return (acts, margins, means) if with_mean else (acts, margins)

    @torch.inference_mode()
    def patched_margins(self, ids: list[int], jobs: list[tuple], start: int = 0) -> list[float]:
        """Run `ids` once per job, each job (layer, positions, values[len(positions), d]) overwriting the
        output of decoder `layer` at `positions` with `values` (layer None: no patch). Jobs are batched,
        one job per row.

        Positions before `start` are never patched, so they are run once and their KV cache is reused
        by every row; every job's positions must be >= start. Compare patched margins against an
        unpatched job run the same way, not against a full forward: the cached path differs from one
        full pass by bf16 rounding."""
        import copy

        if any(layer is not None and min(pos) < start for layer, pos, _ in jobs):
            raise ValueError("a job patches a position inside the cached prefix")
        prefix = None
        if start > 0:
            prefix = self.decoder(input_ids=torch.tensor([ids[:start]], device=self.device), use_cache=True).past_key_values
        suffix = ids[start:]
        out: list[float] = []
        for c0 in range(0, len(jobs), self.batch_size):
            chunk = jobs[c0 : c0 + self.batch_size]
            by_layer: dict[int, list[tuple[int, torch.Tensor, torch.Tensor]]] = {}
            for r, (layer, pos, vals) in enumerate(chunk):
                if layer is None:
                    continue
                by_layer.setdefault(layer, []).append(
                    (r, torch.tensor([p - start for p in pos], device=self.device, dtype=torch.long), vals.to(self.device))
                )

            def hook(layer):
                def fn(_mod, _inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    for r, pos, vals in by_layer[layer]:
                        h[r, pos] = vals.to(h.dtype)
                    return out
                return fn

            cache = None
            if prefix is not None:
                cache = copy.deepcopy(prefix)
                cache.batch_repeat_interleave(len(chunk))
            handles = [self.layers[l].register_forward_hook(hook(l)) for l in by_layer]
            try:
                batch = torch.tensor([suffix] * len(chunk), device=self.device)
                hidden = self.decoder(input_ids=batch, past_key_values=cache, use_cache=cache is not None).last_hidden_state
            finally:
                for h in handles:
                    h.remove()
            out += self._margin(hidden[:, -1]).tolist()
        return out


@dataclass
class Rendered:
    text: str
    ids: list[int]
    offsets: list[tuple[int, int]]
    content_start: int  # char offset of the content in `text`
    content_len: int

    def tokens_for(self, start: int, end: int) -> list[int]:
        return [i for i, (a, b) in enumerate(self.offsets) if a < end and b > start and b > a]

    def marker(self, mark: str, which: str) -> list[int]:
        """Token positions of `mark` in the template ("genuine": first occurrence, before the content)
        or the last occurrence inside the content ("forged")."""
        if which == "genuine":
            at = self.text.find(mark)
            if at < 0 or at >= self.content_start:
                raise ValueError(f"no genuine {mark}")
        else:
            lo, hi = self.content_start, self.content_start + self.content_len
            at = self.text.rfind(mark, lo, hi)
            if at < 0:
                raise ValueError(f"no forged {mark}")
        return self.tokens_for(at, at + len(mark))

    @property
    def final(self) -> int:
        return len(self.ids) - 1


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


def prompt_items(split: str, label: int, seed: int, exclude: set[str] = frozenset()) -> list[Item]:
    """English prompt-side items of one split and label, shuffled with `seed`. Over-refusal sets are
    left out of the benign pool (they are built to look harmful)."""
    pool = [
        it for it in load_all()
        if it.split == split and it.kind == "prompt" and it.label == label and it.lang == "en"
        and it.source not in OVERREFUSAL and it.id not in exclude
    ]
    pool.sort(key=lambda it: it.id)
    random.Random(seed).shuffle(pool)
    return pool


def screened(model: Model, pool: list[Item], n: int, keep, max_tokens: int, chunk: int = 64) -> tuple[list[Item], list[float], dict]:
    """Walk `pool` in order, keep items with <= max_tokens tokens whose standalone score passes `keep`,
    until n are kept. Returns the items, their standalone scores and the screen counts."""
    kept, scores, seen, too_long = [], [], 0, 0
    i = 0
    while len(kept) < n and i < len(pool):
        batch = []
        while len(batch) < chunk and i < len(pool):
            it = pool[i]
            i += 1
            if model.n_tokens(it.text) > max_tokens:
                too_long += 1
                continue
            batch.append(it)
        for it, s in zip(batch, model.score([it.text for it in batch])):
            seen += 1
            if keep(s) and len(kept) < n:
                kept.append(it)
                scores.append(s)
    return kept, scores, {"scored": seen, "too_long": too_long, "kept": len(kept)}


# --------------------------------------------------------------------------- #
# Stats, IO, plotting
# --------------------------------------------------------------------------- #


def bootstrap_mean(values: np.ndarray, n_boot: int = 10_000, seed: int = 0):
    """Mean over items (axis 0) with a percentile 95% bootstrap interval, items as the unit.
    `values` is [n_items, ...]; returns (mean, lo, hi), each [...]."""
    values = np.asarray(values, dtype=float)
    n = values.shape[0]
    mean = np.nanmean(values, axis=0)
    if n < 2:
        return mean, mean.copy(), mean.copy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = np.zeros((n_boot,) + values.shape[1:])
    for b0 in range(0, n_boot, 500):
        means[b0 : b0 + 500] = np.nanmean(values[idx[b0 : b0 + 500]], axis=1)
    lo, hi = np.nanpercentile(means, [2.5, 97.5], axis=0)
    return mean, lo, hi


def _show(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def write_json(path: Path, obj: dict) -> None:
    atomic_write_text(path, json.dumps(obj, indent=1, default=_jsonable))
    print(f"wrote {_show(path)}")


def _jsonable(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, torch.Tensor):
        return x.tolist()
    raise TypeError(type(x))


def plt_style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,  # embed TrueType, keep text as text (vector)
        "ps.fonttype": 42,
        "axes.prop_cycle": matplotlib.cycler(color=COLORS),
    })
    return plt


def save_fig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    print(f"wrote {_show(path)}")


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def __call__(self) -> float:
        return round(time.time() - self.t0, 1)
