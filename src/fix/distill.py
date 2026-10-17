"""Segment-max self-distillation for Shieldstral.

The teacher is the frozen base guard reading each piece of the content on its
own, in the stock format, with no attack around it. The student reads the whole
(possibly attacked) content and must give the highest of those scores:

    target(content) = max(teacher(content), teacher(seg) for seg in segments)

Clean content is its own only segment, so clean inputs distill to the teacher's
own score and calibration is kept. Attacked content has the original text and
the decoy as segments, so a harmful original keeps its high score no matter
what is appended. The whole content is always included, so harm that only shows
up in combination (Prompt Overflow) is never scored below the teacher's own
read of the full text.

Two conditions share this code: T (stock input format) and B+T (bounded format,
src/fix/inputs.py). B alone is the base model with the bounded format.

    uv run python -m src.fix.distill build --out data/fix/train.jsonl
    uv run python -m src.fix.distill train --data data/fix/train.jsonl --fmt bounded --out models/fix/bt

Round 2: `build --decoy-split train` draws each attacked example's decoy and forged query from the
TRAIN halves of the pools (attacks_grammar.POOL_SPLIT), seeded by the item id; the default (fixed)
is the original build, byte for byte. The HELDOUT halves are kept for evaluate / adaptive.

Other guards (T only): `build --guard llama_guard_3_1b` scores segments with that
guard and writes the attacks in its syntax; `train --guard llama_guard_3_1b
--fmt stock` trains its own model on its own rendering and label tokens.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from ..prompt import BROAD_QUERY, DEFAULT_INSTRUCT
from ..safe_io import atomic_write_text, read_jsonl
from ..scorer import ShieldstralScorer
from .inputs import bounded_ids, split_ids, stock_ids

FORMATS = {"stock": stock_ids, "split": split_ids, "bounded": bounded_ids}


@dataclass
class Example:
    id: str
    document: str        # the full content the student reads
    segments: list[str]  # pieces the teacher scores on their own
    target: float = -1.0
    variant: str = "clean"


# --- building -----------------------------------------------------------------

BUILD_MODES = ("segment_max", "naive_at", "clean_copy")


def _variants(item, attacks, guard, rng: random.Random, k: int, mode: str = "segment_max",
              decoy_split: str = "fixed") -> list[Example]:
    from ..bench.attacks_grammar import draw_split

    out = [Example(f"{item.id}|clean", item.text, [item.text])]
    for attack in rng.sample(attacks, min(k, len(attacks))):
        # decoy_split="train": the decoy and forged query are redrawn from the TRAIN halves of their pools
        # (attacks_grammar.POOL_SPLIT), seeded by the item id; "fixed" keeps the attack's own (index 0).
        # The draw has its own RNG, so the items and attacks sampled are the same as the fixed build's.
        attack = draw_split(attack, decoy_split, item.id)
        document = attack.apply(item.text, guard)
        decoy = attack.params.get("decoy_text")
        segments = [document, item.text] + ([decoy] if decoy else [])
        if mode == "clean_copy":
            # Ablation (clean-context distillation, as in SecOPD/RAISED): the teacher sees only the
            # original item, so whatever the forged pair adds, including a harmful decoy, is ignored.
            segments = [item.text]
        out.append(Example(f"{item.id}|{attack.name}", document, segments, variant=attack.name))
    if mode == "naive_at":
        # Hard labels (PROTOCOL_v2 F6): the item's ground-truth label for every
        # variant of it (clean or attacked), so attacked-harmful and clean-harmful
        # target 1.0 and clean-benign and attacked-benign target 0.0. No teacher.
        for ex in out:
            ex.target = float(item.label)
    return out


def label_targets(examples: list[Example], guard, cache_path: Path, batch: int = 32) -> None:
    """Fill target = max teacher score over segments. Each unique segment is
    scored once, batched, by the same guard class the evaluation uses. Scores
    are appended to cache_path as they come in, so an interrupted build resumes
    where it stopped."""
    cache: dict[str, float] = {}
    for row in read_jsonl(cache_path):
        cache[row["text"]] = row["score"]
    todo = sorted({seg for ex in examples for seg in ex.segments} - cache.keys(), key=len)
    print(f"cached {len(cache)}, to score {len(todo)}", flush=True)
    t0 = time.time()
    with cache_path.open("a") as f:
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            for text, score in zip(chunk, guard.score(chunk)):
                cache[text] = score
                f.write(json.dumps({"text": text, "score": score}) + "\n")
            f.flush()
            if (i // batch) % 10 == 0:
                print(f"scored {i + len(chunk)}/{len(todo)} segments {time.time() - t0:.0f}s", flush=True)
    for ex in examples:
        ex.target = max(cache[seg] for seg in ex.segments)


def build_examples(per_item: int, max_items: int | None, seed: int, max_chars: int,
                   mode: str, guard: str = "shieldstral", decoy_split: str = "fixed") -> list[Example]:
    """Construct the (unlabeled for segment_max, hard-labeled for naive_at)
    examples. No model is loaded here; segment_max targets are filled later by
    label_targets. Identical example structure across modes (checked in tests).
    Attacks are written in `guard`'s own syntax (GUARD_SPECS[guard]); the item
    sample and attack draws do not depend on the guard."""
    from ..bench.attacks_grammar import grammar, split_attacks
    from ..bench.data import load_all, structured_benign
    from ..bench.attacks_grammar import GUARD_SPECS

    rng = random.Random(seed)
    # Very long prompts (some adversarial WildGuardMix rows run to thousands of
    # tokens) dominate scoring time on MPS and are rare at deployment; cap them.
    items = [it for it in load_all()
             if it.split == "train" and it.kind == "prompt" and len(it.text) <= max_chars]
    rng.shuffle(items)
    items = items[:max_items] if max_items else items
    train_attacks, _ = split_attacks(grammar(core=False))
    train_attacks = [a for a in train_attacks if a.params.get("structure") != "none"]

    examples: list[Example] = []
    for item in items:
        examples += _variants(item, train_attacks, GUARD_SPECS[guard], rng, per_item, mode, decoy_split)
    for j, sb in enumerate(structured_benign(len(items) // 4)):
        text = sb.text
        ex = Example(f"structured_benign:{j}", text, [text], variant="hard_negative")
        if mode == "naive_at":
            ex.target = 0.0  # structured benign is a hard negative
        examples.append(ex)
    return examples


def build(out: Path, per_item: int = 3, max_items: int | None = None, seed: int = 0,
          max_chars: int = 1200, mode: str = "segment_max", cache: Path | None = None,
          guard: str = "shieldstral", device: str | None = None, decoy_split: str = "fixed") -> None:
    if mode not in BUILD_MODES:
        raise ValueError(f"mode must be one of {BUILD_MODES}")
    if decoy_split not in ("fixed", "train"):
        raise ValueError("decoy_split must be fixed or train (heldout decoys are for evaluation only)")
    examples = build_examples(per_item, max_items, seed, max_chars, mode, guard, decoy_split)

    if mode in ("segment_max", "clean_copy"):
        from ..bench.guards import load_guard
        # Teacher scores live in a cache keyed by text; pass --cache to reuse another build's
        # forward passes (clean_copy segments are a subset of segment_max's).
        label_targets(examples, load_guard(guard, batch_size=16, device=device),
                      cache or out.with_suffix(".teacher_cache.jsonl"), batch=16)
    # naive_at targets are hard labels, already set in build_examples (no teacher).

    atomic_write_text(out, "".join(json.dumps(asdict(ex)) + "\n" for ex in examples))
    print(f"wrote {len(examples)} examples ({mode}) -> {out}")


# --- training -----------------------------------------------------------------

def _batches(rows, tokenizer, fmt, batch: int, max_len: int, rng):
    rng.shuffle(rows)
    build_ids = FORMATS[fmt]
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        seqs = [build_ids(tokenizer, DEFAULT_INSTRUCT, BROAD_QUERY, r["document"])[-max_len:] for r in chunk]
        width = max(len(s) for s in seqs)
        pad = tokenizer.pad_token_id or 0
        ids = torch.full((len(seqs), width), pad, dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        for j, s in enumerate(seqs):  # right padding; read logits at each row's last real token
            ids[j, :len(s)] = torch.tensor(s)
            mask[j, :len(s)] = 1
        last = torch.tensor([len(s) - 1 for s in seqs])
        target = torch.tensor([r["target"] for r in chunk], dtype=torch.float32)
        yield ids, mask, last, target


CKPT_EVERY = int(__import__("os").environ.get("CKPT_EVERY", 50))


class GuardStudent:
    """Any HF guard but Shieldstral as the student (T, stock format only): its
    model + LoRA, its own rendering (guard.build_ids, the ids its scorer reads)
    and its own label tokens (guard.unsafe_ids vs guard.safe_ids). The margin is
    computed exactly as HFGuard.score_ids does, from the decoder's last hidden
    state and only the label rows of the head, in fp32, so at init (LoRA B = 0)
    sigmoid(margin) is the guard's own score (src/fix/test_generality.py)."""

    def __init__(self, guard: str, device: str | None = None, dtype: torch.dtype | None = None):
        from ..bench.guards import load_guard

        self.guard = load_guard(guard, device=device, dtype=dtype)
        self.model = self.guard.backend.model
        self.device = self.model.device
        self.yes_ids, self.no_ids = self.guard.unsafe_ids, self.guard.safe_ids
        self.pad_id = self.guard.backend.pad_id

    def ids(self, document: str) -> list[int]:
        return self.guard.build_ids([document], kind="prompt")[0]

    def margin(self, model, ids, mask, last) -> torch.Tensor:
        """unsafe-vs-safe logit margin at each row's last real token. `model` is
        the (possibly peft-wrapped) model whose decoder and head are used."""
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        hidden = base.get_decoder()(input_ids=ids, attention_mask=mask).last_hidden_state
        h = hidden[torch.arange(len(last), device=ids.device), last.to(ids.device)]
        cand = torch.tensor(self.yes_ids + self.no_ids, device=ids.device)
        logits = self.guard.backend._label_logits(h, cand)  # fp32, + bias / softcap if the model has them
        k = len(self.yes_ids)
        return torch.logsumexp(logits[:, :k], -1) - torch.logsumexp(logits[:, k:], -1)


def _guard_batches(rows, student: GuardStudent, batch: int, rng):
    """_batches for a GuardStudent on rows with precomputed "ids": same seeded
    shuffle and right padding, no truncation."""
    rng.shuffle(rows)
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        seqs = [r["ids"] for r in chunk]
        width = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), width), student.pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        for j, s in enumerate(seqs):
            ids[j, :len(s)] = torch.tensor(s)
            mask[j, :len(s)] = 1
        last = torch.tensor([len(s) - 1 for s in seqs])
        target = torch.tensor([r["target"] for r in chunk], dtype=torch.float32)
        yield ids, mask, last, target


def train_guard(data: Path, guard: str, out: Path, epochs: int = 1, batch: int = 8, lr: float = 1e-4,
                rank: int = 16, max_len: int = 2048, seed: int = 0, device: str | None = None,
                max_steps: int | None = None, dtype: torch.dtype | None = None) -> None:
    """T (stock format, LoRA) for a guard other than Shieldstral. Same loss,
    schedule, LoRA config, checkpoint/resume and DONE marker as train()."""
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

    torch.manual_seed(seed)
    random.seed(seed)

    rows = [json.loads(line) for line in data.open()]
    student = GuardStudent(guard, device=device, dtype=dtype)
    # The guard's prompt starts with its template and ends in its answer prefix, so ids are never
    # truncated (a cut would drop part of what the scorer reads). Checked up front, not mid-run.
    for r in rows:
        r["ids"] = student.ids(r["document"])
    longest = max(len(r["ids"]) for r in rows)
    if longest > max_len:
        raise SystemExit(f"{sum(len(r['ids']) > max_len for r in rows)} rows over --max-len {max_len} tokens "
                         f"(longest {longest})")
    print(f"{len(rows)} rows, longest {longest} tokens", flush=True)
    model = get_peft_model(student.model, LoraConfig(
        r=rank, lora_alpha=2 * rank, lora_dropout=0.0,
        target_modules=r".*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
    ))
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    steps = epochs * math.ceil(len(rows) / batch)
    if max_steps:
        steps = min(steps, max_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * max(0.0, 1 - s / steps))
    device = student.device
    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "train_log.jsonl").open("a")

    ckpt_path = out / "ckpt.pt"
    done = 0
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        set_peft_model_state_dict(model, ck["lora"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        done = ck["step"]
        print(f"resumed from step {done}", flush=True)

    def save_ckpt(at: int) -> None:
        tmp = out / "ckpt.tmp"
        torch.save({"lora": get_peft_model_state_dict(model), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "step": at, "guard": guard}, tmp)
        tmp.replace(ckpt_path)

    step, t0 = 0, time.time()
    for epoch in range(epochs):
        for ids, mask, last, target in _guard_batches(rows, student, batch, rng):
            if step >= steps:
                break
            if step < done:
                step += 1
                continue
            ids, mask, target = ids.to(device), mask.to(device), target.to(device)
            margin = student.margin(model, ids, mask, last)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(margin, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 10 == 0 or step == steps:
                rec = {"step": step, "of": steps, "loss": round(loss.item(), 5), "s": round(time.time() - t0)}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(rec, flush=True)
            if step % CKPT_EVERY == 0 or step == steps:
                save_ckpt(step)
    model.save_pretrained(out)
    # Written last: the queue treats a model as trained only when this marker exists.
    atomic_write_text(out / "DONE", f"{step} steps\n")
    print(f"saved -> {out}")


def train(data: Path, fmt: str, out: Path, epochs: int = 1, batch: int = 8, lr: float = 1e-4,
          rank: int = 16, max_len: int = 384, seed: int = 0) -> None:
    from peft import LoraConfig, get_peft_model

    # Reproducible per seed: torch (init/dropout/shuffle) and python RNG.
    torch.manual_seed(seed)
    random.seed(seed)

    rows = [json.loads(line) for line in data.open()]
    scorer = ShieldstralScorer()
    model = get_peft_model(scorer.model, LoraConfig(
        r=rank, lora_alpha=2 * rank, lora_dropout=0.0,
        target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
    ))
    boundary_delta = None
    if fmt == "bounded":
        # The boundary ids are unused in pretraining, so their embeddings must learn. Making the
        # whole embedding matrix trainable would cost a 131k x d dense gradient (~0.8GB) per
        # step, so a 2-row delta is added to those two ids' embeddings instead.
        emb_mod = model.get_input_embeddings()
        boundary_delta = torch.nn.Parameter(torch.zeros(2, emb_mod.weight.shape[1],
                                                        device=emb_mod.weight.device, dtype=torch.float32))

        def add_delta(module, args, output):
            ids_in = args[0]
            d = boundary_delta.to(output.dtype)
            return output + (ids_in == 36).unsqueeze(-1) * d[0] + (ids_in == 37).unsqueeze(-1) * d[1]

        emb_mod.register_forward_hook(add_delta)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    if boundary_delta is not None:
        params.append(boundary_delta)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    steps = epochs * math.ceil(len(rows) / batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * max(0.0, 1 - s / steps))
    yes, no = scorer.yes_ids, scorer.no_ids
    device = scorer.device
    decoder = model.get_base_model().get_decoder()
    head = model.get_base_model().get_output_embeddings()
    cand = torch.tensor(list(yes) + list(no), device=device)
    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "train_log.jsonl").open("a")

    # Resumable: a checkpoint every CKPT_EVERY steps holds the LoRA weights, the
    # boundary rows, optimizer and scheduler state. On restart, the same seeded
    # shuffle is replayed and the batches already trained on are skipped.
    from peft import get_peft_model_state_dict, set_peft_model_state_dict

    ckpt_path = out / "ckpt.pt"
    done = 0
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        set_peft_model_state_dict(model, ck["lora"])
        if fmt == "bounded":
            with torch.no_grad():
                boundary_delta.copy_(ck["boundary_delta"].to(boundary_delta.device))
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        done = ck["step"]
        print(f"resumed from step {done}", flush=True)

    def save_ckpt(at: int) -> None:
        tmp = out / "ckpt.tmp"
        torch.save({
            "lora": get_peft_model_state_dict(model), "opt": opt.state_dict(), "sched": sched.state_dict(),
            "step": at,
            "boundary_delta": boundary_delta.detach().cpu() if boundary_delta is not None else None,
        }, tmp)
        tmp.replace(ckpt_path)

    step, t0 = 0, time.time()
    for epoch in range(epochs):
        for ids, mask, last, target in _batches(rows, scorer.tokenizer, fmt, batch, max_len, rng):
            if step < done:
                step += 1
                continue
            ids, mask, target = ids.to(device), mask.to(device), target.to(device)
            # Only the last real position and only the yes/no rows of the output layer are
            # needed. Full-vocab logits for every position cost ~0.8GB per batch on MPS.
            hidden = decoder(input_ids=ids, attention_mask=mask).last_hidden_state
            h = hidden[torch.arange(len(last), device=device), last.to(device)]
            final = (h @ head.weight[cand].T).float()
            margin = torch.logsumexp(final[:, :len(yes)], -1) - torch.logsumexp(final[:, len(yes):], -1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(margin, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 10 == 0:
                rec = {"step": step, "of": steps, "loss": round(loss.item(), 5), "s": round(time.time() - t0)}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(rec, flush=True)
            if step % CKPT_EVERY == 0:
                save_ckpt(step)
    model.save_pretrained(out)
    if fmt == "bounded":
        # Saved as the full trained rows (base + delta), which is what ShieldstralFixed loads.
        rows = model.get_input_embeddings().weight[[36, 37]].detach().float() + boundary_delta.detach()
        torch.save(rows.to(torch.bfloat16).cpu(), out / "boundary_embeddings.pt")
    # Written last: the queue treats a model as trained only when this marker exists.
    atomic_write_text(out / "DONE", f"{step} steps\n")
    print(f"saved -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--per-item", type=int, default=3)
    b.add_argument("--max-items", type=int)
    b.add_argument("--mode", choices=list(BUILD_MODES), default="segment_max",
                   help="segment_max (self-distillation), naive_at (hard labels, PROTOCOL_v2 F6) or clean_copy "
                        "(teacher on the original item only; ablation)")
    b.add_argument("--cache", type=Path, help="teacher score cache (default: next to --out)")
    b.add_argument("--guard", default="shieldstral", help="teacher and attack syntax (a load_guard name)")
    b.add_argument("--device", help="default: cuda, then mps, then cpu")
    b.add_argument("--decoy-split", choices=["fixed", "train"], default="fixed",
                   help="fixed: each attack's own decoy/query (index 0, the original build); train: per attacked "
                        "example, a seeded draw from the TRAIN halves of the decoy and query pools")
    t = sub.add_parser("train")
    t.add_argument("--data", type=Path, required=True)
    t.add_argument("--fmt", choices=list(FORMATS), required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--epochs", type=int, default=1)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--guard", default="shieldstral",
                   help="student guard; other than shieldstral only --fmt stock (T) is supported")
    t.add_argument("--device", help="non-shieldstral guards only (default: cuda, then mps, then cpu)")
    t.add_argument("--max-len", type=int, default=2048,
                   help="non-shieldstral guards only: longest input in tokens (never truncated)")
    t.add_argument("--max-steps", type=int, help="non-shieldstral guards only: stop after this many steps")
    t.add_argument("--fp32", action="store_true", help="non-shieldstral guards only: fp32 weights (CPU tests)")
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.out, a.per_item, a.max_items, mode=a.mode, cache=a.cache, guard=a.guard, device=a.device,
              decoy_split=a.decoy_split)
    elif a.guard == "shieldstral":
        train(a.data, a.fmt, a.out, a.epochs, a.batch, a.lr, seed=a.seed)
    else:
        if a.fmt != "stock":
            raise SystemExit(f"--guard {a.guard}: only --fmt stock (condition T) is supported")
        train_guard(a.data, a.guard, a.out, a.epochs, a.batch, a.lr, max_len=a.max_len, seed=a.seed,
                    device=a.device, max_steps=a.max_steps, dtype=torch.float32 if a.fp32 else None)


if __name__ == "__main__":
    main()
