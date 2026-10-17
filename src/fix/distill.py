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

def _variants(item, attacks, guard, rng: random.Random, k: int) -> list[Example]:
    out = [Example(f"{item.id}|clean", item.text, [item.text])]
    for attack in rng.sample(attacks, min(k, len(attacks))):
        document = attack.apply(item.text, guard)
        decoy = attack.params.get("decoy_text")
        segments = [document, item.text] + ([decoy] if decoy else [])
        out.append(Example(f"{item.id}|{attack.name}", document, segments, variant=attack.name))
    return out


def label_targets(examples: list[Example], guard, cache_path: Path, batch: int = 32) -> None:
    """Fill target = max teacher score over segments. Each unique segment is
    scored once, batched, by the same guard class the evaluation uses. Scores
    are appended to cache_path as they come in, so an interrupted build resumes
    where it stopped."""
    cache: dict[str, float] = {}
    if cache_path.exists():
        for line in cache_path.open():
            row = json.loads(line)
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


def build(out: Path, per_item: int = 3, max_items: int | None = None, seed: int = 0,
          max_chars: int = 1200) -> None:
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
        examples += _variants(item, train_attacks, GUARD_SPECS["shieldstral"], rng, per_item)
    for j, sb in enumerate(structured_benign(len(items) // 4)):
        text = sb.text
        examples.append(Example(f"structured_benign:{j}", text, [text], variant="hard_negative"))

    from ..bench.guards import load_guard

    label_targets(examples, load_guard("shieldstral", batch_size=16), out.with_suffix(".teacher_cache.jsonl"), batch=16)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex)) + "\n")
    print(f"wrote {len(examples)} examples -> {out}")


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


def train(data: Path, fmt: str, out: Path, epochs: int = 1, batch: int = 8, lr: float = 1e-4,
          rank: int = 16, max_len: int = 384, seed: int = 0) -> None:
    from peft import LoraConfig, get_peft_model

    rows = [json.loads(line) for line in data.open()]
    scorer = ShieldstralScorer()
    model = get_peft_model(scorer.model, LoraConfig(
        r=rank, lora_alpha=2 * rank, lora_dropout=0.0,
        target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
    ))
    if fmt == "bounded":
        # The boundary ids are unused in pretraining, so let their embeddings learn.
        # Only those two rows may move: mask every other row's gradient.
        emb = model.get_input_embeddings()
        emb.weight.requires_grad_(True)
        keep = torch.zeros(emb.weight.shape[0], 1, device=emb.weight.device, dtype=emb.weight.dtype)
        keep[[36, 37]] = 1
        emb.weight.register_hook(lambda g: g * keep)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    steps = epochs * math.ceil(len(rows) / batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * max(0.0, 1 - s / steps))
    yes, no = scorer.yes_ids, scorer.no_ids
    device = scorer.device
    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "train_log.jsonl").open("a")

    step, t0 = 0, time.time()
    for epoch in range(epochs):
        for ids, mask, last, target in _batches(rows, scorer.tokenizer, fmt, batch, max_len, rng):
            ids, mask, target = ids.to(device), mask.to(device), target.to(device)
            logits = model(input_ids=ids, attention_mask=mask).logits
            final = logits[torch.arange(len(last)), last.to(device)].float()
            margin = torch.logsumexp(final[:, yes], -1) - torch.logsumexp(final[:, no], -1)
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
    model.save_pretrained(out)
    if fmt == "bounded":
        torch.save(model.get_input_embeddings().weight[[36, 37]].detach().cpu(), out / "boundary_embeddings.pt")
    print(f"saved -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--per-item", type=int, default=3)
    b.add_argument("--max-items", type=int)
    t = sub.add_parser("train")
    t.add_argument("--data", type=Path, required=True)
    t.add_argument("--fmt", choices=list(FORMATS), required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--epochs", type=int, default=1)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--lr", type=float, default=1e-4)
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.out, a.per_item, a.max_items)
    else:
        train(a.data, a.fmt, a.out, a.epochs, a.batch, a.lr)


if __name__ == "__main__":
    main()
