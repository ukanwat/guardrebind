"""Can we LoRA-train Shieldstral on this machine fast enough?

Loads the model, wraps the language model in LoRA, runs a few forward/backward
steps on prompts of realistic length and reports seconds per step and peak
memory. No data, no saving: this only decides local vs cloud.
"""

from __future__ import annotations

import time

import torch

from ..scorer import ShieldstralScorer


def main(steps: int = 6, batch: int = 4, seq: int = 512) -> None:
    from peft import LoraConfig, get_peft_model

    scorer = ShieldstralScorer()
    model = scorer.model
    print(type(model).__name__, scorer.device)

    config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    ids = torch.randint(1000, 30000, (batch, seq), device=scorer.device)
    yes, no = scorer.yes_ids[0], scorer.no_ids[0]

    for step in range(steps):
        t0 = time.time()
        logits = model(input_ids=ids).logits[:, -1, [yes, no]].float()
        loss = torch.nn.functional.cross_entropy(logits, torch.zeros(batch, dtype=torch.long, device=ids.device))
        loss.backward()
        opt.step()
        opt.zero_grad()
        torch.mps.synchronize()
        print(f"step {step} loss {loss.item():.4f} {time.time() - t0:.2f}s "
              f"mem {torch.mps.driver_allocated_memory() / 2**30:.1f}GB", flush=True)


if __name__ == "__main__":
    main()
