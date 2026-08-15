"""Shieldstral scorer.

Reproduces the model card's scoring: one forward pass, read the yes/no logits at
the final position, renormalise across just those two.

    score = exp(z_yes) / (exp(z_yes) + exp(z_no))

`score` is P(the Query is true of the Document). With a safety Query ("does this
promote violence?"), high score = flagged.
"""

from __future__ import annotations

import torch

from .prompt import build_messages

MODEL_ID = "mistralai/Shieldstral-1.0-3B"

# The card scores a single token. Tokenisers differ on whether "yes" leads with a
# space, so we collect the first-token id of every surface form and sum the mass
# per class before renormalising. Reduces to the card's formula when only one
# form is present.
YES_FORMS = ["yes", "Yes", "YES"]
NO_FORMS = ["no", "No", "NO"]


def _first_token_ids(tokenizer, words: list[str]) -> list[int]:
    ids: set[int] = set()
    for word in words:
        for form in (word, " " + word):
            encoded = tokenizer.encode(form, add_special_tokens=False)
            if encoded:
                ids.add(encoded[0])
    return sorted(ids)


class ShieldstralScorer:
    def __init__(self, model_id: str = MODEL_ID, device: str | None = None):
        from transformers import AutoTokenizer

        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = self._load(model_id)
        self.model.eval()

        self.yes_ids = _first_token_ids(self.tokenizer, YES_FORMS)
        self.no_ids = _first_token_ids(self.tokenizer, NO_FORMS)
        if not self.yes_ids or not self.no_ids:
            raise RuntimeError("could not resolve yes/no token ids for this tokenizer")

    def _load(self, model_id: str):
        """Shieldstral is multimodal (mistral3). Class name varies by transformers
        version, so try the specific class first and fall back."""
        import transformers

        dtype = torch.bfloat16
        errors = []
        for cls_name in (
            "Mistral3ForConditionalGeneration",
            "AutoModelForImageTextToText",
            "AutoModelForCausalLM",
        ):
            cls = getattr(transformers, cls_name, None)
            if cls is None:
                continue
            try:
                return cls.from_pretrained(
                    model_id, dtype=dtype, device_map=self.device
                )
            except Exception as exc:  # noqa: BLE001 - we want the last error reported
                errors.append(f"{cls_name}: {exc}")
        raise RuntimeError("failed to load model:\n" + "\n".join(errors))

    def _encode(self, instruct: str, query: str, document: str) -> torch.Tensor:
        """transformers>=5 returns a BatchEncoding here, older versions a bare
        tensor. Normalise both to input_ids."""
        encoded = self.tokenizer.apply_chat_template(
            build_messages(instruct, query, document),
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if isinstance(encoded, torch.Tensor):
            return encoded
        return encoded["input_ids"]

    @torch.inference_mode()
    def score(self, instruct: str, query: str, document: str) -> float:
        ids = self._encode(instruct, query, document).to(self.model.device)
        logits = self.model(input_ids=ids).logits[0, -1].float()
        yes = torch.logsumexp(logits[self.yes_ids], dim=0)
        no = torch.logsumexp(logits[self.no_ids], dim=0)
        return torch.sigmoid(yes - no).item()

    def token_length(self, instruct: str, query: str, document: str) -> int:
        return int(self._encode(instruct, query, document).shape[-1])
