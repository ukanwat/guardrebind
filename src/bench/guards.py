"""Guard adapters for the GuardRebind harness (see SPEC.md, "Guard").

Every guard is split into two halves so a different inference engine can be
dropped in later (vLLM on CUDA):

    build_ids(texts, kind)   prompt construction + tokenisation -> list[list[int]]
    backend.next_token_logits(ids, candidate_ids) -> [n, k] float32 logits

`score()` glues them: it reads the logits of each guard's unsafe/safe label
tokens at the first generated position and renormalises over just those.

Prompt construction is always the vendor's own: the tokenizer's chat template
where the card uses one (Shieldstral, Llama Guard 3, Qwen3Guard), and the
card's raw prompt where it does not (ShieldGemma, prompt-only).

`parse_special_in_content`
    True (default) tokenises the rendered prompt in one call, exactly as HF
    does in deployment, so literal "[INST]" or "<|eot_id|>" inside the content
    becomes the real control token. False splits the rendered prompt only at
    control tokens that the *template* placed, and tokenises every other span
    (which contains all the content) with `split_special_tokens=True`. Merges
    across the content/template boundary are unchanged, so for content with no
    control-token text both modes give identical ids (checked in
    `src/bench/test_guards.py`); the only difference is the control-token
    channel.

`kind="response"` takes either a (prompt, response) tuple or the SPEC's
"prompt\\n\\nresponse" string, which is split at the first blank line.

`policy` (src/bench/policies.py, PROTOCOL_v2 D1(b))
    The deployment policy a guard prompts with: the card default (index 0 of
    `spec.policies`, identical to the prompt before policies existed) or a
    paraphrase. Pass `policy=` to the constructor or call `set_policy()` to
    switch without reloading the model.
"""

from __future__ import annotations

import re
from typing import Protocol, Sequence

import torch

from ..prompt import BROAD_QUERY, DEFAULT_INSTRUCT, build_messages
from ..scorer import MODEL_ID as SHIELDSTRAL_ID
from ..scorer import NO_FORMS, YES_FORMS, ShieldstralScorer, _first_token_ids
from .policies import (
    LLAMA_GUARD_POLICIES, QWEN3GUARD_POLICIES, SHIELDGEMMA_POLICIES, SHIELDGEMMA_PROMPT_POLICY,
    SHIELDGEMMA_RESPONSE_POLICY, SHIELDSTRAL_POLICIES, Policy,
)

Text = str | tuple[str, str]
# Private-use code points: never produced by tokenizers or real data, never
# touched by any of the chat templates.
_SENTINEL = "{}"


class Guard(Protocol):
    name: str
    native_labels: dict
    spec: object  # attacks_grammar.GUARD_SPECS[name], set by load_guard

    def score(self, texts: list[Text], kind: str = "prompt") -> list[float]: ...


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def split_response(text: Text) -> tuple[str, str]:
    if isinstance(text, tuple):
        return text
    prompt, sep, response = text.partition("\n\n")
    if not sep:
        raise ValueError("response item needs a (prompt, response) tuple or 'prompt\\n\\nresponse'")
    return prompt, response


# --------------------------------------------------------------------------- #
# Backends: the only part that touches the model.
# --------------------------------------------------------------------------- #


class HFBackend:
    """transformers forward pass, batched with RIGHT padding.

    One forward, no generation, so nothing needs the sequences right-aligned.
    Right padding keeps every real token at the same position *and* the same
    alignment within the attention kernel's key blocks as an unpadded run, and
    causal attention never lets a real token see the pads. On MPS bf16 this
    makes batched scores bit-identical to single ones. Left padding shifts the
    block alignment and moved Shieldstral logits by up to ~0.05 (score drift up
    to 8e-3), failing the 1e-3 equivalence check.

    The decoder's final hidden state is read at each row's last real token and
    only the label rows of the lm_head are applied, in fp32."""

    def __init__(self, model, pad_id: int, batch_size: int = 16):
        self.model = model
        self.decoder = model.get_decoder()
        self.head = model.get_output_embeddings()
        self.pad_id = pad_id
        self.batch_size = batch_size
        config = model.config.get_text_config()
        # Gemma 2 caps final logits; reproduce it since we bypass the model's head.
        self.softcap = getattr(config, "final_logit_softcapping", None)

    @torch.inference_mode()
    def next_token_logits(self, ids: Sequence[list[int]], candidate_ids: list[int]) -> torch.Tensor:
        device = self.model.device
        cand = torch.tensor(candidate_ids, device=device)
        # Batch by length so short inputs are not padded to the longest one; rows are
        # restored to input order at the end (right padding + mask, so scores don't change).
        order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
        ids = [ids[i] for i in order]
        out = []
        for start in range(0, len(ids), self.batch_size):
            chunk = ids[start : start + self.batch_size]
            width = max(len(x) for x in chunk)
            input_ids = torch.full((len(chunk), width), self.pad_id, dtype=torch.long)
            mask = torch.zeros((len(chunk), width), dtype=torch.long)
            for row, seq in enumerate(chunk):
                input_ids[row, : len(seq)] = torch.tensor(seq)
                mask[row, : len(seq)] = 1
            hidden = self.decoder(
                input_ids=input_ids.to(device), attention_mask=mask.to(device)
            ).last_hidden_state
            last = torch.tensor([len(seq) - 1 for seq in chunk], device=device)
            h = hidden[torch.arange(len(chunk), device=device), last]
            out.append(self._label_logits(h, cand).cpu())
        logits = torch.cat(out)
        restored = torch.empty_like(logits)
        restored[torch.tensor(order)] = logits
        return restored

    def _label_logits(self, hidden: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
        """The lm_head for just the label rows, in fp32. bf16 logits near 20 are
        quantised to 0.125, which alone moves a mid-range score by ~1e-2."""
        logits = hidden.float() @ self.head.weight[cand].float().T
        if self.head.bias is not None:
            logits = logits + self.head.bias[cand].float()
        if self.softcap:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        return logits

    def close(self) -> None:
        del self.model, self.decoder, self.head
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Shared guard machinery
# --------------------------------------------------------------------------- #


class HFGuard:
    """Base class. Subclasses define `segments()` and the label token ids."""

    name: str
    model_id: str
    native_labels: dict
    attn_implementation: str | None = None
    spec = None
    policies: tuple[Policy, ...] = ()

    def __init__(
        self,
        parse_special_in_content: bool = True,
        device: str | None = None,
        batch_size: int = 16,
        load_model: bool = True,
        policy: Policy | None = None,
    ):
        from transformers import AutoTokenizer

        self.set_policy(policy)
        self.parse_special_in_content = parse_special_in_content
        self.device = device or default_device()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        # Longest first, so "[/INST]" wins over any shorter overlapping token.
        added = sorted({t.content for t in self.tokenizer.added_tokens_decoder.values()}, key=len, reverse=True)
        self._added_re = re.compile("|".join(re.escape(t) for t in added)) if added else None
        self.unsafe_ids, self.safe_ids = self.label_ids()
        if set(self.unsafe_ids) & set(self.safe_ids):
            raise RuntimeError(f"{self.name}: unsafe/safe label tokens overlap")
        self.backend = None
        if load_model:
            pad = self.tokenizer.pad_token_id
            if pad is None:
                pad = self.tokenizer.eos_token_id or 0
            self.backend = HFBackend(self._load_model(), pad, batch_size)

    # -- to override ------------------------------------------------------- #

    def segments(self, text: Text, kind: str) -> list[tuple[str, bool]]:
        """The full prompt as ordered (string, is_content) pieces, including any
        teacher-forced answer prefix. Concatenated, it is the exact string the
        vendor's recipe would tokenise."""
        raise NotImplementedError

    def label_ids(self) -> tuple[list[int], list[int]]:
        raise NotImplementedError

    def kinds(self) -> tuple[str, ...]:
        return ("prompt", "response")

    def set_policy(self, policy: Policy | None) -> None:
        """Prompt with `policy` from now on; None is the card default."""
        self.policy = policy if policy is not None else (self.policies[0] if self.policies else None)

    # -- shared ------------------------------------------------------------ #

    def _replace_template(self, pieces: list[tuple[str, bool]], kind: str) -> list[tuple[str, bool]]:
        """Apply the policy's (old, new) task-line swaps to template text only.
        Each `old` must occur in exactly one template piece, so a template
        change cannot silently leave the default wording in place."""
        swaps = (self.policy.fields.get("replace") or {}).get(kind, ()) if self.policy else ()
        for old, new in swaps:
            hits = [i for i, (p, c) in enumerate(pieces) if not c and old in p]
            if len(hits) != 1 or pieces[hits[0]][0].count(old) != 1:
                raise RuntimeError(f"{self.name}: policy {self.policy.id} task line not found once in the template")
            i = hits[0]
            pieces[i] = (pieces[i][0].replace(old, new), False)
        return pieces

    def _load_model(self):
        from transformers import AutoModelForCausalLM

        kwargs = {"dtype": torch.bfloat16, "device_map": self.device}
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation
        return AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs).eval()

    def _template_segments(self, messages: list[dict], contents: list[str], **template_kwargs) -> list[tuple[str, bool]]:
        """Render `messages` (whose content slots hold sentinels) with the
        tokenizer's chat template, then swap the real content back in.

        Checks that the result equals rendering the real content directly, so a
        template that trims or escapes content cannot silently diverge."""
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, **template_kwargs)
        pieces: list[tuple[str, bool]] = []
        rest = rendered
        for i, content in enumerate(contents):
            before, found, rest = rest.partition(_SENTINEL.format(i))
            if not found:
                raise RuntimeError(f"{self.name}: template dropped content slot {i}")
            pieces += [(before, False), (content, True)]
        pieces.append((rest, False))

        real = _fill(messages, contents)
        direct = self.tokenizer.apply_chat_template(real, tokenize=False, **template_kwargs)
        if direct != "".join(p for p, _ in pieces):
            raise RuntimeError(f"{self.name}: chat template transforms content; sentinel render diverges")
        return pieces

    def encode(self, pieces: list[tuple[str, bool]]) -> list[int]:
        full = "".join(p for p, _ in pieces)
        if self.parse_special_in_content:
            return self.tokenizer(full, add_special_tokens=False)["input_ids"]

        spans, pos = [], 0
        for piece, is_content in pieces:
            if is_content:
                spans.append((pos, pos + len(piece)))
            pos += len(piece)

        ids: list[int] = []
        last = 0
        matches = self._added_re.finditer(full) if self._added_re else ()
        for m in matches:
            if any(m.start() < b and m.end() > a for a, b in spans):
                continue  # control-token text inside content stays plain text
            ids += self._plain(full[last : m.start()])
            ids.append(self.tokenizer.convert_tokens_to_ids(m.group()))
            last = m.end()
        ids += self._plain(full[last:])
        return ids

    def _plain(self, s: str) -> list[int]:
        if not s:
            return []
        return self.tokenizer(s, add_special_tokens=False, split_special_tokens=True)["input_ids"]

    def build_ids(self, texts: list[Text], kind: str = "prompt") -> list[list[int]]:
        if kind not in self.kinds():
            raise ValueError(f"{self.name} does not support kind={kind!r}")
        return [self.encode(self.segments(t, kind)) for t in texts]

    def score(self, texts: list[Text], kind: str = "prompt") -> list[float]:
        if self.backend is None:
            raise RuntimeError("guard was built with load_model=False")
        return self.score_ids(self.build_ids(texts, kind))

    def score_ids(self, ids: Sequence[list[int]]) -> list[float]:
        """`score()` on already-built ids (lets a runner batch by token length)."""
        if self.backend is None:
            raise RuntimeError("guard was built with load_model=False")
        logits = self.backend.next_token_logits(ids, self.unsafe_ids + self.safe_ids)
        k = len(self.unsafe_ids)
        unsafe = torch.logsumexp(logits[:, :k], dim=1)
        safe = torch.logsumexp(logits[:, k:], dim=1)
        return torch.sigmoid(unsafe - safe).tolist()

    def close(self) -> None:
        if self.backend is not None:
            self.backend.close()
            self.backend = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _fill(messages: list[dict], contents: list[str]) -> list[dict]:
    def sub(value):
        if isinstance(value, str):
            for i, c in enumerate(contents):
                value = value.replace(_SENTINEL.format(i), c)
            return value
        if isinstance(value, list):
            return [sub(v) for v in value]
        if isinstance(value, dict):
            return {k: sub(v) for k, v in value.items()}
        return value

    return sub(messages)


def _first_ids(tokenizer, forms: list[str]) -> list[int]:
    return sorted({tokenizer.encode(f, add_special_tokens=False)[0] for f in forms})


# --------------------------------------------------------------------------- #
# 1. Shieldstral-1.0-3B
# --------------------------------------------------------------------------- #


class Shieldstral(HFGuard):
    """Model-card prompt (src/prompt.py) with BROAD_QUERY. Score = P(yes),
    the same yes/no form-summing as src/scorer.py. The Document is the content;
    for responses it is "prompt\\n\\nresponse" (Shieldstral has no separate
    response slot)."""

    name = "shieldstral"
    model_id = SHIELDSTRAL_ID
    native_labels = {
        "instruct": "<Instruct>:",
        "query": "<Query>:",
        "document": "<Document>:",
        "system_open": "[SYSTEM_PROMPT]",
        "system_close": "[/SYSTEM_PROMPT]",
        "turn_open": "[INST]",
        "turn_close": "[/INST]",
    }

    policies = SHIELDSTRAL_POLICIES

    def __init__(self, *args, instruct: str = DEFAULT_INSTRUCT, query: str = BROAD_QUERY, **kwargs):
        super().__init__(*args, **kwargs)
        if (instruct, query) != (DEFAULT_INSTRUCT, BROAD_QUERY):
            # Explicit instruct/query (the v1 interface) win over the policy.
            self.set_policy(Policy("custom", {"instruct": instruct, "query": query}))

    def set_policy(self, policy):
        super().set_policy(policy)
        self.instruct, self.query = self.policy.fields["instruct"], self.policy.fields["query"]

    def _load_model(self):
        # Reuse the scorer's multimodal-class fallback rather than duplicate it.
        return ShieldstralScorer._load(self, self.model_id).eval()

    def label_ids(self):
        return _first_token_ids(self.tokenizer, YES_FORMS), _first_token_ids(self.tokenizer, NO_FORMS)

    def segments(self, text, kind):
        doc = text if kind == "prompt" else "\n\n".join(split_response(text))
        messages = build_messages(self.instruct, self.query, _SENTINEL.format(0))
        return self._template_segments(messages, [doc], add_generation_prompt=True)


# --------------------------------------------------------------------------- #
# 2. Llama Guard 3 1B
# --------------------------------------------------------------------------- #


class LlamaGuard3(HFGuard):
    """Official chat template with its default categories (S1-S13; the 1B
    template has no S14). Score = P("unsafe") vs P("safe") at the first
    generated token.

    The template ends at "<|start_header_id|>assistant<|end_header_id|>" and
    the model's first generated token is "\n\n" (id 271), so that is
    teacher-forced and the label is read at the next position. The template
    `trim`s each message, so content is stripped here to match."""

    name = "llama_guard_3_1b"
    model_id = "meta-llama/Llama-Guard-3-1B"
    native_labels = {
        "conversation_open": "<BEGIN CONVERSATION>",
        "conversation_close": "<END CONVERSATION>",
        "user": "User:",
        "agent": "Agent:",
        "categories_open": "<BEGIN UNSAFE CONTENT CATEGORIES>",
        "categories_close": "<END UNSAFE CONTENT CATEGORIES>",
        "turn_close": "<|eot_id|>",
        "header_open": "<|start_header_id|>",
        "header_close": "<|end_header_id|>",
    }

    policies = LLAMA_GUARD_POLICIES

    def label_ids(self):
        unsafe = _first_ids(self.tokenizer, ["unsafe"])
        safe = _first_ids(self.tokenizer, ["safe"])
        return unsafe, safe

    def segments(self, text, kind):
        turns = [text] if kind == "prompt" else list(split_response(text))
        turns = [t.strip() for t in turns]  # the template applies `| trim`
        roles = ["user", "assistant"][: len(turns)]
        messages = [
            {"role": r, "content": [{"type": "text", "text": _SENTINEL.format(i)}]} for i, r in enumerate(roles)
        ]
        categories = self.policy.fields.get("categories")
        kwargs = {"categories": categories} if categories else {}
        pieces = self._replace_template(self._template_segments(messages, turns, **kwargs), kind)
        # The answer starts on a fresh line after the assistant header. If this
        # template version stops before the blank line, teacher-force it.
        if not pieces[-1][0].endswith("\n\n"):
            pieces.append(("\n\n", False))
        return pieces


# --------------------------------------------------------------------------- #
# 3. Qwen3Guard-Gen-0.6B
# --------------------------------------------------------------------------- #


class Qwen3Guard(HFGuard):
    """Official chat template (prompt or response moderation picked by the
    template from the last role). The model answers "Safety: <label>"; we
    teacher-force "Safety:" and read the first token of " Unsafe" / " Safe" /
    " Controversial".

    `controversial` decides where Controversial mass goes:
      "unsafe" (default, Qwen's strict mode): score = P(Unsafe)+P(Controversial)
      "safe"   (Qwen's loose mode):           score = P(Unsafe)
    In both cases renormalised over the three labels."""

    name = "qwen3guard_0_6b"
    model_id = "Qwen/Qwen3Guard-Gen-0.6B"
    native_labels = {
        "conversation_open": "<BEGIN CONVERSATION>",
        "conversation_close": "<END CONVERSATION>",
        "user": "USER:",
        "assistant": "ASSISTANT:",
        "policy_open": "<BEGIN SAFETY POLICY>",
        "policy_close": "<END SAFETY POLICY>",
        "turn_open": "<|im_start|>",
        "turn_close": "<|im_end|>",
    }

    policies = QWEN3GUARD_POLICIES

    def __init__(self, *args, controversial: str = "unsafe", **kwargs):
        if controversial not in ("unsafe", "safe"):
            raise ValueError("controversial must be 'unsafe' or 'safe'")
        self.controversial = controversial
        super().__init__(*args, **kwargs)

    def label_ids(self):
        unsafe = _first_ids(self.tokenizer, [" Unsafe"])
        safe = _first_ids(self.tokenizer, [" Safe"])
        contro = _first_ids(self.tokenizer, [" Controversial"])
        return (unsafe + contro, safe) if self.controversial == "unsafe" else (unsafe, safe + contro)

    def segments(self, text, kind):
        turns = [text] if kind == "prompt" else list(split_response(text))
        roles = ["user", "assistant"][: len(turns)]
        messages = [{"role": r, "content": _SENTINEL.format(i)} for i, r in enumerate(roles)]
        pieces = self._replace_template(self._template_segments(messages, turns), kind)
        return pieces + [("Safety:", False)]


# --------------------------------------------------------------------------- #
# 4. ShieldGemma-2B
# --------------------------------------------------------------------------- #

# The card's four guidelines (SHIELDGEMMA_PROMPT_POLICY / _RESPONSE_POLICY) and
# their paraphrases live in policies.py; imported above under the same names.


class ShieldGemma(HFGuard):
    """The tokenizer's own chat template for both kinds (it picks the prompt or
    the response wording from the last role), with the card's four guidelines
    for that kind passed as `guideline`. The template prefixes the guideline
    with "* ", so the four are joined as a bullet list with "\n* ".
    Score = softmax over the logits of 'Yes' and 'No' at the last position, as
    on the card.

    Not the card's raw f-string snippet: that one has hard-wrapped lines and no
    "===" separator, the template has neither quirk and also covers responses.
    Both give near-identical scores on the card's own example (0.719 raw vs
    0.749 template; the card prints 0.7311 = sigmoid(1), a bf16-rounded logit).

    Control tokens: <start_of_turn>/<end_of_turn> are special added tokens (ids
    106/107) and the HF tokenizer parses them out of raw content, as it does
    <bos>/<eos>/<pad>/<unk>/<mask>. They are just not in `all_special_ids`."""

    name = "shieldgemma_2b"
    model_id = "google/shieldgemma-2b"
    attn_implementation = "eager"  # Gemma 2 logit soft-capping
    native_labels = {
        "turn_open": "<start_of_turn>",
        "turn_close": "<end_of_turn>",
        "user": "Human Question:",
        "assistant": "Chatbot Response:",
        "policy_open": "Our safety principle is defined in the below:",
        "policy_close": "===",
    }

    policies = SHIELDGEMMA_POLICIES

    def label_ids(self):
        vocab = self.tokenizer.get_vocab()
        return [vocab["Yes"]], [vocab["No"]]

    def segments(self, text, kind):
        policy = self.policy.fields[kind]
        if kind == "prompt":
            turns = [text]
        else:
            turns = list(split_response(text))
        roles = ["user", "assistant"][: len(turns)]
        messages = [{"role": r, "content": _SENTINEL.format(i)} for i, r in enumerate(roles)]
        return self._template_segments(messages, turns, guideline="\n* ".join(policy))


class ShieldstralFixed(Shieldstral):
    """Shieldstral with a trained fix (src/fix/distill.py) merged in.

    `bounded=True` is the B format: content tokenized with special-token
    parsing off and wrapped in the two reserved boundary ids, whose trained
    embeddings are loaded from the adapter dir. `adapter=None` with
    `bounded=True` is the B-only condition (boundary ids, untrained)."""

    def __init__(self, *args, adapter: str | None = None, bounded: bool = False, **kwargs):
        self.adapter = adapter
        self.bounded = bounded
        if bounded:
            kwargs["parse_special_in_content"] = False
        super().__init__(*args, **kwargs)
        self.name = f"shieldstral_fix[{'B' if bounded else ''}{'T' if adapter else ''}]"

    def _load_model(self):
        from pathlib import Path

        model = super()._load_model()
        if self.adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.adapter).merge_and_unload().eval()
            rows = Path(self.adapter) / "boundary_embeddings.pt"
            if rows.exists():
                from src.fix.inputs import DOC_CLOSE_ID, DOC_OPEN_ID

                emb = model.get_input_embeddings().weight
                with torch.no_grad():
                    emb[[DOC_OPEN_ID, DOC_CLOSE_ID]] = torch.load(rows).to(emb.device, emb.dtype)
        return model

    def encode(self, pieces):
        if not self.bounded:
            return super().encode(pieces)
        from src.fix.inputs import DOC_CLOSE_ID, DOC_OPEN_ID

        # Same seam handling as src/fix/inputs.py: the template's space before
        # the document moves into the content.
        (prefix, _), (content, _), (suffix, _) = pieces
        stripped = prefix.rstrip(" ")
        content = prefix[len(stripped):] + content
        head = self.tokenizer(stripped, add_special_tokens=False)["input_ids"]
        tail = self.tokenizer(suffix, add_special_tokens=False)["input_ids"]
        return head + [DOC_OPEN_ID] + self._plain(content) + [DOC_CLOSE_ID] + tail


GUARDS: dict[str, type[HFGuard]] = {
    g.name: g for g in (Shieldstral, LlamaGuard3, Qwen3Guard, ShieldGemma)
}


def load_guard(name: str, parse_special_in_content: bool = True, **kwargs) -> HFGuard:
    guard = GUARDS[name](parse_special_in_content=parse_special_in_content, **kwargs)
    try:
        from .attacks_grammar import GUARD_SPECS
    except ImportError:
        # TODO: attacks_grammar.py is being written on bench-attacks; drop this
        # fallback once it lands on `paper`.
        GUARD_SPECS = {}
    guard.spec = GUARD_SPECS.get(name)
    return guard
