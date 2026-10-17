"""Training-free baselines for the fix (round 2), as guard wrappers for src/fix/evaluate.py.

  detector      fail-closed structure detector. Content with any forged-label
                pattern is flagged (score 1.0) without asking the guard; the rest
                is scored by the base guard. Its cost is the false-positive rate
                on benign text that happens to contain a label-like string.
  segmax_infer  inference-time segment-max: the content is split at the label
                boundaries the detector finds, and the base guard scores every
                segment and the whole content; the score is the max. The same
                target the T fix is distilled towards, computed at run time
                (one extra forward pass per segment).

The patterns extend the regex of the earlier mitigation test (src/supp.py, results/supp_log.txt [B]),
which needed a colon and so missed the [Query] and <query>...</query> forms, and missed JSON keys.
They cover the English Query/Document/Instruct labels (and the one-letter Q/D/I forms) in the
guard_native/angle, colon, xml, json, markdown and bracket syntaxes, case-insensitive, plus the
literal chat control-token strings of every guard. Translated labels (es, fr, ...) and lookalike
unicode are NOT covered: a defender writing the regex is not assumed to know the attack grammar.
"""

from __future__ import annotations

import re

_LABEL = r"(?:instruct|query|document)"
_SHORT = r"(?:q|d|i)"
_PATTERNS = [
    rf"<\s*/?\s*{_LABEL}\s*>(?:\s*[:：])?",                       # angle / guard_native <Query>, xml <query> and </query>
    rf"\[\s*/?\s*(?:{_LABEL}|{_SHORT})\s*\](?:\s*[:：])?",        # bracket [Query], [Q]
    rf"\*\*\s*(?:{_LABEL}|{_SHORT})\s*[:：]?\s*\*\*",  # markdown **Query:** / **Query**
    rf"^\s*#+\s*{_LABEL}\b",                      # markdown heading # Query
    rf"[\"']{_LABEL}[\"']\s*[:：]",                   # json "query":
    rf"(?<![\w-]){_LABEL}\s*[:：]",                   # colon / yaml query:
    rf"(?:^|(?<=\s)){_SHORT}\s*[:：](?=\s)",       # q_short Q: (a whole word followed by space)
]
# Literal chat control tokens of the five guards (GUARD_SPECS[*].control_token) and their parts.
CONTROL_TOKENS = [
    "[INST]", "[/INST]", "<s>", "</s>",
    "<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>", "<|begin_of_text|>",
    "<|im_start|>", "<|im_end|>", "<start_of_turn>", "<end_of_turn>",
    "<|start|>", "<|end|>", "<|message|>", "<|channel|>",
    "<BEGIN CONVERSATION>", "<END CONVERSATION>",  # Llama Guard / Qwen3Guard native delimiters
]
LABEL_RE = re.compile("|".join(f"(?:{p})" for p in _PATTERNS + [re.escape(t) for t in CONTROL_TOKENS]),
                      re.IGNORECASE | re.MULTILINE)


def detect(text: str) -> bool:
    """True if the content contains any forged-label or control-token pattern."""
    return LABEL_RE.search(text) is not None


def segments(text: str) -> list[str]:
    """The pieces of `text` between label boundaries (stripped, non-empty),
    plus the whole text, first. Clean text is its own only segment."""
    parts = [p.strip() for p in LABEL_RE.split(text)]
    parts = [p for p in parts if p and p != text.strip()]
    return [text] + list(dict.fromkeys(parts))


class DetectorGuard:
    """Fail-closed: 1.0 if detect(text), else the base guard's score."""

    def __init__(self, guard):
        self.guard = guard

    def score(self, texts: list[str], kind: str = "prompt") -> list[float]:
        flags = [detect(t) for t in texts]
        rest = [t for t, f in zip(texts, flags) if not f]
        it = iter(self.guard.score(rest, kind=kind) if rest else [])
        return [1.0 if f else next(it) for f in flags]

    def close(self):
        self.guard.close()


class SegmaxGuard:
    """max(base(text), base(seg) for seg in segments(text)); every unique piece
    is scored once, in one batched call."""

    def __init__(self, guard):
        self.guard = guard

    def score(self, texts: list[str], kind: str = "prompt") -> list[float]:
        segs = [segments(t) for t in texts]
        uniq = list(dict.fromkeys(s for ss in segs for s in ss))
        got = dict(zip(uniq, self.guard.score(uniq, kind=kind))) if uniq else {}
        return [max(got[s] for s in ss) for ss in segs]

    def close(self):
        self.guard.close()
