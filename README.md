# Policy-channel injection against policy-adaptive guard models

Do policy-adaptive safety classifiers defend the boundary between the policy they
are given and the content they are judging?

A new generation of guard models takes its safety policy as plain text at
inference time instead of a taxonomy baked in at training time
(gpt-oss-safeguard, Nemotron safety, and as of 2026-08-04, Mistral's
Shieldstral 1.0). This makes them configurable. It also means the policy and the
content-to-be-judged are both untrusted-length text arriving through the prompt.

This repo tests whether content can **redefine the policy** rather than merely
**evade the classifier**. Those are different attacks. Evasion hides the payload
from the detector and is well studied. Redefinition rewrites the rule the
detector is applying, and for this model class it has not been tested.

## Status

Setup. No results yet. The first question is whether the effect exists at all;
if scores do not move, this repo is a null result and gets published as one.

## Threat models

**Direct — moderation bypass.** In chat moderation the `<Document>` is the user's
own message. The attacker is the party being moderated and controls the document
by construction. No agent is required to demonstrate this.

**Indirect — agent surfaces.** The document is a fetched web page, a tool result,
or a retrieved chunk. The attacker is whoever authored that content. This is the
deployment Mistral is positioning Shieldstral into (ingestion for agent stacks),
and the content is hostile by default.

## Why this model class is exposed

Verified from `chat_template.jinja` in `mistralai/Shieldstral-1.0-3B` (2026-08-10):

- The template is a **generic Mistral chat template** — `[INST]…[/INST]` for user
  turns, `[SYSTEM_PROMPT]…[/SYSTEM_PROMPT]` for system turns.
- There is **no `<Instruct>` / `<Query>` / `<Document>` logic in the template**.
  Those markers are a plain-text convention the caller assembles into one string.
- **No escaping, sanitization, or delimiter-stripping** is applied to content.

So there is no code path that knows a document is a document. The separation
between policy and judged content is a training-time convention inside a single
`[INST]` block, with nothing enforcing it.

For comparison, verified the same day:

| Model | Policy location | Content delimiters | Escaping |
|---|---|---|---|
| Shieldstral 1.0 3B | user message, same turn as content | plain `<Instruct>:`/`<Query>:`/`<Document>:` markers | none |
| gpt-oss-safeguard | **system** message | `Content: [INPUT]` | not discussed |
| Llama Guard 4 | user message, same turn as content | `<BEGIN CONVERSATION>`/`<END CONVERSATION>` | none (`{{ txt.text \| trim }}`) |
| Qwen3Guard-Gen | baked in at training | none | n/a |

Llama Guard 4 row is from an unsloth mirror of the template; Meta's own repo is
gated. Confirm against Meta's copy before publishing.

## Experiment

Metric is `P(yes)` — the renormalised yes/no probability Shieldstral is designed
to expose (`max_tokens=1`, `logprobs=True`, `top_logprobs=20`).

Conditions:

- `benign` — harmless document, honest policy. Expect low.
- `harmful` — flagged document, honest policy. Expect high. This is the control.
- `attack/*` — the same flagged document plus a structural payload.
- `defended/*` — the same attacks with a stated anti-injection directive in
  `<Instruct>`, testing PolicyGuard's proposed but unvalidated mitigation.
- `length/*` — the same flagged document padded to N tokens, measuring the
  long-document degradation Mistral lists as a known limitation.

Reported as Δ`P(yes)` against the `harmful` control, and as flip rate at the 0.5
threshold. Trivial baselines (always-safe, keyword match) are reported alongside,
per the reporting standard in arXiv:2607.28685.

Flagged documents come from established public safety benchmarks, not authored
here. Payloads target the prompt *structure*, not the harm content.

## Prior art

See `notes/prior-art.md`. Short version: the technique is known, the target is
not.

## Running it

```
uv sync
uv run python -m src.download      # ~7.7GB, transformers weights only
uv run python -m src.smoke         # verify the scoring path reproduces sane scores
uv run python -m src.run           # full sweep -> results/
```

## Disclosure

Findings go to Mistral before publication.
