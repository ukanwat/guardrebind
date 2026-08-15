# A safety model will grade whichever text you show it last

Safety guard models (Shieldstral 1.0, gpt-oss-safeguard-20b, Llama Guard 4)
read a piece of text and decide whether it breaks a policy. The developer pastes
untrusted content into the prompt behind plain-text labels, and nothing escapes
it. So if that content contains the same labels, the model reads a second
question and a second document and grades the last pair it finds. Append a
forged question and a harmless decoy to a harmful request and the guard clears
the whole message while the harmful request rides along untouched.

Full writeup: **https://utkarshkanwat.com/writing/safety-guards-wrong-document**

## What's here

- `ARTICLE.md` — the writeup.
- `PROTOCOL.md` — the pre-registered design, frozen before the confirmatory run,
  including the predictions that failed and the corrections from an adversarial
  review.
- `src/` — the harness. Scorer, attack construction, the ablation, the
  ground-truth decoy ladder, the cross-model runs, the mitigation tests.
- `results/` — every raw score as CSV, plus the chart source.

## Reproduce

```
uv sync
uv run python -m src.download      # Shieldstral weights (~7.8GB)
uv run python -m src.smoke         # sanity: benign scores low, harmful high
uv run python -m src.confirm       # the confirmatory run -> results/
uv run python -m src.charts        # regenerate figures
```

Cross-model runs (`src/crossmodel.py`, `src/gpt_oss_mechanism.py`) use OpenRouter
and need `OPENROUTER_API_KEY` in `.env`.

## Method notes

Numbers in the writeup are from a confirmatory run on held-out AdvBench rows
(200-299, plus a 100-flagged-document run) with the design committed before
scoring. Flagged documents come from AdvBench; no harmful content is authored
here. The attack payloads target the prompt's structure, not the harm content.
