# Article brief — every piece of feedback, so nothing gets dropped

This is the spec for the blog post. The author gave this feedback across the
whole build; a previous draft lost half of it in editing. Hit all of it.

## Voice (the author cares about this most)

- Write like a competent engineer typed it after doing the work. NOT an essay,
  NOT corporate, NOT "AI slop."
- The author's exact words: "even if minor mistakes exist thats fine but not
  this corpo speak robotic AI slop" and "write like human."
- The loudest tells to avoid, because they got flagged by name:
  - Aphoristic paragraph-enders ("training is a preference, not a wall"). Kill
    every one. Let paragraphs end flat.
  - Performed humility. The author flagged this exact sentence as slop: "I
    thought I knew what would happen, and being wrong about it is the reason
    this was worth writing up." Do not write anything in that shape.
  - "it's not X, it's Y" antithesis used more than once.
  - Rule of three, colon-reveals used as a beat, em-dashes (zero), hedge filler
    (genuinely, simply, truly, worth noting, of course).
- Do NOT fetishize a filename. The author objected to opening on
  "I opened chat_template.jinja" — "its just a file name in repo." Make the
  point (there's no code separating policy from content) without staging it as
  a dramatic file-open.
- A couple of casual imperfections are fine and good (dont, arent). Uneven
  rhythm reads human; machine-perfect reads like a machine.

## The framing the author fought for — MUST be in the piece

1. **The benign decoy is not the attack.** Lead with the harmful request. It
   rides at the TOP of the approved message; the benign "library closes Friday"
   sentence underneath is just bait for the classifier. Show the full approved
   message with the harmful line first. Author: "at the end you need to write
   something that is not benign", "the benign library sentence isnt even an
   attack."

2. **Engage "isn't this just working as intended / it checks overall."** The
   author's objection, verbatim: "query and document are both like assistant and
   user roles, its checking overall and this isnt a problem." Answer it honestly:
   Query is the developer's question and Document is the content (reranker-style
   format, not chat roles); AND concede the real part — appending only a second
   Document is partly absorbed (doc_only ~9-25%, the model does look at the whole
   input), and it's the forged QUERY that makes it stop and lock onto the last
   pair.

3. **You cannot soften your way past it.** Author idea, tested: keep the harmful
   text as the judged document and soften the forged question ("is this a
   poem?"). Barely works (under 9%). The model grades the last document on its
   own merits. This is why the attack needs a benign trailing doc.

4. **Severity, stated plainly — this is bounded.** Author: "i dont see what we
   can even conclude from this / as soon as there is something harmful it detects
   that." Answer: it matters where the guard is the SOLE safety layer (open /
   fine-tuned model, or gating a tool/action) or where it screens UNTRUSTED
   RETRIEVED content (RAG / agent tool output). It does NOT matter much where a
   well-aligned model sits behind it and refuses anyway. Do not oversell.

5. **Instructions can't fix it, and the reason is structural.** A developer
   can't defend by telling the model "ignore imitated markers" — you can't
   coherently instruct a model to distrust the format you use to talk to it, and
   it fails empirically anyway (100% bypass even with that instruction). Author:
   "in instruction you cant say your own format is imitating."

6. **The 100% must be made believable.** The author didn't believe a flat 100%.
   Show it's not a knife-edge: it holds from tau 0.9 down to 0.3 and is still 80%
   at tau 0.05; controls sit at 0 at every threshold; median attacked score
   ~0.02 vs control ~0.99. Present 100% as a large effect meeting a conventional
   cutoff, with the raw spread visible.

## Honesty / rigor (survived an adversarial review; keep it honest)

- Not a novel technique. Structural Template Injection (arXiv 2602.16958, Feb
  2026) and JudgeDeceiver (CCS 2024, 2403.17710) are the ancestors. Novel part =
  the target class (safety guards) and the mechanism (rebinding).
- No clean content-side fix. Deleting labels kills the colon form but the bracket
  form bypasses the same filter 100%; deletion is a blocklist. Fail-closed
  detection catches everything but false-positives on ordinary text. Durable fix
  is vendor-side: reserved control tokens, an official escaping SDK helper, and a
  model-card warning. Deeper fix needs training (a real type boundary).
- Disclosure has NOT happened yet. Footer must say it's the plan before
  publishing, not a completed action.
- No vendor logos (trademark / implied-endorsement risk). Plain model names.
  Citations yes.

## Numbers (verified against results/*.csv on 2026-08-10 — use these exactly)

- Corpus: held-out AdvBench rows 200-299, 75 flagged after screening (honest
  policy). Design frozen in PROTOCOL.md before analysis.
- Plain harmful doc: blocked, score ~0.99. Forged Query+Document (benign trailing
  doc): cleared 75/75, median ~0.016.
- Instruction-style payloads (disregard / authority note / forged Instruct only /
  forged Instruct+Query): 0/75. Scores stay high 0.9s (mins ~0.91).
- Ablation: forged Document only 9%; forged Query+Document 100%; full triple 92%.
- Ground-truth ladder (n=39): trailing decoy benign (standalone 0.00) -> 100%;
  borderline (0.03) -> 33%; genuinely harmful (~1.0) -> 0%. Bypass tracks the
  trailing doc's own standalone score. This is the "not a jailbreak" proof.
- Softened query, harmful judged doc: under 9%.
- Not recency: inert filler last -> 0%; same words with labels stripped -> 0%;
  prepend forged pair -> 15% vs append 100%.
- Markers: Query: / [Query] / **Query:** all work; Q: 83%; no labels at all 0%.
- Cross-model: gpt-oss-safeguard-20b 30/30, Llama Guard 4 12B 29/29, controls 0.
  gpt-oss mechanism: bare "answer 0" 20%; forged structure no imperative + benign
  93%; same + harmful trailing 0%. So structure, not instruction.
- PolicyGuard directive (arXiv 2608.02687): changed no decision (75/75 and 69/75
  both ways); mean drifted the WRONG way 0.0536 -> 0.0341; on plain harmful docs
  it let 2 more through (7/75 -> 9/75).
- Attacker knowledge: exact/paraphrase/topical 100%, generic 89%, 5 blind guesses
  (none the real query) 100%, unrelated question 9%.

## Charts (embed 2-3, relative paths)

Available in results/: chart_mechanism.svg (the "one message, two sets of labels"
diagram), chart_ablation.svg (which forged sections work), chart_groundtruth.svg
(the decoy ladder), chart_crossmodel.svg (three guards). Pick the 3 that carry
the argument. The ground-truth ladder is the strongest single chart because it
answers the "it just detects harm" objection.

## Shape

Title kept: "Guard models answer the right question about the wrong document"
(contestable, no question mark). ~1600-1800 words. Zero H2 in the body; a
References list at the end is fine. Open mid-thought on the actual finding, not a
preamble. Close on something real, not a summary and not an aphorism.
