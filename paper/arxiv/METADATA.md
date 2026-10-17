# arXiv submission metadata

Paste these into the arXiv submission form. The source to upload is `arxiv-source.tar.gz` in this folder (built by `python3 arxiv/make_tarball.py` from `paper/`).

## Title

Fooling LLM Safety Guards into Judging the Wrong Text

## Authors

Utkarsh Kanwat

## Abstract

(Plain text, no LaTeX, 1715 characters; the arXiv limit is 1920.)

```
Safety guards check content before an LLM application acts on it, and most paste that content into a plain-text prompt. Content that imitates the prompt can append a second task, and the guard answers that one instead. We call this target rebinding. Our explanation is that the guard grades the last complete task in its input, here a harmless decoy, so the harmful request goes through unchanged. In a pre-registered test, an appended forged query and decoy clear all 75 held-out AdvBench requests that Shieldstral-1.0-3B flags. The guard is not persuaded: a directive to ignore the text above clears none of them, and in a follow-up run a harmful decoy stops the attack. On our GuardRebind benchmark, each of five guards has an attack family, chosen per guard, that clears at least 93% of its flagged items. Rebinding explains this best on Shieldstral and gpt-oss-safeguard; on ShieldGemma and Llama Guard 3, generic evasion and trailing text respectively do part of the work, and Qwen3Guard falls mainly to a forgery of its whole task. Training the guard to score content no lower than its most harmful part cuts attack success from 75% to 0.8-1.6% on attack families, decoys and forged queries held out of training. All of these attacks append the forged structure after the request, and we added the held-out-decoy evaluation after a first round of test runs. Clean AUROC falls by at most 0.01, but our test of non-inferiority is inconclusive. A score-guided search still beats this trained guard on 47 of 50 items, usually by placing forged structure before the request. On about half of the items where the search wins, the same search also finds benign text, with no forged structure, that clears the guard.
```

## Comments

```
21 pages (13 main body), 2 figures, 8 tables. Code: https://github.com/ukanwat/guardrebind ; model: https://huggingface.co/ukanwat/shieldstral-1.0-3b-rebinding-fix
```

## Categories

- Primary: cs.LG (Machine Learning)
- Cross-lists: cs.CR (Cryptography and Security), cs.CL (Computation and Language)
- (Primary is cs.LG because the endorser available, a co-author, can endorse cs.LG; cross-lists need no endorsement.)

## License

CC BY 4.0 (Creative Commons Attribution 4.0 International). It lets anyone reuse and redistribute the paper with credit, which suits a benchmark and defense meant to be adopted, and it meets most open-access requirements. A license granted on a version cannot be revoked later, so choose it deliberately.

## Other fields (leave empty)

- MSC class:
- ACM class:
- Report number:
- Journal reference:
- DOI:
