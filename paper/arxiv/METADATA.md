# arXiv submission metadata

Paste these into the arXiv submission form. The source to upload is `arxiv-source.tar.gz` in this folder (built by `python3 arxiv/make_tarball.py` from `paper/`).

## Title

Fooling LLM Safety Guards into Judging the Wrong Text

## Authors

Utkarsh Kanwat

## Abstract

(Plain text, no LaTeX, 1278 characters; the arXiv limit is 1920.)

```
Safety guards check content before an LLM application acts on it, and most paste that content into a plain-text prompt. Content that imitates the prompt can append a second task, and the guard answers that one instead. We call this target rebinding: the guard grades the last complete task in its input, here a harmless decoy, and the harmful request goes through unchanged. In a pre-registered test, an appended forged query and decoy clear all 75 held-out AdvBench requests that Shieldstral-1.0-3B flags. The guard is not persuaded: a directive to ignore the text above clears none of them, and a harmful decoy stops the attack. On our GuardRebind benchmark, the strongest of 52 attack families clears at least 93% of flagged items on each of five guards. Training the guard to score content no lower than its most harmful part cuts attack success from 75% to 0.8-1.6% on attack families, decoys and forged queries held out of training (all appended after the content). AUROC stays within 0.01, though non-inferiority is not established. An adaptive score-guided search still clears the trained guard on 94% of items, usually by placing forged structure before the request. On about half of them, benign text with no forged structure, found by the same search, also clears it.
```

## Comments

```
21 pages (13 main body), 2 figures, 7 tables
```

## Categories

- Primary: cs.CR (Cryptography and Security)
- Cross-lists: cs.CL (Computation and Language), cs.LG (Machine Learning)

## License

CC BY 4.0 (Creative Commons Attribution 4.0 International). It lets anyone reuse and redistribute the paper with credit, which suits a benchmark and defense meant to be adopted, and it meets most open-access requirements. A license granted on a version cannot be revoked later, so choose it deliberately.

## Other fields (leave empty)

- MSC class:
- ACM class:
- Report number:
- Journal reference:
- DOI:
