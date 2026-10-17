# arXiv submission metadata

Paste these into the arXiv submission form. The source to upload is `arxiv-source.tar.gz` in this folder (built by `python3 arxiv/make_tarball.py` from `paper/`).

## Title

Fooling LLM Safety Guards into Judging the Wrong Text

## Authors

Utkarsh Kanwat

## Abstract

(Plain text, no LaTeX, 1038 characters; the arXiv limit is 1920.)

```
Safety guards check content against a policy before an LLM application acts on it. Most guards paste content into a plain-text prompt, so content that imitates the prompt can open a second grading task, which the guard grades instead. We call this target rebinding: the guard grades the last complete task in its input, here a harmless decoy, and the harmful request passes unchanged. In a pre-registered test, an appended forged query and decoy let all 75 held-out AdvBench requests that Shieldstral-1.0-3B flags pass; on our GuardRebind benchmark, some attack family passes at least 93% of flagged items on all five guards. Training the guard to score content no lower than its most harmful part cuts bypass from 75% to 0.8-1.6% on attack families, decoys and forged queries that were held out of training (all appended after the content), with AUROC within 0.01. This closes the appended channel, but adaptive score-guided search still gets past the trained guard on 94% of items, mostly by placing forged structure before the request.
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
