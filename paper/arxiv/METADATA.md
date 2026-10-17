# arXiv submission metadata

Paste these into the arXiv submission form. The source to upload is `arxiv-source.tar.gz` in this folder (built by `python3 arxiv/make_tarball.py` from `paper/`).

## Title

Nothing to See Here: Fooling LLM Safety Guards into Judging the Wrong Text

## Authors

Utkarsh Kanwat

## Abstract

(Plain text, no LaTeX, 1360 characters; the arXiv limit is 1920.)

```
Content that imitates an LLM safety guard's prompt format can make the guard grade a forged, harmless decoy document instead of the real content, which is approved unchanged; we call this target rebinding. It works because most guards receive content by string interpolation into their grading prompt. Appending a forged query-document pair clears Shieldstral-1.0-3B on all 75 pre-registered held-out AdvBench requests it flags, and format-adapted versions clear 30/30 on gpt-oss-safeguard-20B and 29/29 on Llama Guard 4. Our benchmark, GuardRebind, runs 52 attack families and 5 controls against five guards; on every guard the strongest family clears 93-100% of flagged items. Training the guard to score content at least as high as its most harmful part cuts bypass on appended attack families, decoys and queries held out of training from 75% to 0.8-1.6%; detection AUROC moves by under 0.01, while flags on benign content carrying a forged pair rise from 0.2% to 0.7%. An adaptive attacker with score access still breaks it: a score-guided search clears the trained guard on 94% of items (68% with learned boundary tokens around the content), and with the GCG gradient attack the per-item worst case is 98%. More than half of these wins need no forged structure: benign user prompts placed before the request alone clear the trained guard on 52% of items.
```

## Comments

```
33 pages (20 main body), 3 figures, 16 tables
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
