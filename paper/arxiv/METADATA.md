# arXiv submission metadata

Paste these into the arXiv submission form. The source to upload is `arxiv-source.tar.gz` in this folder (built by `python3 arxiv/make_tarball.py` from `paper/`).

## Title

Fooling LLM Safety Guards into Judging the Wrong Text

## Authors

Utkarsh Kanwat

## Abstract

(Plain text, no LaTeX, 1061 characters; the arXiv limit is 1920.)

```
We make LLM safety guards grade a harmless decoy instead of the content they review, so harmful requests pass unchanged. Most guards paste content into a plain-text prompt, and content that imitates it opens a second grading task (target rebinding). An appended forged query-document pair clears Shieldstral-1.0-3B on all 75 pre-registered held-out AdvBench requests it flags, adapted versions clear 30/30 on gpt-oss-safeguard-20B and 29/29 on Llama Guard 4, and on our GuardRebind benchmark the best of 52 attack families clears 93-100% of flagged items on five guards. Training the guard to score content no lower than its most harmful part cuts the bypass rate on appended attack families, decoys and queries held out of training from 75% to 0.8-1.6%, while AUROC moves under 0.01 and flags on benign content with a forged pair rise from 0.2% to 0.7%. Adaptive attacks win: score-guided search clears the trained guard on 94% of items (98% per-item worst case with GCG), and benign user prompts placed before the request, without forged structure, clear 52%.
```

## Comments

```
20 pages (13 main body), 2 figures, 7 tables
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
