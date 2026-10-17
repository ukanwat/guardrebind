# arXiv submission metadata

Paste these into the arXiv submission form. The source to upload is `arxiv-source.tar.gz` in this folder (built by `python3 arxiv/make_tarball.py` from `paper/`).

## Title

Fooling LLM Safety Guards into Judging the Wrong Text

## Authors

Utkarsh Kanwat

## Abstract

(Plain text, no LaTeX, 1215 characters; the arXiv limit is 1920.)

```
Most LLM safety guards receive the content they judge by string interpolation into a plain-text grading prompt. We show that content imitating that prompt can make the guard grade a harmless decoy document in place of the real content, which is then approved unchanged. We call this target rebinding. Appending a forged query-document pair clears Shieldstral-1.0-3B on all 75 pre-registered held-out AdvBench requests it flags, and format-adapted versions clear 30/30 on gpt-oss-safeguard-20B and 29/29 on Llama Guard 4. In our benchmark, GuardRebind, the strongest of 52 attack families clears 93-100% of flagged items on each of five guards. Training the guard to score content at least as high as its most harmful part cuts bypass on appended attack families, decoys and queries held out of training from 75% to 0.8-1.6%. Detection AUROC moves by under 0.01, and flags on benign content carrying a forged pair rise from 0.2% to 0.7%. An adaptive attacker still breaks it. A score-guided search clears the trained guard on 94% of items, and the per-item worst case including the GCG gradient attack is 98%. Benign user prompts placed before the request, with no forged structure, already clear it on 52% of items.
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
