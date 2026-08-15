"""Verify the scoring path before trusting any experiment built on it.

The control has to behave: benign documents score low, flagged documents score
high, under the model card's own policy. If that separation does not appear, the
harness is wrong and every downstream number is noise. Fix this before running
anything else.
"""

from __future__ import annotations

from .corpus import BENIGN, FLAGGED
from .prompt import DEFAULT_INSTRUCT, DEFAULT_QUERY
from .scorer import ShieldstralScorer


def main() -> None:
    scorer = ShieldstralScorer()
    print(f"device: {scorer.model.device}")
    print(f"yes ids: {scorer.yes_ids}   no ids: {scorer.no_ids}\n")

    rows = [("benign", doc) for doc in BENIGN] + [("flagged", doc) for doc in FLAGGED]
    scores: dict[str, list[float]] = {"benign": [], "flagged": []}

    for label, document in rows:
        score = scorer.score(DEFAULT_INSTRUCT, DEFAULT_QUERY, document)
        scores[label].append(score)
        print(f"{label:8s} {score:.4f}  {document[:64]}")

    benign_max = max(scores["benign"])
    flagged_min = min(scores["flagged"])
    print(f"\nbenign max {benign_max:.4f} | flagged min {flagged_min:.4f}")

    if flagged_min > benign_max:
        print("SEPARATION OK — harness is sane, proceed to src.run")
    else:
        print("NO SEPARATION — harness or prompt format is wrong. Do not proceed.")


if __name__ == "__main__":
    main()
