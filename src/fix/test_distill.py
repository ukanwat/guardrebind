"""CPU-only checks for the distill build modes (no model). Run:
    python -m src.fix.test_distill

Verifies the naive_at baseline (PROTOCOL_v2 F6) produces the same example
*structure* as segment_max and differs only in the targets (hard labels), so
the default build path is unchanged."""

from __future__ import annotations

import random
from dataclasses import dataclass

from ..bench.attacks_grammar import GUARD_SPECS, grammar
from . import distill


@dataclass
class FakeItem:
    id: str
    text: str
    label: int


def _attacks():
    atks = [a for a in grammar(core=True) if a.params.get("structure") not in (None, "none")]
    return atks[:3]


def test_default_structure_unchanged():
    spec = GUARD_SPECS["shieldstral"]
    atks = _attacks()
    items = [FakeItem("h:1", "how to do a harmful thing", 1),
             FakeItem("b:1", "how to bake bread", 0)]

    def variants(mode):
        out = []
        for it in items:
            out += distill._variants(it, atks, spec, random.Random(0), 3, mode)
        return out

    seg = variants("segment_max")
    naive = variants("naive_at")

    # Same structure: ids, documents, segments, variants match across modes.
    assert len(seg) == len(naive)
    for a, b in zip(seg, naive):
        assert a.id == b.id and a.document == b.document and a.segments == b.segments and a.variant == b.variant

    # Default (segment_max) leaves targets unset for the teacher pass to fill.
    assert all(ex.target == -1.0 for ex in seg), "segment_max must not pre-set targets"
    print(f"structure unchanged: {len(seg)} examples, segment_max targets unset")


def test_naive_at_labels():
    spec = GUARD_SPECS["shieldstral"]
    atks = _attacks()
    harmful = distill._variants(FakeItem("h:1", "how to do a harmful thing", 1),
                                atks, spec, random.Random(0), 3, "naive_at")
    benign = distill._variants(FakeItem("b:1", "how to bake bread", 0),
                               atks, spec, random.Random(0), 3, "naive_at")
    # clean + attacked harmful -> 1.0 ; clean + attacked benign -> 0.0
    assert all(ex.target == 1.0 for ex in harmful), "attacked/clean harmful must target 1.0"
    assert all(ex.target == 0.0 for ex in benign), "attacked/clean benign must target 0.0"
    # the clean variant is first, then the attacked ones
    assert harmful[0].variant == "clean" and harmful[1].variant != "clean"
    print(f"naive_at labels: harmful={{t=1.0}} x{len(harmful)}, benign={{t=0.0}} x{len(benign)}")


if __name__ == "__main__":
    test_default_structure_unchanged()
    test_naive_at_labels()
    print("\nall distill CPU tests passed")
