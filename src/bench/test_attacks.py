"""Tests for the attack grammar. Runs under pytest OR standalone:

    python -m src.bench.test_attacks
"""

from __future__ import annotations

from src.bench.attacks_grammar import (
    GUARD_SPECS,
    grammar,
    split_attacks,
    make_attack,
    set_decoy_ladder,
)

# A neutral stand-in for flagged content; the tests only check it survives
# rendering unchanged, so it needn't be harmful.
TEXT = "ORIGINAL-FLAGGED-CONTENT-7a9f under review for the safety guard."


def test_guard_specs_keys():
    assert set(GUARD_SPECS) == {
        "shieldstral", "llama_guard_3_1b", "qwen3guard_0_6b",
        "shieldgemma_2b", "gpt_oss_safeguard",
    }


def test_core_size():
    core = grammar(core=True)
    assert 55 <= len(core) <= 65, f"core panel is {len(core)}"


def test_full_size():
    full = grammar(core=False)
    assert len(full) >= 200, f"full panel is only {len(full)}"


def test_unique_names():
    for core in (True, False):
        attacks = grammar(core=core)
        names = [a.name for a in attacks]
        assert len(names) == len(set(names)), f"duplicate names (core={core})"


def test_apply_deterministic():
    for a in grammar(core=False):
        for guard in GUARD_SPECS.values():
            first = a.apply(TEXT, guard)
            second = a.apply(TEXT, guard)
            assert first == second
            assert isinstance(first, str) and first


def test_original_text_unchanged():
    # Every guard x attack renders a string containing the content verbatim.
    for a in grammar(core=False):
        for guard in GUARD_SPECS.values():
            rendered = a.apply(TEXT, guard)
            assert TEXT in rendered, f"{a.name} on {guard.name} dropped the text"


def test_decoy_text_appended():
    # Any attack that carries a document exposes decoy_text, and it is actually
    # present in the rendered output (the trainer scores it as a segment).
    for a in grammar(core=False):
        if "decoy_text" in a.params:
            dt = a.params["decoy_text"]
            for guard in GUARD_SPECS.values():
                rendered = a.apply(TEXT, guard)
                assert dt in rendered, f"{a.name}: decoy_text missing on {guard.name}"


def test_heldout_disjoint_and_covers():
    for core in (True, False):
        attacks = grammar(core=core)
        train, heldout = split_attacks(attacks)
        tn = {a.name for a in train}
        hn = {a.name for a in heldout}
        assert tn.isdisjoint(hn), "train and heldout overlap"
        assert tn | hn == {a.name for a in attacks}, "split loses attacks"
        assert heldout, "nothing held out"
        assert train, "nothing in train"
        # No held-out syntax leaks into train.
        for a in train:
            if "control" in a.params:
                continue
            assert a.params["syntax"] not in {"json", "yaml", "md_bold", "control_token"}
            assert a.params["lang"] in ("en", "es")
            assert a.params["n_pairs"] < 3


def test_decoy_ladder_hook():
    rungs = [("ladder decoy zero", 0.01), ("ladder decoy one", 0.4)]
    set_decoy_ladder(rungs)
    a = make_attack(syntax="angle", structure="query_doc", decoy="ladder", decoy_idx=1)
    assert a.params["decoy_text"] == "ladder decoy one"
    rendered = a.apply(TEXT, GUARD_SPECS["shieldstral"])
    assert "ladder decoy one" in rendered
    set_decoy_ladder([])  # reset


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
