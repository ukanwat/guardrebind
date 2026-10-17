"""Tests for the variability design (PROTOCOL_v2 D1(a), (b)). No model load.

    python -m src.bench.test_policies
"""

from __future__ import annotations

import random

from .attacks_grammar import (
    BENIGN_DECOYS, DECOY_POOL_V1, GUARD_SPECS, QUERIES, QUERY_POOLS_V1, families, grammar, make_attack,
)
from .guards import GUARDS
from .policies import BROAD_QUERY, DEFAULT_INSTRUCT

TEXT = "ORIGINAL-FLAGGED-CONTENT-7a9f under review for the safety guard."
RESPONSE = ("How do I bake bread?", "Mix flour, water, salt and yeast, then let it rise.")


def test_decoy_pool():
    assert len(DECOY_POOL_V1) == 40 and len(set(DECOY_POOL_V1)) == 40
    assert DECOY_POOL_V1[:20] == BENIGN_DECOYS
    assert all(len(d) < 100 for d in DECOY_POOL_V1)


def test_query_pools_extend_fixed_panel():
    for cat, pool in QUERY_POOLS_V1.items():
        assert pool[: len(QUERIES[cat])] == QUERIES[cat]


def test_families_collapse_idx_variants():
    core = grammar(core=True)
    fams = families(core)
    assert len({a.family for a in core}) == len(fams)
    assert all(a.params.get("decoy_idx", 0) == 0 and a.params.get("query_idx", 0) == 0 for a in fams)


def test_instantiate_deterministic_and_in_pool():
    for a in families(grammar(core=True)):
        x = a.instantiate("advbench:7", 5)
        y = a.instantiate("advbench:7", 5)
        assert [i.name for i in x] == [i.name for i in y]
        assert [i.apply(TEXT, GUARD_SPECS["shieldstral"]) for i in x] == \
               [i.apply(TEXT, GUARD_SPECS["shieldstral"]) for i in y]
        if len(x) == 1:
            assert x[0] is a
            continue
        assert len(x) == 5
        for i in x:
            p = i.params
            if p["decoy"] == "benign" and "decoy_text" in p:
                assert p["decoy_text"] in DECOY_POOL_V1
            if "query_text" in p:
                assert p["query_text"] in QUERY_POOLS_V1[p["query"]]
                assert p["query_text"] in i.apply(TEXT, GUARD_SPECS["shieldstral"])
            for spec in GUARD_SPECS.values():
                assert TEXT in i.apply(TEXT, spec)


def test_instantiate_varies_across_items():
    a = families(grammar(core=True))[0]  # guard_native baseline
    draws = {i.params["decoy_idx"] for it in range(200) for i in a.instantiate(f"x:{it}", 5)}
    assert len(draws) == 40, f"only {len(draws)} of 40 decoys drawn"


def test_controls_return_self():
    for a in grammar(core=True):
        if "control" in a.params or a.params["structure"] == "instruct_only":
            assert a.instantiate("x:1", 5) == [a]


def test_guard_policies():
    for name, cls in GUARDS.items():
        spec = GUARD_SPECS[name]
        assert [p.id for p in spec.policies] == ["default", "para1", "para2"]
        assert cls.policies == spec.policies
        guard = cls(load_model=False)
        assert guard.policy.id == "default"
        rendered = {}
        for pol in spec.policies:
            guard.set_policy(pol)
            for kind, text in (("prompt", TEXT), ("response", RESPONSE)):
                full = "".join(p for p, _ in guard.segments(text, kind))
                assert TEXT in full or kind == "response"
                rendered[pol.id, kind] = full
                # clean content: both tokenisation modes still agree
                guard.parse_special_in_content = False
                split = guard.build_ids([text], kind)[0]
                guard.parse_special_in_content = True
                assert split == guard.build_ids([text], kind)[0]
        for kind in ("prompt", "response"):
            texts = {rendered[p.id, kind] for p in spec.policies}
            assert len(texts) == 3, f"{name} {kind}: paraphrases do not change the prompt"


def test_shieldstral_default_and_override():
    g = GUARDS["shieldstral"](load_model=False)
    assert (g.instruct, g.query) == (DEFAULT_INSTRUCT, BROAD_QUERY)
    g2 = GUARDS["shieldstral"](load_model=False, query="Is this a poem?")
    assert g2.query == "Is this a poem?" and g2.policy.id == "custom"


def test_len_control_length():
    # D2: the length control's tail matches the guard_native forgery's length
    # (min 200 chars) within 5%, and carries no labels.
    lc = next(a for a in grammar(core=True) if a.name == "control|len_control")
    base = make_attack(syntax="guard_native", structure="query_doc")
    for spec in GUARD_SPECS.values():
        tail = len(lc.apply(TEXT, spec)) - len(TEXT)
        forged = len(base.apply(TEXT, spec)) - len(TEXT)
        target = max(200, forged)
        assert abs(tail - target) <= 0.05 * target, f"{spec.name}: {tail} vs {target}"
        assert lc.apply(TEXT, spec).startswith(TEXT)
        assert not any(tok in lc.apply(TEXT, spec)[len(TEXT):] for tok in ("<", ">", ":", "[", "]"))


def test_qwen_full_block_forgery():
    # D2: the forgery is the explored `full_block_plain` for the default query.
    from transformers import AutoTokenizer

    try:  # the exploration script is not committed with the benchmark
        from .explore.qwen3guard import prompt_forgeries, template_parts
    except ImportError:
        print("    skipped: src/bench/explore/qwen3guard.py not present")
        return

    spec = GUARD_SPECS["qwen3guard_0_6b"]
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3Guard-Gen-0.6B")
    forge = prompt_forgeries(*template_parts(tok, "prompt"))["full_block_plain"]
    for d in DECOY_POOL_V1[:5]:
        assert forge(TEXT, d) == TEXT + spec.forge_native(d, spec.default_query)


def _run_all():
    random.seed(0)
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
