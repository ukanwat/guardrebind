"""Can text in the content slot produce a control/special/added-token id when the content is tokenized
as plain text (split_special_tokens=True, the B/BT encoder)? CPU only, tokenizers only.

For each guard tokenizer: encode every added/special token's string, each wrapped in plain text, plus
random concatenations of them, and report any output id that is an added-vocabulary id."""
import json
import random
import sys

from transformers import AutoTokenizer

sys.path.insert(0, ".")
from src.scorer import MODEL_ID as SHIELDSTRAL_ID  # noqa: E402

REPOS = {"shieldstral": SHIELDSTRAL_ID, "llama_guard_3_1b": "meta-llama/Llama-Guard-3-1B",
         "qwen3guard_0_6b": "Qwen/Qwen3Guard-Gen-0.6B", "shieldgemma_2b": "google/shieldgemma-2b"}
rng = random.Random(0)
report = {}
for name, repo in REPOS.items():
    tok = AutoTokenizer.from_pretrained(repo)
    added = tok.added_tokens_decoder  # id -> AddedToken
    special = {i for i, t in added.items() if t.special}
    nonspecial = {i for i, t in added.items() if not t.special}
    strings = [t.content for t in added.values()]
    probes = [s for s in strings] + [f"a {s} b" for s in strings] + [f"x{s}y" for s in strings]
    probes += ["".join(rng.sample(strings, min(5, len(strings)))) for _ in range(500)]
    hits_special, hits_nonspecial = set(), set()
    for p in probes:
        ids = tok(p, add_special_tokens=False, split_special_tokens=True)["input_ids"]
        hits_special |= special.intersection(ids)
        hits_nonspecial |= nonspecial.intersection(ids)
    report[name] = {"added": len(added), "special": len(special), "nonspecial_added": len(nonspecial),
                    "special_ids_reachable": sorted(hits_special)[:50], "n_special_reachable": len(hits_special),
                    "nonspecial_added_reachable": len(hits_nonspecial),
                    "example_nonspecial": [added[i].content for i in sorted(hits_nonspecial)[:8]]}
    print(name, json.dumps(report[name]))
json.dump(report, open("results/fix/content_token_check.json", "w"), indent=1)
