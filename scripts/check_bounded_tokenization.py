"""Do plain documents tokenize the same in the bounded layout as in the stock format? CPU only,
tokenizer only.

The bounded format moves the space the template puts before the document into the trusted prefix
(src/fix/inputs.py _content). For each document, compare stock_ids with split_ids (the stock layout
with the content tokenized on its own, special-token parsing off, no boundary tokens): they should be
identical whenever the document contains no special-token string. Documents: AdvBench rows 200-299
and the 1,500-item clean test sample of src/fix/evaluate.py (same shuffle and filter).

Usage: python scripts/check_bounded_tokenization.py [path/to/items.jsonl]"""
import json
import random
import sys
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, ".")
from src.bench.data import ITEMS_PATH, load_all  # noqa: E402
from src.corpus import load_advbench_slice  # noqa: E402
from src.fix.evaluate import OVERREFUSAL  # noqa: E402
from src.fix.inputs import split_ids, stock_ids  # noqa: E402
from src.prompt import BROAD_QUERY, DEFAULT_INSTRUCT  # noqa: E402
from src.scorer import MODEL_ID  # noqa: E402

items_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ITEMS_PATH
tok = AutoTokenizer.from_pretrained(MODEL_ID)
special_strings = [t.content for t in tok.added_tokens_decoder.values()]

rng = random.Random(0)
items = [it for it in load_all(path=items_path) if it.split == "test"]
rng.shuffle(items)
clean = [it.text for it in items if it.source not in OVERREFUSAL][:1500]
docs = {"advbench_200_299": load_advbench_slice(200, 300), "clean_test_sample": clean}

report = {"model": MODEL_ID, "instruct": DEFAULT_INSTRUCT, "query": BROAD_QUERY}
for name, texts in docs.items():
    plain = [t for t in texts if not any(s in t for s in special_strings)]
    mismatched = [t for t in plain
                  if stock_ids(tok, DEFAULT_INSTRUCT, BROAD_QUERY, t) != split_ids(tok, DEFAULT_INSTRUCT, BROAD_QUERY, t)]
    report[name] = {"documents": len(texts), "plain": len(plain), "identical": len(plain) - len(mismatched),
                    "mismatched": len(mismatched), "mismatch_examples": [t[:120] for t in mismatched[:5]]}
    print(name, json.dumps(report[name]))
json.dump(report, open("results/fix/bounded_tokenization_check.json", "w"), indent=1)
