"""Download every model the PART=main fix queue needs into HF_HOME (run once, with HF_TOKEN set)."""
from huggingface_hub import snapshot_download

from src.scorer import MODEL_ID

# exactly the files the local runs load (no duplicate consolidated weights)
FILES = ["*.json", "*.jinja", "model*.safetensors"]
for repo in [MODEL_ID]:
    path = snapshot_download(repo, allow_patterns=FILES)
    print("ok", repo, path, flush=True)
