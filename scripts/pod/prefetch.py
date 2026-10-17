"""Download every model the PART=main fix queue needs into HF_HOME (run once, with HF_TOKEN set)."""
from huggingface_hub import snapshot_download

from src.scorer import MODEL_ID

# exactly the files the local runs load (no duplicate consolidated weights)
FILES = ["*.json", "*.jinja", "model*.safetensors", "*.txt", "tokenizer*"]
for repo in [MODEL_ID, "meta-llama/Llama-Guard-3-1B", "Qwen/Qwen3Guard-Gen-0.6B"]:
    path = snapshot_download(repo, allow_patterns=FILES)
    print("ok", repo, path, flush=True)
