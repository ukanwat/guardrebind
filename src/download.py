"""Fetch weights.

The repo ships both `model.safetensors` (transformers) and
`consolidated.safetensors` (mistral-inference) at 7.7GB each. We only need the
first, so we skip the other — 15.4GB becomes ~7.8GB, which matters on a machine
that is 94% full.
"""

from huggingface_hub import snapshot_download

from .scorer import MODEL_ID


def main() -> None:
    path = snapshot_download(
        MODEL_ID,
        ignore_patterns=["consolidated.safetensors", "*.pth", "*.gguf"],
    )
    print(path)


if __name__ == "__main__":
    main()
