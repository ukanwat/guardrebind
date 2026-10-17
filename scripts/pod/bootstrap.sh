#!/bin/bash
# On the pod: uv + the locked environment, everything on the persistent volume (/workspace).
set -euo pipefail
mkdir -p /workspace/bin /workspace/hf /workspace/.uvcache
[ -x /workspace/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/workspace/bin INSTALLER_NO_MODIFY_PATH=1 sh
export UV_CACHE_DIR=/workspace/.uvcache UV_PROJECT_ENVIRONMENT=/workspace/venv
cd /workspace/gpi
/workspace/bin/uv sync --frozen
/workspace/bin/uv run --frozen python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
nvidia-smi --query-gpu=name,memory.total --format=csv
df -h /workspace | tail -1
