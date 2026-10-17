#!/bin/bash
# On the pod: uv + the locked environment, everything on the persistent volume (/workspace).
set -euo pipefail
mkdir -p /workspace/bin /workspace/hf /workspace/.uvcache
[ -x /workspace/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/workspace/bin INSTALLER_NO_MODIFY_PATH=1 sh
export UV_CACHE_DIR=/workspace/.uvcache UV_PROJECT_ENVIRONMENT=/workspace/venv
cd /workspace/gpi
/workspace/bin/uv sync --frozen
# The locked torch wheel targets CUDA 13; hosts with a CUDA 12.x driver need the cu129 build of the same version.
TV=$(/workspace/venv/bin/python -c "import torch; print(torch.__version__.split('+')[0])")
/workspace/venv/bin/python -c "import torch; torch.zeros(1).cuda()" 2>/dev/null || \
  /workspace/bin/uv pip install --python /workspace/venv/bin/python --reinstall-package torch "torch==$TV+cu129" \
    --index-url https://download.pytorch.org/whl/cu129 --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
/workspace/venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
nvidia-smi --query-gpu=name,memory.total --format=csv
df -h /workspace | tail -1
