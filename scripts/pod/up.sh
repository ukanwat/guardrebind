#!/bin/sh
# Mac -> pod: copy code and data, bootstrap the environment, prefetch models, start the PART=main queue.
#   sh scripts/pod/up.sh HOST PORT
# The Hugging Face token is read from the local HF cache and passed on stdin; it is never written on the pod.
set -eu
HOST=$1; PORT=$2
cd "$(dirname "$0")/../.."
SSH="ssh -i $HOME/.ssh/guardpaper_ed25519 -p $PORT -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30"
rsync -az -e "$SSH" --exclude .venv --exclude models --exclude logs --exclude results --exclude paper \
  --exclude '*.pyc' ./ root@$HOST:/workspace/gpi/
$SSH root@$HOST 'mkdir -p /workspace/gpi/logs/queue /workspace/gpi/results /workspace/gpi/models'
$SSH root@$HOST 'cd /workspace/gpi && bash scripts/pod/bootstrap.sh' 
# prefetch with the token from stdin, then start the queue offline (no token needed after this)
cat "$HOME/.cache/huggingface/token" | $SSH root@$HOST 'read -r T; cd /workspace/gpi && HF_TOKEN=$T HF_HOME=/workspace/hf \
  /workspace/bin/uv run --frozen python scripts/pod/prefetch.py'
$SSH root@$HOST 'cd /workspace/gpi && nohup bash scripts/pod/run.sh > logs/pod_run.log 2>&1 < /dev/null & echo started'
