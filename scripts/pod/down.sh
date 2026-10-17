#!/bin/sh
# Pod -> Mac: copy results, trained models and logs back (run every ~15 min from the Mac). When the pod queue is
# finished and everything is copied, tell the pod it may stop itself.
#   sh scripts/pod/down.sh HOST PORT
set -u
HOST=$1; PORT=$2
cd "$(dirname "$0")/../.."
SSH="ssh -i $HOME/.ssh/guardpaper_ed25519 -p $PORT -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20"
mkdir -p logs/pod
rsync -az -e "$SSH" root@$HOST:/workspace/gpi/results/ results/ || exit 1
rsync -az -e "$SSH" --exclude ckpt.pt --exclude ckpt.tmp root@$HOST:/workspace/gpi/models/ models/ || exit 1
rsync -az -e "$SSH" root@$HOST:/workspace/gpi/logs/queue/ logs/pod/ || exit 1
rsync -az -e "$SSH" --include '*/' --include '*.jsonl' --exclude '*' root@$HOST:/workspace/gpi/data/fix/ data/fix/ || exit 1
if [ -f logs/pod/FIX_DONE ] || [ -f logs/pod/FIX_GAVE_UP ]; then
  $SSH root@$HOST 'touch /workspace/gpi/logs/queue/SYNCED' && echo SYNCED
fi
tail -2 logs/pod/queue_fix.log 2>/dev/null
