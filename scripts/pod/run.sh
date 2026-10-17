#!/bin/bash
# On the pod: run the PART=main fix queue to completion (retrying passes), then wait for the Mac to confirm it has
# copied everything back (logs/queue/SYNCED), then stop this pod so billing ends. Hard stop at MAX_HOURS.
set -u
cd /workspace/gpi
export PATH=/workspace/bin:$PATH UV_CACHE_DIR=/workspace/.uvcache UV_PROJECT_ENVIRONMENT=/workspace/venv
# UV_NO_SYNC: keep the CUDA-12.9 torch that bootstrap.sh installed over the locked (CUDA 13) wheel
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 OFF=0 TRIES=3 UV_NO_SYNC=1
PARTS=${PARTS:-main generality seeds}   # in priority order; the budget cap cuts the last ones
MAX_HOURS=${MAX_HOURS:-14}   # $0.89/h (Secure 4090): 14h = $12.5, inside the $15 budget
START=$(cat logs/queue/POD_START 2>/dev/null || date +%s | tee logs/queue/POD_START)
stop_pod() { echo "$(date '+%F %T') stopping pod: $1" >> logs/queue/queue.log; sleep 5
  runpodctl stop pod "$RUNPOD_POD_ID" || runpodctl pod stop "$RUNPOD_POD_ID"; }
budget_left() { [ $(( $(date +%s) - START )) -lt $(( MAX_HOURS * 3600 )) ]; }
( while sleep 300; do budget_left || { pkill -f 'src\.(fix|bench|mech)\.'; stop_pod "budget ($MAX_HOURS h)"; }; done ) &
# if an earlier run.sh's queue is still going (restart of this script), let it finish first
while pgrep -f "sh scripts/queue_fix_full.sh" > /dev/null; do sleep 30; done
for pass in 1 2 3 4 5; do
  echo "$(date '+%F %T') POD pass $pass" >> logs/queue/queue.log
  ok=1
  if [ -n "${QUEUE:-}" ]; then sh scripts/$QUEUE || ok=0
  else for P in $PARTS; do PART=$P sh scripts/queue_fix_full.sh || ok=0; done; fi
  if [ $ok = 1 ]; then touch logs/queue/FIX_DONE; break; fi
  sleep 60
done
[ -f logs/queue/FIX_DONE ] || touch logs/queue/FIX_GAVE_UP
# wait up to 30 min for the Mac's final sync, then stop
for i in $(seq 1 6); do [ -f logs/queue/SYNCED ] && break; sleep 300; done
stop_pod "queue finished (synced=$([ -f logs/queue/SYNCED ] && echo yes || echo no))"
