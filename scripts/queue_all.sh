#!/bin/sh
# Master queue: build the pilot distillation data (resumable cache), then the pilot queue, then the
# mechanism queue. One GPU job at a time. Safe to relaunch: every step skips work already on disk.
set -u
cd "$(dirname "$0")/.."
L=logs/queue
mkdir -p $L data/fix
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.4
echo "$(date '+%H:%M:%S') QUEUE_ALL start" >> $L/queue.log
pgrep -f mem_guard.sh > /dev/null || (nohup sh scripts/mem_guard.sh > /dev/null 2>&1 &)
if [ ! -s data/fix/pilot.jsonl ]; then
  echo "$(date '+%H:%M:%S') START build_pilot" >> $L/queue.log
  nice -n 10 uv run --frozen python -m src.fix.distill build --out data/fix/pilot.jsonl --max-items 3000 --per-item 3 \
    > $L/build_pilot.log 2>&1
  echo "$(date '+%H:%M:%S') END build_pilot exit=$?" >> $L/queue.log
fi
sh scripts/queue_pilot.sh
sh scripts/queue_mech.sh
echo "$(date '+%H:%M:%S') QUEUE_ALL done" >> $L/queue.log
