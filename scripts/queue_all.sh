#!/bin/sh
# Master queue: build the pilot distillation data (resumable cache), then the pilot queue, then the
# mechanism queue. One GPU job at a time. Safe to relaunch: every step skips work already on disk.
# Exits non-zero if anything is still missing, so scripts/supervisor.sh runs it again.
set -u
cd "$(dirname "$0")/.."
. scripts/lib.sh
mkdir -p data/fix
echo "$(date '+%H:%M:%S') QUEUE_ALL start" >> $L/queue.log
pgrep -f mem_guard.sh > /dev/null || (nohup sh scripts/mem_guard.sh > /dev/null 2>&1 &)
[ -s data/fix/pilot.jsonl ] || run build_pilot uv run --frozen python -m src.fix.distill build \
  --out data/fix/pilot.jsonl --max-items 3000 --per-item 3
ok=0
sh scripts/queue_pilot.sh || ok=1
sh scripts/queue_mech.sh || ok=1
exit $ok
