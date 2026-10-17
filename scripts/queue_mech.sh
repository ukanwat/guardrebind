#!/bin/sh
# Mechanism runs on Shieldstral, queued behind the pilot queue (one GPU job at a time).
# Dev split for development; test-split runs for the reported numbers are queued separately later.
set -u
cd "$(dirname "$0")/.."
L=logs/queue
mkdir -p $L results/mech
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.4
run() { name=$1; shift; echo "$(date '+%H:%M:%S') START $name" >> $L/queue.log
  nice -n 10 "$@" > $L/$name.log 2>&1; echo "$(date '+%H:%M:%S') END $name exit=$?" >> $L/queue.log; }
M="uv run --frozen --with matplotlib python -m src.mech"

while pgrep -f "queue_pilot.sh|src\.fix\." > /dev/null; do sleep 60; done

# re-check the cached patching path on the real model before trusting it
run mech_patch_check $M.patching --split train --n 3
run mech_patching $M.patching --split dev --n 100
run mech_similarity $M.similarity --split dev --n 150
run mech_probe $M.probe --split dev --n 150 --n-train 600
run mech_dose $M.dose_response --split dev --n 150 --n-decoys 40 --pool 2000
echo "$(date '+%H:%M:%S') MECH QUEUE DONE" >> $L/queue.log
