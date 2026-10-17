#!/bin/sh
# Mechanism runs on Shieldstral, after the pilot queue (one GPU job at a time).
# Dev split for development; test-split runs for the reported numbers are added here later.
set -u
cd "$(dirname "$0")/.."
. scripts/lib.sh
mkdir -p results/mech
M="uv run --frozen --with matplotlib python -m src.mech"
R=results/mech

while pgrep -f "queue_pilot.sh|src\.fix\." > /dev/null; do sleep 60; done

# re-check the cached patching path on the real model before trusting it
[ -s $R/patching_train_n3.json ]          || run mech_patch_check $M.patching --split train --n 3
[ -s $R/patching_dev_n100.json ]          || run mech_patching $M.patching --split dev --n 100
[ -s $R/similarity_dev_n150.json ]        || run mech_similarity $M.similarity --split dev --n 150
[ -s $R/probe_dev_n150_train600.json ]    || run mech_probe $M.probe --split dev --n 150 --n-train 600
[ -s $R/dose_response_dev_n150_d40.json ] || run mech_dose $M.dose_response --split dev --n 150 --n-decoys 40 --pool 2000

m=$(missing $R/patching_train_n3.json $R/patching_dev_n100.json $R/similarity_dev_n150.json \
  $R/probe_dev_n150_train600.json $R/dose_response_dev_n150_d40.json)
[ -z "$m" ] && echo "$(date '+%H:%M:%S') MECH QUEUE DONE" >> $L/queue.log || { echo "mech missing: $m" >> $L/queue.log; exit 1; }
