#!/bin/sh
# Pilot queue: train T and BT on the pilot data, then evaluate base/B/T/BT on the dev split.
# One GPU job at a time; a step whose output is on disk is skipped, so the queue can be rerun.
set -u
cd "$(dirname "$0")/.."
. scripts/lib.sh
mkdir -p models/fix results/fix
[ -s data/fix/pilot.jsonl ] || { echo "no pilot data" >> $L/queue.log; exit 1; }
PY="uv run --frozen python"

# The pilots predate the DONE marker: a saved adapter (plus boundary rows for BT) also counts.
pilot_t()  { trained models/fix/t_pilot  || [ -s models/fix/t_pilot/adapter_model.safetensors ]; }
pilot_bt() { trained models/fix/bt_pilot || [ -s models/fix/bt_pilot/boundary_embeddings.pt ]; }

pilot_t  || run train_t  $PY -m src.fix.distill train --data data/fix/pilot.jsonl --fmt stock   --out models/fix/t_pilot
pilot_bt || run train_bt $PY -m src.fix.distill train --data data/fix/pilot.jsonl --fmt bounded --out models/fix/bt_pilot

[ -s results/fix/pilot_base.json ] || run eval_base $PY -m src.fix.evaluate --cond base --split dev --out results/fix/pilot_base.json
[ -s results/fix/pilot_B.json ]    || run eval_B    $PY -m src.fix.evaluate --cond B    --split dev --out results/fix/pilot_B.json
pilot_t  && { [ -s results/fix/pilot_T.json ]  || run eval_T  $PY -m src.fix.evaluate --cond T  --adapter models/fix/t_pilot  --split dev --out results/fix/pilot_T.json; }
pilot_bt && { [ -s results/fix/pilot_BT.json ] || run eval_BT $PY -m src.fix.evaluate --cond BT --adapter models/fix/bt_pilot --split dev --out results/fix/pilot_BT.json; }

m=$(missing results/fix/pilot_base.json results/fix/pilot_B.json results/fix/pilot_T.json results/fix/pilot_BT.json)
[ -z "$m" ] && echo "$(date '+%H:%M:%S') PILOT QUEUE DONE" >> $L/queue.log || { echo "pilot missing: $m" >> $L/queue.log; exit 1; }
