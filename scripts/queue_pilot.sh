#!/bin/sh
# Overnight queue: one GPU job at a time. Each step logs to logs/queue/<step>.log;
# a step that already wrote its output is skipped, so the queue can be rerun.
set -u
cd "$(dirname "$0")/.."
L=logs/queue
mkdir -p $L models/fix results/fix results/bench/core
# Be a good neighbour: low CPU priority and a cap on how much unified memory MPS may take,
# so the Mac stays usable and other jobs keep running.
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.75 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6
run() { name=$1; shift; echo "$(date '+%H:%M:%S') START $name" >> $L/queue.log
  nice -n 10 "$@" > $L/$name.log 2>&1; echo "$(date '+%H:%M:%S') END $name exit=$?" >> $L/queue.log; }

# 1. wait for the pilot distillation data
while pgrep -f "src.fix.distill build" > /dev/null; do sleep 30; done
[ -s data/fix/pilot.jsonl ] || { echo "no pilot data" >> $L/queue.log; exit 1; }

# 2-3. pilot fixes: T (stock format) and BT (bounded format)
[ -f models/fix/t_pilot/adapter_config.json ] || run train_t uv run --frozen python -m src.fix.distill train --data data/fix/pilot.jsonl --fmt stock --out models/fix/t_pilot
[ -f models/fix/bt_pilot/adapter_config.json ] || run train_bt uv run --frozen python -m src.fix.distill train --data data/fix/pilot.jsonl --fmt bounded --out models/fix/bt_pilot

# 4. dev-split evaluation of the 2x2
[ -f results/fix/pilot_base.json ] || run eval_base uv run --frozen python -m src.fix.evaluate --cond base --split dev --out results/fix/pilot_base.json
[ -f results/fix/pilot_B.json ] || run eval_B uv run --frozen python -m src.fix.evaluate --cond B --split dev --out results/fix/pilot_B.json
[ -f results/fix/pilot_T.json ] || run eval_T uv run --frozen python -m src.fix.evaluate --cond T --adapter models/fix/t_pilot --split dev --out results/fix/pilot_T.json
[ -f results/fix/pilot_BT.json ] || run eval_BT uv run --frozen python -m src.fix.evaluate --cond BT --adapter models/fix/bt_pilot --split dev --out results/fix/pilot_BT.json

# 5. GuardRebind core panel: moved to scripts/queue_bench.sh (variability design, PROTOCOL_v2 deviation D1)
exit 0
run bench_core uv run --frozen python -m src.bench.run_bench --guards shieldstral,llama_guard_3_1b,shieldgemma_2b --attacks core --items core --n 600 --out results/bench/core
echo "$(date '+%H:%M:%S') QUEUE DONE" >> $L/queue.log
