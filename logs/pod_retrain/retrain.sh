#!/bin/bash
# On the pod, after prefetch: build full_r2, sanity-check it, train T seed 0 only if SANITY_OK, then evaluate.
# Same commands and seed as scripts/queue_round2.sh. Each step is skipped if its output exists.
cd /workspace/gpi
export PATH=/workspace/bin:$PATH UV_CACHE_DIR=/workspace/.uvcache UV_PROJECT_ENVIRONMENT=/workspace/venv
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 UV_NO_SYNC=1
PY="uv run --frozen python"
L=logs/queue; mkdir -p $L data/fix models/fix results/fix
s() { echo "$(date '+%H:%M:%S') $*" >> $L/retrain.log; }
s START
[ -s data/fix/full_r2.jsonl ] || { s build; $PY -m src.fix.distill build --out data/fix/full_r2.jsonl \
  --max-items 5000 --per-item 2 --decoy-split train >> $L/build_r2.log 2>&1 || { s BUILD_FAILED; exit 1; }; }
PYTHONPATH=. $PY sanity_r2.py data/fix/full_r2.jsonl > $L/sanity_r2.log 2>&1
grep -q SANITY_OK $L/sanity_r2.log || { s SANITY_FAILED; exit 1; }
s sanity_ok
[ -f models/fix/r2_t_seed0/DONE ] || { s train; $PY -m src.fix.distill train --data data/fix/full_r2.jsonl \
  --fmt stock --seed 0 --out models/fix/r2_t_seed0 >> $L/train_r2_t_seed0.log 2>&1 || { s TRAIN_FAILED; exit 1; }; }
s trained
[ -f results/fix/r2_test_t_seed0_retrain.json ] || { s eval; $PY -m src.fix.evaluate --cond T \
  --adapter models/fix/r2_t_seed0 --split test --allow-test --decoy-split fixed,heldout \
  --out results/fix/r2_test_t_seed0_retrain.json >> $L/eval_r2_t_seed0_retrain.log 2>&1 || { s EVAL_FAILED; exit 1; }; }
s ALL_DONE
