#!/bin/sh
# Round-2 queue (after peer review): the fix trained on diverse decoys and evaluated on decoys it never saw.
# One GPU job at a time, serial. Each step logs to logs/queue/<step>.log and is skipped if its output is
# already on disk, so the queue is safe to relaunch. Waits at startup until no other queue_*/src.* GPU job
# is running. Exits non-zero if anything is still missing, so a supervisor runs it again.
#
#   sh scripts/queue_round2.sh      # do NOT run while another queue is live
#
# Lean set (to save GPU): T seeds 0-1, BT seed 0 and the clean-copy ablation CC seed 0, all on r2 data
# (decoy and forged query drawn per example from the TRAIN halves of the pools, attacks_grammar.POOL_SPLIT).
# No naive-AT model and no BT seed 1 in round 2, so no naive_at build.
set -u
cd "$(dirname "$0")/.."
QLOG=logs/queue/queue_round2.log
. scripts/lib.sh
mkdir -p data/fix models/fix results/fix

wait_free() {
  while pgrep -f "scripts/queue_(all|pilot|mech|bench|fix_full)\.sh" > /dev/null \
     || pgrep -f "src\.(fix|bench|mech)\." > /dev/null; do
    echo "$(date '+%H:%M:%S') waiting for another GPU job to finish" >> $QLOG
    sleep 30
  done
}

PY="uv run --frozen python"
F=results/fix

# Shared helpers (defined before any conditional block).
# train <name> <data> <fmt> <seed>: models/fix/r2_<name>_seed<seed>
train() { kind=$1; data=$2; fmt=$3; s=$4
  [ -s $data ] || return 0
  trained models/fix/r2_${kind}_seed$s || run train_r2_${kind}_seed$s $PY -m src.fix.distill train --data $data \
    --fmt $fmt --seed $s --out models/fix/r2_${kind}_seed$s; }
# evalc <tag> <evaluate args...>: results/fix/r2_test_<tag>.json, fixed and heldout decoys
evalc() { tag=$1; shift
  [ -f $F/r2_test_$tag.json ] || run eval_r2_$tag $PY -m src.fix.evaluate "$@" --split test --allow-test \
    --decoy-split fixed,heldout --out $F/r2_test_$tag.json; }
# evalm <kind> <seed> <cond>: a trained r2 model
evalm() { ! trained models/fix/r2_$1_seed$2 || evalc $1_seed$2 --cond $3 --adapter models/fix/r2_$1_seed$2; }
# adaptive <cond> <tag> [--adapter dir]: score-guided search (n=50, 300 calls/item) and structural
# search with heldout decoys (n=50), both resumable per item (<out>.jsonl)
adaptive() { cond=$1; tag=$2; shift 2
  [ -f $F/r2_adaptive_${tag}_search.json ] || \
    run adaptive_r2_${tag}_search $PY -m src.fix.adaptive --cond $cond "$@" --split test \
      --n 50 --attack search --budget 300 --out $F/r2_adaptive_${tag}_search.json
  [ -f $F/r2_adaptive_${tag}_structural_heldout.json ] || \
    run adaptive_r2_${tag}_structural $PY -m src.fix.adaptive --cond $cond "$@" --split test \
      --n 50 --attack structural --decoy-split heldout --out $F/r2_adaptive_${tag}_structural_heldout.json
}

echo "$(date '+%H:%M:%S') QUEUE_ROUND2 start" >> $QLOG
pgrep -f mem_guard.sh > /dev/null || (nohup sh scripts/mem_guard.sh > /dev/null 2>&1 &)
wait_free

# 1. Data ------------------------------------------------------------------- #
# The teacher cache is keyed by text, so the round-1 cache (clean items, structured benign, index-0
# decoys) is copied in to skip forward passes already paid for.
[ -s data/fix/full_r2.teacher_cache.jsonl ] || [ ! -s data/fix/full.teacher_cache.jsonl ] || \
  cp data/fix/full.teacher_cache.jsonl data/fix/full_r2.teacher_cache.jsonl
[ -s data/fix/full_r2.jsonl ] || \
  run build_r2 $PY -m src.fix.distill build --out data/fix/full_r2.jsonl --max-items 5000 --per-item 2 \
    --decoy-split train
# Clean-copy ablation: same examples, targets from full_r2's cache (no new forward passes).
[ ! -s data/fix/full_r2.jsonl ] || [ -s data/fix/clean_copy_r2.jsonl ] || \
  run build_r2_cc $PY -m src.fix.distill build --out data/fix/clean_copy_r2.jsonl --max-items 5000 --per-item 2 \
    --mode clean_copy --decoy-split train --cache data/fix/full_r2.teacher_cache.jsonl

# 2. Models (seed 0 first; T seed 1 last, it only adds an error bar) ------- #
train t  data/fix/full_r2.jsonl       stock   0
train bt data/fix/full_r2.jsonl       bounded 0
train cc data/fix/clean_copy_r2.jsonl stock   0

# 3. Test evals, fixed + heldout decoys -------------------------------------- #
evalc base --cond base
evalc B --cond B
evalc detector --cond detector
evalc segmax_infer --cond segmax_infer
evalm t 0 T; evalm bt 0 BT; evalm cc 0 T

# 4. Adaptive attacks: search + heldout-decoy structural on base, T, BT; GCG on T for the appendix ---- #
adaptive base base
trained models/fix/r2_t_seed0  && adaptive T  t_seed0  --adapter models/fix/r2_t_seed0
trained models/fix/r2_bt_seed0 && adaptive BT bt_seed0 --adapter models/fix/r2_bt_seed0
! trained models/fix/r2_t_seed0 || [ -f $F/r2_adaptive_t_seed0_gcg.json ] || \
  run adaptive_r2_t_seed0_gcg $PY -m src.fix.adaptive --cond T --adapter models/fix/r2_t_seed0 --split test \
    --n 20 --attack gcg --steps 250 --batch 32 --out $F/r2_adaptive_t_seed0_gcg.json

# 5. Extra seed ------------------------------------------------------------- #
train t data/fix/full_r2.jsonl stock 1
evalm t 1 T

m=$(missing data/fix/full_r2.jsonl data/fix/clean_copy_r2.jsonl \
  models/fix/r2_t_seed0/DONE models/fix/r2_t_seed1/DONE models/fix/r2_bt_seed0/DONE models/fix/r2_cc_seed0/DONE \
  $F/r2_test_base.json $F/r2_test_B.json $F/r2_test_detector.json $F/r2_test_segmax_infer.json \
  $F/r2_test_t_seed0.json $F/r2_test_t_seed1.json $F/r2_test_bt_seed0.json $F/r2_test_cc_seed0.json \
  $F/r2_adaptive_base_search.json $F/r2_adaptive_base_structural_heldout.json \
  $F/r2_adaptive_t_seed0_search.json $F/r2_adaptive_t_seed0_structural_heldout.json \
  $F/r2_adaptive_bt_seed0_search.json $F/r2_adaptive_bt_seed0_structural_heldout.json \
  $F/r2_adaptive_t_seed0_gcg.json)
m=$(echo $m)
[ -z "$m" ] && echo "$(date '+%H:%M:%S') QUEUE_ROUND2 done" >> $QLOG || { echo "round2 missing: $m" >> $QLOG; exit 1; }
