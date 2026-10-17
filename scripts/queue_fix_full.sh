#!/bin/sh
# Full fix experiment queue (PROTOCOL_v2 Fix + Adaptive). One GPU job at a time,
# serial. Each step logs to logs/queue/<step>.log and is skipped if its output is
# already on disk, so the queue is safe to relaunch. Waits at startup until no
# other queue_*/src.* GPU job is running, so it does not collide with
# queue_all/queue_pilot/queue_bench. Exits non-zero if anything is still missing, so
# scripts/supervisor.sh runs it again.
#
#   sh scripts/queue_fix_full.sh      # run it (do NOT run while another queue is live)
set -u
cd "$(dirname "$0")/.."
# Good neighbour: low CPU priority and an MPS memory cap so the Mac stays usable (scripts/lib.sh).
QLOG=logs/queue/queue_fix.log
. scripts/lib.sh
mkdir -p data/fix models/fix results/fix

# Wait until no other fix/bench/mech GPU job and no sibling queue script is live.
# Called once at startup: after this, our own steps run serially so we are the
# only GPU job.
wait_free() {
  while pgrep -f "scripts/queue_(all|pilot|mech|bench)\.sh" > /dev/null \
     || pgrep -f "src\.(fix|bench|mech)\." > /dev/null; do
    echo "$(date '+%H:%M:%S') waiting for another GPU job to finish" >> $L/queue_fix.log
    sleep 30
  done
}

echo "$(date '+%H:%M:%S') QUEUE_FIX start" >> $L/queue_fix.log
pgrep -f mem_guard.sh > /dev/null || (nohup sh scripts/mem_guard.sh > /dev/null 2>&1 &)
wait_free

PY="uv run --frozen python"

# 0. Mechanism, TEST split (the reported numbers; same settings as the dev runs) -- #
M="uv run --frozen --with matplotlib python -m src.mech"
R=results/mech
[ -s $R/patching_test_n100.json ]          || run mech_test_patching $M.patching --split test --allow-test --n 100
[ -s $R/similarity_test_n150.json ]        || run mech_test_similarity $M.similarity --split test --allow-test --n 150
[ -s $R/probe_test_n150_train600.json ]    || run mech_test_probe $M.probe --split test --allow-test --n 150 --n-train 600
[ -s $R/dose_response_test_n150_d40.json ] || run mech_test_dose $M.dose_response --split test --allow-test --n 150 \
  --n-decoys 40 --pool 2000

# 1. Build data -------------------------------------------------------------- #
# Segment-max full set (the teacher cache lands next to it and is reused).
[ -s data/fix/full.jsonl ] || \
  run build_full $PY -m src.fix.distill build --out data/fix/full.jsonl --max-items 5000 --per-item 2
# Naive adversarial-training set: same items/attacks, hard labels, no teacher.
[ -s data/fix/naive_at.jsonl ] || \
  run build_naive $PY -m src.fix.distill build --out data/fix/naive_at.jsonl --max-items 5000 --per-item 2 --mode naive_at
# Clean-copy ablation (teacher on the original item only, as in SecOPD/RAISED): same examples, targets from
# the full build's teacher cache, so this costs no new forward passes.
[ -s data/fix/clean_copy.jsonl ] || \
  run build_cc $PY -m src.fix.distill build --out data/fix/clean_copy.jsonl --max-items 5000 --per-item 2 \
    --mode clean_copy --cache data/fix/full.teacher_cache.jsonl

# 2-4. Models, test evals, controls and adaptive attacks ---------------------- #
#   T  = stock format  + segment-max       (data/fix/full.jsonl)
#   BT = bounded format + segment-max       (data/fix/full.jsonl)
#   AT = bounded format + naive hard labels (data/fix/naive_at.jsonl); the naive adversarial-training
#        baseline (F6), bounded so only the labelling scheme differs from BT. Evaluated as cond BT.
#   CC = stock format  + clean-copy target  (data/fix/clean_copy.jsonl); ablation against T. Evaluated as cond T.
# Seed 0 of everything is trained, evaluated and attacked first, so the deciding results arrive ~1.5 days
# before the extra seeds (T/AT seed 1, BT seeds 1-2), which only add error bars.
train() { kind=$1; s=$2; data=$3; fmt=$4
  trained models/fix/${kind}_seed$s || run train_${kind}_seed$s $PY -m src.fix.distill train --data $data \
    --fmt $fmt --seed $s --out models/fix/${kind}_seed$s; }
evalm() { kind=$1; s=$2; cond=$3
  ! trained models/fix/${kind}_seed$s || [ -f results/fix/test_${kind}_seed$s.json ] || \
    run eval_${kind}_seed$s $PY -m src.fix.evaluate --cond $cond --adapter models/fix/${kind}_seed$s \
      --split test --allow-test --out results/fix/test_${kind}_seed$s.json; }

train t  0 data/fix/full.jsonl       stock
train bt 0 data/fix/full.jsonl       bounded
train at 0 data/fix/naive_at.jsonl   bounded
train cc 0 data/fix/clean_copy.jsonl stock

[ -f results/fix/test_base.json ] || \
  run eval_base $PY -m src.fix.evaluate --cond base --split test --allow-test --out results/fix/test_base.json
[ -f results/fix/test_B.json ] || \
  run eval_B $PY -m src.fix.evaluate --cond B --split test --allow-test --out results/fix/test_B.json
evalm t 0 T; evalm bt 0 BT; evalm at 0 BT; evalm cc 0 T

# Forged verdict vs forged task, and the split-payload probe (exploratory), on base and the seed-0 fixes.
fc() { tag=$1; shift; [ -f results/fix/forgery_controls_$tag.json ] || \
  run forgery_controls_$tag $PY -m src.fix.forgery_controls "$@" --split test --allow-test \
    --out results/fix/forgery_controls_$tag.json; }
fc base --cond base
trained models/fix/t_seed0  && fc t_seed0  --cond T  --adapter models/fix/t_seed0
trained models/fix/bt_seed0 && fc bt_seed0 --cond BT --adapter models/fix/bt_seed0

# Adaptive attacks on base, B, T-seed0, BT-seed0. GCG: 250 steps, 20-token suffix, n=20 screened test items,
# batch 32 (PROTOCOL_v2 D3). Structural search: n=50. Both resumable per item (<out>.jsonl).
adaptive() { cond=$1; tag=$2; shift 2
  [ -f results/fix/adaptive_${tag}_gcg.json ] || \
    run adaptive_${tag}_gcg $PY -m src.fix.adaptive --cond $cond "$@" --split test \
      --n 20 --attack gcg --steps 250 --batch 32 --out results/fix/adaptive_${tag}_gcg.json
  [ -f results/fix/adaptive_${tag}_structural.json ] || \
    run adaptive_${tag}_structural $PY -m src.fix.adaptive --cond $cond "$@" --split test \
      --n 50 --attack structural --out results/fix/adaptive_${tag}_structural.json
}
adaptive base base
adaptive B    B
trained models/fix/t_seed0  && adaptive T  t_seed0  --adapter models/fix/t_seed0
trained models/fix/bt_seed0 && adaptive BT bt_seed0 --adapter models/fix/bt_seed0

# Convergence check (Carlini): doubling the GCG budget on a subset should change bypass by < 2 points.
! trained models/fix/bt_seed0 || [ -f results/fix/adaptive_bt_seed0_gcg500.json ] || \
  run adaptive_bt_seed0_gcg500 $PY -m src.fix.adaptive --cond BT --adapter models/fix/bt_seed0 --split test \
    --n 10 --attack gcg --steps 500 --batch 32 --out results/fix/adaptive_bt_seed0_gcg500.json

# Extra seeds (PROTOCOL_v2 D3: BT 3 seeds, T and AT 2).
train t  1 data/fix/full.jsonl     stock;   evalm t  1 T
train bt 1 data/fix/full.jsonl     bounded; evalm bt 1 BT
train at 1 data/fix/naive_at.jsonl bounded; evalm at 1 BT
train bt 2 data/fix/full.jsonl     bounded; evalm bt 2 BT

F=results/fix
m=$(missing $F/test_base.json $F/test_B.json $F/test_t_seed0.json $F/test_t_seed1.json $F/test_at_seed0.json \
  $F/test_at_seed1.json $F/test_bt_seed0.json $F/test_bt_seed1.json $F/test_bt_seed2.json \
  $F/adaptive_base_gcg.json $F/adaptive_base_structural.json $F/adaptive_B_gcg.json $F/adaptive_B_structural.json \
  $F/adaptive_t_seed0_gcg.json $F/adaptive_t_seed0_structural.json $F/adaptive_bt_seed0_gcg.json \
  $F/adaptive_bt_seed0_structural.json $F/adaptive_bt_seed0_gcg500.json \
  $R/patching_test_n100.json $R/similarity_test_n150.json $R/probe_test_n150_train600.json \
  $R/dose_response_test_n150_d40.json $F/test_cc_seed0.json $F/forgery_controls_base.json \
  $F/forgery_controls_t_seed0.json $F/forgery_controls_bt_seed0.json)
[ -z "$m" ] && echo "$(date '+%H:%M:%S') QUEUE_FIX done" >> $QLOG || { echo "fix missing: $m" >> $QLOG; exit 1; }
