#!/bin/sh
# Simulates scripts/queue_round2.sh in a temp dir with a fake `uv` (writes each --out file, or the model dir
# and its DONE marker for `distill train`, and logs the call), a fake `pgrep` (nothing else running) and a
# no-op mem_guard. No python, no GPU.  sh scripts/test_queue_round2.sh
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
T=$(mktemp -d)
mkdir -p $T/sim/scripts $T/bin $T/sim/data/fix
cp $REPO/scripts/queue_round2.sh $REPO/scripts/lib.sh $T/sim/scripts/
: > $T/sim/scripts/mem_guard.sh
cat > $T/bin/pgrep <<'EOF'
#!/bin/sh
exit 1
EOF
cat > $T/bin/uv <<'EOF'
#!/bin/sh
echo "$*" >> "$CALLS"
case " $* " in *" $FAIL "*) [ -n "$FAIL" ] && exit 1;; esac
out=""; prev=""
for a in "$@"; do [ "$prev" = "--out" ] && out=$a; prev=$a; done
[ -n "$out" ] || exit 0
case " $* " in
  *".distill train "*) mkdir -p "$out" && echo "1 steps" > "$out/DONE" ;;
  *) mkdir -p "$(dirname "$out")" && echo '{}' > "$out" ;;
esac
EOF
chmod +x $T/bin/*
export PATH=$T/bin:$PATH CALLS=$T/calls TRIES=1 FAIL=""
fails=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fails=$((fails + 1)); fi; }
q() { (cd $T/sim && sh scripts/queue_round2.sh); }

# 1. Nothing done: everything runs once, in order, and the completeness check passes.
echo '{"text": "x", "score": 0.1}' > $T/sim/data/fix/full.teacher_cache.jsonl
: > $CALLS; q; rc=$?
check "fresh run exits 0" "[ $rc = 0 ]"
check "21 steps ran" "[ $(wc -l < $CALLS) = 21 ]"
check "round-1 teacher cache copied" "[ -s $T/sim/data/fix/full_r2.teacher_cache.jsonl ]"
check "r2 build uses train decoys" "grep -q 'distill build --out data/fix/full_r2.jsonl --max-items 5000 --per-item 2 --decoy-split train' $CALLS"
check "cc build reuses full_r2 cache" "grep -q 'clean_copy_r2.jsonl.*--mode clean_copy --decoy-split train --cache data/fix/full_r2.teacher_cache.jsonl' $CALLS"
check "no naive_at build" "! grep -q naive_at $CALLS"
check "4 trainings" "[ $(grep -c 'distill train' $CALLS) = 4 ]"
for m in t_seed0:stock t_seed1:stock bt_seed0:bounded cc_seed0:stock; do
  check "train r2_${m%%:*} (${m##*:})" "grep -q -- '--fmt ${m##*:} --seed .* --out models/fix/r2_${m%%:*}\$' $CALLS"
done
check "no r2_at / r2_bt_seed1" "! grep -q 'r2_at\|r2_bt_seed1' $CALLS"
check "8 evals, all fixed+heldout on test" "[ $(grep 'src.fix.evaluate' $CALLS | grep -c -- '--split test --allow-test --decoy-split fixed,heldout') = 8 ]"
for c in base B detector segmax_infer; do check "eval $c" "grep -q -- '--cond $c --split test' $CALLS"; done
check "3 search runs n=50 budget 300" "[ $(grep -c -- '--n 50 --attack search --budget 300' $CALLS) = 3 ]"
check "3 heldout structural runs" "[ $(grep -c -- '--attack structural --decoy-split heldout' $CALLS) = 3 ]"
check "1 GCG n=20 on r2_t_seed0" "[ $(grep -c -- '--attack gcg' $CALLS) = 1 ] && grep -q 'models/fix/r2_t_seed0 --split test --n 20 --attack gcg' $CALLS"
check "seed-1 training is the last training" "[ \"\$(grep 'distill train' $CALLS | tail -1 | grep -c r2_t_seed1)\" = 1 ]"
check "log says done" "grep -q 'QUEUE_ROUND2 done' $T/sim/logs/queue/queue_round2.log"

# 2. Everything done: nothing runs, exit 0.
: > $CALLS; q; rc=$?
check "rerun with all done exits 0" "[ $rc = 0 ]"
check "rerun runs nothing" "[ ! -s $CALLS ]"

# 3. One output lost: only it reruns.
rm $T/sim/results/fix/r2_test_detector.json
: > $CALLS; q; rc=$?
check "lost output: exit 0" "[ $rc = 0 ]"
check "lost output: only that step ran" "[ $(wc -l < $CALLS) = 1 ] && grep -q 'cond detector' $CALLS"

# 4. Fresh, with BT training failing: its evals/attacks are skipped, the rest runs, exit 1 with BT missing.
rm -rf $T/sim/data/fix/*_r2* $T/sim/models $T/sim/results $T/sim/logs
: > $CALLS; FAIL=bounded q; rc=$?
check "failed training: exit 1" "[ $rc = 1 ]"
check "failed training: no BT eval or attack" "! grep -q 'adapter models/fix/r2_bt_seed0' $CALLS"
check "failed training: T still attacked" "grep -q 'adapter models/fix/r2_t_seed0 --split test --n 20 --attack gcg' $CALLS"
check "failed training: missing reported" "grep -q 'round2 missing: models/fix/r2_bt_seed0/DONE results/fix/r2_test_bt_seed0.json' $T/sim/logs/queue/queue_round2.log"
check "failed training: FAILED logged" "grep -q 'FAILED train_r2_bt_seed0' $T/sim/logs/queue/queue_round2.log"

# 5. Fresh, with the r2 build failing: nothing that needs its data runs; base-model steps still do.
rm -rf $T/sim/data/fix/*_r2* $T/sim/models $T/sim/results $T/sim/logs
: > $CALLS; FAIL=data/fix/full_r2.jsonl q; rc=$?
check "failed build: exit 1" "[ $rc = 1 ]"
check "failed build: no training, no cc build" "! grep -q 'distill train\|clean_copy_r2' $CALLS"
check "failed build: base/B/detector/segmax evals + base attacks ran" "[ $(grep -c 'evaluate\|adaptive --cond base' $CALLS) = 6 ]"

[ -n "${KEEP:-}" ] && echo "kept $T" || rm -rf $T
[ $fails = 0 ] && echo "all queue_round2 simulations passed" || { echo "$fails checks failed"; exit 1; }
