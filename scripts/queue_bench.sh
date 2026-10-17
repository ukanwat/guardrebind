#!/bin/sh
# GuardRebind v2 benchmark queue (PROTOCOL_v2 D1, amended: K=3, policy paraphrases on 6 families, screened).
# One GPU job at a time, behind the fix/mech queues. Each step logs to logs/queue/<step>.log; a step whose
# output is on disk is skipped and run_bench_v2 resumes cell by cell, so the queue can be rerun.
# Exits non-zero if anything is still missing, so scripts/supervisor.sh runs it again.
#
# Estimated GPU time on this Mac (M4 Max), from a timed train-split run (480 cells/guard, batch 16, unsorted,
# GPU shared with the distill build): shieldstral 5.3 cells/s, llama_guard_3_1b 8.2/s, shieldgemma_2b 2.3/s.
# Cells per guard (n=600+600): 121.2k with every harmful item attacked; ~87k if ~70% pass the screen.
#   unsorted rates:            shieldstral 4.6-6.4h, llama_guard 2.9-4.1h, shieldgemma 10.5-14.6h = 18-25h
#   token-length batching is expected to cut this by roughly a third (not yet timed: run the smoke step and
#   read the cells/s in logs/queue/bench_<guard>.log). ShieldGemma is ~60% of the total; if it runs over,
#   it is last in the queue and resumable, so it can be stopped and finished later.
set -u
cd "$(dirname "$0")/.."
. scripts/lib.sh
OUT=results/bench/v2
mkdir -p $OUT results/bench/v2_smoke
B="uv run --frozen python -m src.bench"

# Wait for the GPU: no fix/mech job and no other queue holding it. Launch standalone; do not call this from
# queue_all.sh (it would wait on its own parent).
while pgrep -f "src\.fix\.|src\.mech\.|queue_pilot.sh|queue_mech.sh|queue_all.sh" > /dev/null; do sleep 60; done
pgrep -f mem_guard.sh > /dev/null || (nohup sh scripts/mem_guard.sh > /dev/null 2>&1 &)

# 0. smoke: 20 cells per guard x policy on the TRAIN split (catches load/memory problems before the test run)
[ -f results/bench/v2_smoke/shieldgemma_2b.smoke ] || {
  run bench_smoke $B.run_bench_v2 --guards shieldstral,llama_guard_3_1b,qwen3guard_0_6b,shieldgemma_2b --split train --n 5 \
    --max-cells 20 --out results/bench/v2_smoke \
  && touch results/bench/v2_smoke/shieldgemma_2b.smoke || { echo "bench smoke failed" >> $L/queue.log; exit 1; }; }

# 1-3. the benchmark, test split, core_set(n=600), K=3, 3 policies; headline guard first, slowest last
for g in shieldstral llama_guard_3_1b qwen3guard_0_6b shieldgemma_2b; do
  [ -f $OUT/$g.done ] || run bench_$g $B.run_bench_v2 --guards $g --split test --n 600 --k 3 \
    --policies default,para1,para2 --out $OUT
done

# 4. stability (D1(f)): 200 random cells per guard, re-scored twice and batched vs unbatched
[ -s $OUT/stability.json ] || run bench_stability $B.stability --inp $OUT --n 200

# 5. analysis (CPU): tables, figures, summary.md
run bench_analyze uv run --frozen --with matplotlib python -m src.bench.analyze --inp $OUT

m=$(missing $OUT/shieldstral.done $OUT/llama_guard_3_1b.done $OUT/qwen3guard_0_6b.done $OUT/shieldgemma_2b.done \
  $OUT/stability.json $OUT/summary.md)
# .done markers are empty files: check existence, not size
m=$(for f in $m; do [ -e "$f" ] || echo "$f"; done)
[ -z "$m" ] && echo "$(date '+%H:%M:%S') BENCH QUEUE DONE" >> $L/queue.log || { echo "bench missing: $m" >> $L/queue.log; exit 1; }
