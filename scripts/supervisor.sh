#!/bin/sh
# Runs every experiment queue until all outputs exist, surviving crashes and reboots.
# launchd starts it at login (scripts/io.guardpaper.supervisor.plist) and restarts it if it dies.
# Each pass reruns the queues in order; every step skips finished work and resumes partial work,
# so a pass after a crash only redoes what was lost. Gives up after MAX_PASSES and logs GAVE_UP.
set -u
cd "$(dirname "$0")/.."
L=logs/queue
mkdir -p $L
MAX_PASSES=${MAX_PASSES:-5}
log() { echo "$(date '+%H:%M:%S') SUPERVISOR $*" >> $L/queue.log; }

# single instance
LOCK=$L/supervisor.lock
if ! mkdir $LOCK 2>/dev/null; then
  pid=$(cat $LOCK/pid 2>/dev/null)
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && exit 0
  rm -rf $LOCK; mkdir $LOCK || exit 0
fi
echo $$ > $LOCK/pid
trap 'rm -rf $LOCK' EXIT
trap 'exit 1' TERM INT

[ -f $L/ALL_DONE ] && exit 0
log "start (pid $$)"

# Let a queue launched by hand finish first, so two GPU jobs never overlap.
while pgrep -f "scripts/queue_(all|pilot|mech|bench|fix_full)\.sh" > /dev/null; do sleep 60; done

pass=1
while :; do
  pgrep -f mem_guard.sh > /dev/null || (nohup sh scripts/mem_guard.sh > /dev/null 2>&1 &)
  free_gb=$(df -g . | awk 'NR==2 {print $4}')
  if [ "$free_gb" -lt 15 ]; then log "DISK_LOW ${free_gb}GB free; waiting"; sleep 600; continue; fi
  log "pass $pass"
  ok=1
  sh scripts/queue_all.sh || ok=0
  sh scripts/queue_bench.sh || ok=0
  sh scripts/queue_fix_full.sh || ok=0
  if [ $ok = 1 ]; then log "ALL_DONE"; touch $L/ALL_DONE; exit 0; fi
  if [ $pass -ge $MAX_PASSES ]; then log "GAVE_UP after $pass passes; see FAILED lines and step logs"; exit 0; fi
  pass=$((pass + 1)); sleep 300
done
