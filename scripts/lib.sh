# Shared by the queue scripts (sourced). One GPU job at a time; a failed step is retried, and every
# step resumes from what is already on disk, so a retry or a relaunch after a crash loses little.
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.4
L=logs/queue
mkdir -p $L
QLOG=${QLOG:-$L/queue.log}
TRIES=${TRIES:-3}
# run <name> <cmd...>: log START/END to $QLOG, output to logs/queue/<name>.log; retry up to $TRIES times.
run() { name=$1; shift; try=1
  while :; do
    t=""; [ $try -gt 1 ] && t=" (try $try)"
    echo "$(date '+%H:%M:%S') START $name$t" >> $QLOG
    nice -n 10 "$@" >> $L/$name.log 2>&1; rc=$?
    echo "$(date '+%H:%M:%S') END $name exit=$rc" >> $QLOG
    [ $rc -eq 0 ] && return 0
    [ $try -ge $TRIES ] && { echo "$(date '+%H:%M:%S') FAILED $name after $try tries" >> $QLOG; return $rc; }
    try=$((try + 1)); sleep 60
  done; }
# trained <dir>: the trainer writes DONE only after every file is saved.
trained() { [ -f "$1/DONE" ]; }
# missing <files...>: print the ones not on disk (non-empty), for the end-of-queue completeness check.
missing() { for f in "$@"; do [ -s "$f" ] || echo "$f"; done; }
