#!/bin/sh
# Keeps the Mac responsive: if free memory drops below LOW %, pause (SIGSTOP) every experiment
# process; resume (SIGCONT) once it recovers above HIGH %. Paused jobs lose nothing.
LOW=${LOW:-15}
HIGH=${HIGH:-30}
LOG=$(dirname "$0")/../logs/queue/mem_guard.log
paused=0
while true; do
  free=$(memory_pressure | awk -F': ' '/free percentage/ {gsub("%","",$2); print $2}')
  pids=$(pgrep -f "src\.(fix|bench|mech)\." | tr '\n' ' ')
  if [ -n "$free" ] && [ "$free" -lt "$LOW" ] && [ $paused -eq 0 ] && [ -n "$pids" ]; then
    kill -STOP $pids 2>/dev/null; paused=1
    echo "$(date '+%H:%M:%S') free=${free}% PAUSE $pids" >> "$LOG"
  elif [ $paused -eq 1 ] && [ -n "$free" ] && [ "$free" -gt "$HIGH" ]; then
    kill -CONT $pids 2>/dev/null; paused=0
    echo "$(date '+%H:%M:%S') free=${free}% RESUME" >> "$LOG"
  fi
  sleep 5
done
