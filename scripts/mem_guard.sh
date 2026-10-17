#!/bin/sh
# Keeps the Mac usable and alive. Pauses (SIGSTOP) every experiment process when
#   - free memory drops below LOW %, or
#   - the Mac is on battery with less than BATT_MIN % left (a GPU job drains the battery fast)
# and resumes (SIGCONT) once memory is above HIGH % and the Mac is on AC power or above BATT_MIN %.
# Paused jobs lose nothing.
LOW=${LOW:-15}
HIGH=${HIGH:-30}
BATT_MIN=${BATT_MIN:-40}
LOG=$(dirname "$0")/../logs/queue/mem_guard.log
paused=0
while true; do
  free=$(memory_pressure | awk -F': ' '/free percentage/ {gsub("%","",$2); print $2}')
  batt=$(pmset -g batt)
  pct=$(echo "$batt" | grep -o '[0-9]*%' | head -1 | tr -d %)
  low_batt=0
  echo "$batt" | grep -q "Battery Power" && [ -n "$pct" ] && [ "$pct" -lt "$BATT_MIN" ] && low_batt=1
  pids=$(pgrep -f "src\.(fix|bench|mech)\." | tr '\n' ' ')
  if [ $paused -eq 0 ] && [ -n "$pids" ] && { { [ -n "$free" ] && [ "$free" -lt "$LOW" ]; } || [ $low_batt -eq 1 ]; }; then
    kill -STOP $pids 2>/dev/null; paused=1
    echo "$(date '+%H:%M:%S') free=${free}% batt=${pct}% low_batt=$low_batt PAUSE $pids" >> "$LOG"
  elif [ $paused -eq 1 ] && { [ $low_batt -eq 1 ] || [ -z "$free" ] || [ "$free" -le "$HIGH" ]; }; then
    # still paused: also stop jobs that started after the pause (e.g. the queue's next step)
    [ -n "$pids" ] && kill -STOP $pids 2>/dev/null
  elif [ $paused -eq 1 ]; then
    kill -CONT $pids 2>/dev/null; paused=0
    echo "$(date '+%H:%M:%S') free=${free}% batt=${pct}% RESUME" >> "$LOG"
  fi
  sleep 5
done
