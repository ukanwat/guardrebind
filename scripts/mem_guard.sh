#!/bin/sh
# Keeps the Mac usable and alive. Pauses (SIGSTOP) every experiment process when
#   - free memory drops below LOW %, or
#   - the Mac is on battery with less than BATT_MIN % left (a GPU job drains the battery fast)
# and resumes (SIGCONT) once memory is above HIGH % and the Mac is on AC power or above BATT_MIN %.
# Paused jobs lose nothing.
# GPU share: outside a pause, jobs run ON seconds then rest OFF seconds (SIGSTOP), so the GPU is idle part of
# the time and the Mac stays responsive. OFF=0 turns this off.
LOW=${LOW:-15}
HIGH=${HIGH:-30}
BATT_MIN=${BATT_MIN:-0}   # 0 = off (the person, 2026-10-16: run on battery too)
ON=${ON:-9}
OFF=${OFF:-1}   # the person, 2026-10-16: ~90% GPU, leave ~10% for normal Mac use
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
  if [ $paused -eq 0 ] && [ -n "$pids" ] && [ "$OFF" -gt 0 ]; then
    kill -STOP $pids 2>/dev/null; sleep $OFF; kill -CONT $pids 2>/dev/null; sleep $ON
  else
    sleep 5
  fi
done
