#!/usr/bin/env bash
# Wait for truly idle GPUs. Default: throughput then exit (no full train).
# Usage: wait_and_launch_dkpn_v2.sh [status|throughput|pilot]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$ROOT/artifacts/results/stage10/logs/dkpn_v2_wait.log"
MODE="${1:-throughput}"
chmod +x "$ROOT/scripts/launch_dkpn_v2_train.sh"
if [[ "$MODE" == "status" ]]; then
  bash "$ROOT/scripts/launch_dkpn_v2_train.sh" status
  exit 0
fi
while true; do
  bash "$ROOT/scripts/launch_dkpn_v2_train.sh" status | tee -a "$LOG" >/dev/null || true
  if [[ "$MODE" == "pilot" && -f /data/mnt_data/yehang/PSdetec/Earthquake_stage10/dkpn_clean_v2_seed42/PILOT.FAILED ]]; then
    echo "$(date -u) PILOT.FAILED" | tee -a "$LOG"
    exit 1
  fi
  if [[ "$MODE" == "pilot" && -f /data/mnt_data/yehang/PSdetec/Earthquake_stage10/dkpn_clean_v2_seed42/PILOT.PASSED ]]; then
    echo "$(date -u) PILOT.PASSED — launching continue-train" | tee -a "$LOG"
    bash "$ROOT/scripts/launch_dkpn_v2_train.sh" train >>"$LOG" 2>&1 && exit 0 || {
      code=$?
      if [[ $code -eq 3 ]]; then sleep 120; continue; fi
      if [[ $code -eq 4 ]]; then echo "$(date -u) refuse duplicate train" | tee -a "$LOG"; exit 4; fi
      exit $code
    }
  fi
  bash "$ROOT/scripts/launch_dkpn_v2_train.sh" "$MODE" >>"$LOG" 2>&1 && exit 0 || {
    code=$?
    if [[ $code -eq 3 ]]; then
      sleep 120
      continue
    fi
    echo "$(date -u) launch exit $code" | tee -a "$LOG"
    exit $code
  }
done
