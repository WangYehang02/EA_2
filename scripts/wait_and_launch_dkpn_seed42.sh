#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
SMOKE_JSON="/data/mnt_data/yehang/PSdetec/Earthquake_stage10/dkpn_clean/smoke_seed42/smoke_report.json"
LOG="$ROOT/artifacts/results/stage10/logs/wait_gpu_seed42.log"
free_gpus() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk -F',' '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($2+0 < 500) print $1}'
}
echo "$(date -u) waiter start" | tee -a "$LOG"
# wait for smoke
while true; do
  if [[ -f "$SMOKE_JSON" ]]; then
    ok=$(/home/yehang/miniconda3/envs/PS/bin/python -c "import json;print(json.load(open('$SMOKE_JSON')).get('smoke_ok',False))")
    if [[ "$ok" == "True" ]]; then
      echo "$(date -u) smoke_ok=True" | tee -a "$LOG"
      break
    fi
    echo "$(date -u) smoke file present but smoke_ok=$ok" | tee -a "$LOG"
  fi
  sleep 60
done
# wait for free GPU
while true; do
  g=$(free_gpus | head -1 || true)
  if [[ -n "${g:-}" ]]; then
    echo "$(date -u) free GPU $g — launching seed42" | tee -a "$LOG"
    exec bash "$ROOT/scripts/run_stage10_pipeline.sh" dkpn-seed42
  fi
  echo "$(date -u) waiting free GPU..." | tee -a "$LOG"
  sleep 120
done
