#!/usr/bin/env bash
# Watch download completion then run post-download pipeline.
set -euo pipefail
OUT=/data/yehang/Earthquake_paper_strengthening_v1/independent_period
LOG=$OUT/logs/watchdog.log
echo "[$(date -u +%FT%TZ)] watchdog start" | tee -a "$LOG"
while true; do
  if [[ -f "$OUT/download/download_summary.json" ]]; then
    echo "[$(date -u +%FT%TZ)] download_summary present" | tee -a "$LOG"
    break
  fi
  n=$(ls "$OUT/waveforms"/*.json 2>/dev/null | wc -l || echo 0)
  echo "[$(date -u +%FT%TZ)] waiting download… json=$n" | tee -a "$LOG"
  sleep 120
done
# ensure download process finished
while pgrep -f 'download_bsi_waveforms.py' >/dev/null; do
  echo "[$(date -u +%FT%TZ)] download still running" | tee -a "$LOG"
  sleep 60
done
bash /home/yehang/yehang/Earthquake/scripts/paper_strengthening_v1/run_post_download_pipeline.sh
echo "[$(date -u +%FT%TZ)] watchdog done" | tee -a "$LOG"
