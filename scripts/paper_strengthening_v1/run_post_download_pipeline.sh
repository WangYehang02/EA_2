#!/usr/bin/env bash
# Post-download pipeline: candidates → READY → formal A–E eval → report.
set -euo pipefail
ROOT=/home/yehang/yehang/Earthquake
OUT=/data/yehang/Earthquake_paper_strengthening_v1/independent_period
LOG=$OUT/logs
mkdir -p "$LOG"
cd "$ROOT"
source /home/yehang/miniconda3/etc/profile.d/conda.sh
conda activate PS
# process-local proxy bypass
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true

echo "[$(date -u +%FT%TZ)] candidates (multi-GPU)" | tee -a "$LOG/pipeline.log"
python -W ignore scripts/paper_strengthening_v1/build_independent_candidates_pairs.py 2>&1 | tee -a "$LOG/candidates.log"

echo "[$(date -u +%FT%TZ)] READY" | tee -a "$LOG/pipeline.log"
python scripts/paper_strengthening_v1/build_ready_and_data_lock.py 2>&1 | tee -a "$LOG/ready.log"

echo "[$(date -u +%FT%TZ)] formal eval" | tee -a "$LOG/pipeline.log"
python scripts/paper_strengthening_v1/run_formal_independent_eval.py 2>&1 | tee -a "$LOG/eval.log"

echo "[$(date -u +%FT%TZ)] report" | tee -a "$LOG/pipeline.log"
python scripts/paper_strengthening_v1/write_independent_validation_report.py 2>&1 | tee -a "$LOG/report.log"

echo "[$(date -u +%FT%TZ)] DONE" | tee -a "$LOG/pipeline.log"
