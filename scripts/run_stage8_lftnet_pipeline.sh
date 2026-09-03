#!/usr/bin/env bash
# Stage 8 LFTNet pipeline — stops at provenance/smoke gate unless complete official release.
set -euo pipefail
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate PS
LOG=artifacts/results/stage8/logs
mkdir -p "$LOG" artifacts/results/stage8 reports/stage8

echo "[stage8] free GPUs (informational; no full infer if gate fails):"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | tee "$LOG/gpu_at_start.txt"
mapfile -t FREE < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F',' '$2+0 < 500 {gsub(/ /,"",$1); print $1}')
echo "FREE_GPUS=${FREE[*]:-none}" | tee -a "$LOG/gpu_at_start.txt"

python -u scripts/audit_lftnet_release.py | tee "$LOG/audit.log"
python -u scripts/smoke_lftnet.py | tee "$LOG/smoke.log"
python -u scripts/stage8_finalize_blocked.py | tee "$LOG/finalize.log"
conda run -n PS --no-capture-output pytest -q tests/test_stage8_lftnet_gate.py 2>&1 | tee "$LOG/pytest_stage8.log"
echo STAGE8_BLOCKED_DONE
