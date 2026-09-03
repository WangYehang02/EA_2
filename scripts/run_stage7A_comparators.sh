#!/usr/bin/env bash
# Stage 7A: cache PhaseNet ethz/scedc peaks on free GPUs, then analyze.
set -euo pipefail
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate PS

LOG=artifacts/results/stage7/logs
mkdir -p "$LOG" artifacts/results/stage7/cache

# free GPUs only (avoid busy 3-6)
mapfile -t FREE < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F',' '$2+0 < 500 {gsub(/ /,"",$1); print $1}')
echo "FREE_GPUS=${FREE[*]}"
N=${#FREE[@]}
if [[ $N -lt 1 ]]; then echo "no free GPUs"; exit 1; fi

python -u scripts/stage7_write_comparator_registry.py | tee "$LOG/registry.log"

launch_cache () {
  local WEIGHT=$1
  local MANIFEST=$2
  local OUT=$3
  local TAG=$4
  mkdir -p "$OUT"
  if [[ -f "$OUT/MERGE.DONE" ]]; then
    echo "skip $TAG already merged"
    return
  fi
  local PIDS=()
  local RANK=0
  for GPU in "${FREE[@]}"; do
    local L="$LOG/cache_${TAG}_rank${RANK}.log"
    CUDA_VISIBLE_DEVICES=$GPU python -u scripts/stage7_cache_phasenet_peaks.py \
      --weight "$WEIGHT" --manifest "$MANIFEST" --out-dir "$OUT" \
      --rank $RANK --world-size $N --device cuda:0 >"$L" 2>&1 &
    PIDS+=($!)
    echo "launch $TAG rank=$RANK gpu=$GPU pid=${PIDS[-1]}"
    RANK=$((RANK+1))
  done
  local fail=0
  for pid in "${PIDS[@]}"; do
    wait "$pid" || fail=1
  done
  if [[ $fail -ne 0 ]]; then echo "FAIL $TAG"; exit 1; fi
  python -u scripts/stage7_cache_phasenet_peaks.py --weight "$WEIGHT" --manifest "$MANIFEST" --out-dir "$OUT" --world-size $N --merge-only
}

DEV_MAN=artifacts/results/stage6/phaseB_eval_manifest.csv
CONF_MAN=artifacts/results/stage6/final_confirm/confirm_s_eval_manifest.csv

launch_cache ethz "$DEV_MAN" artifacts/results/stage7/cache/dev_ethz dev_ethz
launch_cache scedc "$DEV_MAN" artifacts/results/stage7/cache/dev_scedc dev_scedc
launch_cache ethz "$CONF_MAN" artifacts/results/stage7/cache/confirm_ethz confirm_ethz
launch_cache scedc "$CONF_MAN" artifacts/results/stage7/cache/confirm_scedc confirm_scedc

python -u scripts/stage7_analyze_comparators.py | tee "$LOG/analyze.log"
echo CACHE_AND_ANALYZE_DONE
