#!/usr/bin/env bash
# Stage 9 SegPhase: window select on 8k subset → full dev cache → analyze/lock → confirm
set -euo pipefail
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
PY="${PY:-artifacts/venvs/segphase_eval/bin/python}"
LOG=artifacts/results/stage9/logs
mkdir -p "$LOG" artifacts/results/stage9/cache

mapfile -t FREE < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F',' '$2+0 < 500 {gsub(/ /,"",$1); print $1}')
echo "FREE=${FREE[*]}" | tee "$LOG/gpu_free.txt"
N=${#FREE[@]}
if [[ $N -lt 1 ]]; then echo no free GPU; exit 1; fi

# Build window-select manifest: sha256 sort first 8000 of stage6?dev
$PY - <<'PY'
import hashlib, pandas as pd
from pathlib import Path
man=pd.read_csv('artifacts/results/stage6/phaseB_eval_manifest.csv')
man['_h']=man.trace_name.astype(str).map(lambda t: hashlib.sha256(t.encode()).hexdigest())
sub=man.sort_values('_h').head(8000).drop(columns=['_h'])
Path('artifacts/results/stage9').mkdir(parents=True, exist_ok=True)
sub.to_csv('artifacts/results/stage9/window_select_8k_manifest.csv', index=False)
print(len(sub), sub.event_id.nunique())
PY

launch () {
  local SCHEME=$1 MAN=$2 OUT=$3 TAG=$4
  mkdir -p "$OUT"
  if [[ -f "$OUT/MERGE.DONE" ]]; then echo "skip $TAG"; return; fi
  local PIDS=() RANK=0
  for GPU in "${FREE[@]}"; do
    CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/stage9_cache_segphase_peaks.py \
      --manifest "$MAN" --out-dir "$OUT" --window-scheme "$SCHEME" \
      --rank $RANK --world-size $N --device cuda:0 \
      >"$LOG/cache_${TAG}_r${RANK}.log" 2>&1 &
    PIDS+=($!)
    echo "launch $TAG rank=$RANK gpu=$GPU pid=${PIDS[-1]}"
    RANK=$((RANK+1))
  done
  local fail=0
  for pid in "${PIDS[@]}"; do wait "$pid" || fail=1; done
  [[ $fail -eq 0 ]] || { echo FAIL "$TAG"; exit 1; }
  $PY -u scripts/stage9_cache_segphase_peaks.py --manifest "$MAN" --out-dir "$OUT" --window-scheme "$SCHEME" --world-size $N --merge-only
}

# 1) window select A and B on 8k (sequential to avoid HDF5 thrash — A then B)
launch A artifacts/results/stage9/window_select_8k_manifest.csv artifacts/results/stage9/cache/winA_8k winA_8k
launch B artifacts/results/stage9/window_select_8k_manifest.csv artifacts/results/stage9/cache/winB_8k winB_8k

$PY -u scripts/stage9_select_window_and_threshold.py | tee "$LOG/select_lock.log"

# 2) full?dev + confirm with locked scheme (from lock?file)
SCHEME=$(python -c "import json;print(json.load(open('artifacts/results/stage9/segphase_method_lock.json'))['window_scheme'])")
echo "LOCKED_SCHEME=$SCHEME"

launch "$SCHEME" artifacts/results/stage6/phaseB_eval_manifest.csv "artifacts/results/stage9/cache/dev_${SCHEME}" "dev_${SCHEME}"
launch "$SCHEME" artifacts/results/stage6/final_confirm/confirm_s_eval_manifest.csv "artifacts/results/stage9/cache/confirm_${SCHEME}" "confirm_${SCHEME}"

$PY -u scripts/stage9_finalize_segphase.py | tee "$LOG/finalize.log"
echo STAGE9_SEGP_DONE
