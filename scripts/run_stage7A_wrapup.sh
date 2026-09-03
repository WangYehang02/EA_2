#!/usr/bin/env bash
# After comparator caches+analyze finish, run runtime bench on free GPUs, finalize, pytest.
set -euo pipefail
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate PS
LOG=artifacts/results/stage7/logs
mkdir -p "$LOG"

# wait for analyze markers
for i in $(seq 1 600); do
  if [[ -f artifacts/results/stage7/stage7A_final_verdict.json ]]; then
    echo "analyze present"
    break
  fi
  echo "waiting analyze... $i"
  sleep 60
done
[[ -f artifacts/results/stage7/stage7A_final_verdict.json ]] || { echo "analyze timeout"; exit 1; }

# wait until at least 4 free GPUs (cache jobs done)
for i in $(seq 1 120); do
  mapfile -t FREE < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F',' '$2+0 < 500 {gsub(/ /,"",$1); print $1}')
  echo "free=${#FREE[@]} -> ${FREE[*]}"
  if [[ ${#FREE[@]} -ge 4 ]]; then break; fi
  sleep 30
done

# Runtime: 1 and 4 GPU (8 if free). Use moderate timed count for wall time.
python -u scripts/stage7_runtime_benchmark.py --weight ethz --n-bench 400 --warmup 200 --reps 3 \
  2>&1 | tee "$LOG/runtime.log"

python -u scripts/stage7_finalize_reports.py 2>&1 | tee "$LOG/finalize.log"

conda run -n PS pytest -q 2>&1 | tee "$LOG/pytest.log"
echo WRAPUP_DONE
