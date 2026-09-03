#!/usr/bin/env bash
# Stage-2 continuation: wait for PhaseNet audit cache, then evaluate + bootstrap + hard cases.
set -euo pipefail
source /home/yehang/miniconda3/etc/profile.d/conda.sh
conda activate PS
cd /home/yehang/yehang/Earthquake
export INSTANCE_ROOT=/mnt/yehang/PSdetec/INSTANCE
export PYTHONUNBUFFERED=1

LOG=/tmp/stage2_pipeline.log
exec > >(tee -a "$LOG") 2>&1

echo "[$(date)] waiting for audit process / cache"
CACHE=artifacts/results/stage2/phasenet_fixed_cache.parquet
CANDS=artifacts/results/stage2/phasenet_fixed_cache_candidates.parquet
AUDIT_JSON=artifacts/results/stage2/metric_audit.json

# If audit still running, wait; if not started and cache missing, start it.
if ! pgrep -f 'python -u scripts/audit_pick_metrics.py' >/dev/null; then
  if [[ ! -f "$AUDIT_JSON" ]]; then
    echo "[$(date)] starting audit"
    python -u scripts/audit_pick_metrics.py --config configs/fusion_fixed.yaml --device cuda --flush-every 200
  fi
else
  echo "[$(date)] audit already running; waiting"
  while pgrep -f 'python -u scripts/audit_pick_metrics.py' >/dev/null; do
    sleep 30
  done
  # If process exited without writing audit json, re-run (will resume from cache)
  if [[ ! -f "$AUDIT_JSON" ]]; then
    echo "[$(date)] audit process ended without metric_audit.json; resume/finalize"
    python -u scripts/audit_pick_metrics.py --config configs/fusion_fixed.yaml --device cuda --flush-every 200
  fi
fi

echo "[$(date)] cache/audit ready; n=$(python -c "import pandas as pd; print(len(pd.read_parquet('$CACHE')))")"

# Ensure travel-time + residual exist
[[ -f artifacts/results/stage2/travel_time_baseline.pkl ]] || python scripts/fit_travel_time_baseline.py --config configs/fusion_fixed.yaml
[[ -f artifacts/results/stage2/residual_history_meta.json ]] || python scripts/build_residual_history.py --config configs/fusion_fixed.yaml --protocol frozen

echo "[$(date)] evaluate candidate rescoring"
python -u scripts/evaluate_candidate_rescoring.py --config configs/fusion_fixed.yaml

echo "[$(date)] bootstrap significance"
python -u scripts/bootstrap_significance.py --config configs/fusion_fixed.yaml --n-bootstrap 2000

echo "[$(date)] hard cases"
python -u scripts/analyze_hard_cases.py --config configs/fusion_fixed.yaml

echo "[$(date)] pytest"
pytest -q

echo "[$(date)] STAGE2_PIPELINE_COMPLETE"
python - <<'PY'
import json
from pathlib import Path
root=Path('artifacts/results/stage2')
for name in ['metric_audit.json','travel_time_baseline_selection.json','residual_history_meta.json','best_lambdas.json','bootstrap_significance.json','hard_cases_summary.json']:
    p=root/name
    print(name, 'OK' if p.exists() else 'MISSING')
if (root/'candidate_rescoring_comparison.csv').exists():
    import pandas as pd
    df=pd.read_csv(root/'candidate_rescoring_comparison.csv')
    print(df.to_string(index=False))
PY
