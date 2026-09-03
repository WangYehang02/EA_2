#!/usr/bin/env bash
# Stage-4 confirmatory pipeline (idempotent / resumable).
set -euo pipefail
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate PS
export INSTANCE_ROOT="${INSTANCE_ROOT:-/mnt/yehang/PSdetec/INSTANCE}"
mkdir -p artifacts/results/stage4/logs artifacts/results/stage4/tables artifacts/results/stage4/figures

log() { echo "[$(date -Is)] $*" | tee -a artifacts/results/stage4/logs/pipeline.log; }

run_step() {
  local name="$1"; shift
  local marker="artifacts/results/stage4/logs/${name}.done"
  if [[ -f "$marker" ]]; then
    log "skip $name (done)"
    return 0
  fi
  log "start $name"
  if "$@"; then
    date -Is > "$marker"
    log "done $name"
  else
    log "FAIL $name"
    rm -f "artifacts/results/stage4/stage4_final_verdict.json" 2>/dev/null || true
    exit 1
  fi
}

run_step lock python -u scripts/lock_stage4_method.py --config configs/stage4/confirmatory.yaml
run_step holdout python -u scripts/build_stage4_holdout.py --config configs/stage4/confirmatory.yaml
run_step eval_stead python -u scripts/evaluate_stage4_holdout.py --config configs/stage4/confirmatory.yaml --weight stead
# transfer weights (same frozen rescore)
run_step eval_ethz python -u scripts/evaluate_stage4_holdout.py --config configs/stage4/confirmatory.yaml --weight ethz
run_step eval_scedc python -u scripts/evaluate_stage4_holdout.py --config configs/stage4/confirmatory.yaml --weight scedc || true
# merge transfer table
python - <<'PY'
from pathlib import Path
import pandas as pd
root = Path('artifacts/results/stage4/tables')
frames = []
for w in ['stead','ethz','scedc']:
    p = root / f'main_results_{w}.csv'
    if p.exists():
        frames.append(pd.read_csv(p))
if frames:
    pd.concat(frames, ignore_index=True).to_csv(root / 'transfer_weights_results.csv', index=False)
    # ensure main_results.csv is stead
    s = root / 'main_results_stead.csv'
    if s.exists():
        pd.read_csv(s).to_csv(root / 'main_results.csv', index=False)
        abl_methods = {
            'phasenet','distance_only','raw_path_history','mlp_residual_unshrunk','mlp_residual_shrink50',
            'fixed_catalog_rescore','learned_gate','shuffled_history','biased_history','history_unavailable',
            'oracle_selector','oracle_candidate'
        }
        df = pd.read_csv(s)
        df[df.method.isin(abl_methods)].to_csv(root / 'ablation_results.csv', index=False)
print('transfer merge ok')
PY

# copy stead picks to canonical name if needed
if [[ -f artifacts/results/stage4/holdout_picks_stead.parquet && ! -f artifacts/results/stage4/holdout_picks.parquet ]]; then
  cp artifacts/results/stage4/holdout_picks_stead.parquet artifacts/results/stage4/holdout_picks.parquet
fi
# evaluate_stage4 without --weight writes holdout_picks.parquet; with --weight writes holdout_picks_stead.parquet
if [[ -f artifacts/results/stage4/holdout_picks_stead.parquet ]]; then
  cp -f artifacts/results/stage4/holdout_picks_stead.parquet artifacts/results/stage4/holdout_picks.parquet
fi

run_step bootstrap python -u scripts/bootstrap_stage4.py --config configs/stage4/confirmatory.yaml
run_step subsets python -u scripts/analyze_stage4_subsets.py
run_step compute python -u scripts/measure_stage4_compute.py
run_step figures python -u scripts/plot_stage4_figures.py
run_step report python -u scripts/write_stage4_report.py
run_step pytest conda run -n PS pytest -q
log "Stage-4 confirmatory pipeline complete"
