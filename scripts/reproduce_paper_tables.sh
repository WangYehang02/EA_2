#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate PS
# Fail clearly if frozen Stage-3 artifacts missing
for f in \
  artifacts/results/stage3/learned_gate_eval_test.json \
  artifacts/results/stage3/learned_gate_picks_test.parquet \
  artifacts/results/stage3/bootstrap_fixed_vs_phasenet.json \
  artifacts/results/stage4/tables/main_results.csv
do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing frozen artifact: $f" >&2
    exit 1
  fi
done
python scripts/build_paper_assets.py
test -f paper/tables/table_main_stage3.csv
test -f artifacts/results/stage3/bootstrap_fixed_vs_phasenet.json
echo "OK: paper tables rebuilt from frozen artifacts (no Stage 1–4 overwrite)"
