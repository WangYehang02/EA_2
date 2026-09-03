#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate PS
for f in \
  artifacts/results/stage3/learned_gate_picks_test.parquet \
  artifacts/results/stage3/gate_cases/pn_wrong_gate_fix.png
do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing frozen artifact: $f" >&2
    exit 1
  fi
done
python scripts/build_paper_assets.py
test -f paper/figures/fig_error_cdf_stage3.pdf
test -f paper/figures/fig_method_flowchart.pdf
test -f paper/figures/fig_case_success_wrongpeak_fix.pdf
echo "OK: paper figures rebuilt from frozen artifacts"
