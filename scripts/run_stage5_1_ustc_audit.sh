#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate PS
mkdir -p artifacts/results/stage5_1_ustc/figures paper/candidate_edits
python scripts/analyze_path_residual_repeatability.py
python scripts/evaluate_hierarchical_residual_val.py
python scripts/finalize_stage5_1_ustc.py
pytest -q tests/test_stage5_1_ustc.py tests/test_history_no_future.py tests/test_residual_history_no_future.py
echo "Stage 5.1 complete"
