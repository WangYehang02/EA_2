#!/usr/bin/env bash
# Stage 6 pipeline orchestrator. Supports: audit|splits|stead|smoke|train-ida|status
set -euo pipefail
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate PS
CMD=${1:-help}
case "$CMD" in
  audit)
    cat reports/stage6/stage6_kickoff_audit.md
    ;;
  splits)
    python scripts/build_stage6_splits.py
    python -c "from earthquake.stage6.splits import write_confirm_seal; write_confirm_seal()"
    ;;
  stead)
    python scripts/evaluate_stage6_stead_baseline.py --subset stage6_dev --device "${DEVICE:-cuda:1}"
    ;;
  smoke)
    python scripts/train_stage6_phasenet.py --config configs/stage6/phasenet_indomain_smoke.yaml
    ;;
  train-ida)
    python scripts/train_stage6_phasenet.py --config configs/stage6/phasenet_indomain.yaml
    ;;
  status)
    ls -la artifacts/results/stage6/baseline_stead 2>/dev/null || true
    ls -la artifacts/models/stage6 2>/dev/null || true
    tail -n 20 artifacts/results/stage6/logs/* 2>/dev/null || true
    ;;
  help|*)
    echo "usage: $0 {audit|splits|stead|smoke|train-ida|status}"
    ;;
esac
