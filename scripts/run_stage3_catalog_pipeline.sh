#!/usr/bin/env bash
# Wait for catalog dataset build, then train 3 seeds + eval + bootstrap + analyze.
set -euo pipefail
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate PS
export INSTANCE_ROOT="${INSTANCE_ROOT:-/mnt/yehang/PSdetec/INSTANCE}"

LOG=artifacts/results/stage3/catalog_pipeline.log
echo "[$(date -Is)] waiting for build_gate_dataset (catalog) to finish..." | tee -a "$LOG"

while pgrep -f "python -u scripts/build_gate_dataset.py --config configs/gate_catalog.yaml" >/dev/null; do
  sleep 120
  echo "[$(date -Is)] still building..." | tee -a "$LOG"
  tail -c 120 artifacts/results/stage3/build_gate_catalog.log | tr '\r' '\n' | tail -n 1 | tee -a "$LOG" || true
done

if ! grep -q "event_disjoint" artifacts/results/stage3/build_gate_catalog.log; then
  echo "[$(date -Is)] build log missing success marker; abort" | tee -a "$LOG"
  exit 1
fi

echo "[$(date -Is)] build done; oracle on full test" | tee -a "$LOG"
python -u scripts/compute_candidate_oracle.py --config configs/gate_catalog.yaml 2>&1 | tee -a "$LOG"

for seed in 42 123 2026; do
  echo "[$(date -Is)] train scalar_gate seed=${seed}" | tee -a "$LOG"
  python -u scripts/train_learned_gate.py --config configs/gate_catalog.yaml --seed "${seed}" 2>&1 | tee -a "$LOG"
done

echo "[$(date -Is)] train candidate_ranker seed=42 (aux)" | tee -a "$LOG"
python -u scripts/train_learned_gate.py --config configs/candidate_ranker.yaml --seed 42 --model-type candidate_ranker 2>&1 | tee -a "$LOG" || true

echo "[$(date -Is)] evaluate all seeds" | tee -a "$LOG"
python -u scripts/evaluate_learned_gate.py --config configs/gate_catalog.yaml --all-seeds 2>&1 | tee -a "$LOG"

echo "[$(date -Is)] bootstrap" | tee -a "$LOG"
python -u scripts/bootstrap_gate.py --config configs/gate_catalog.yaml --n-bootstrap 2000 2>&1 | tee -a "$LOG"

echo "[$(date -Is)] analyze + plot" | tee -a "$LOG"
python -u scripts/analyze_gate_behavior.py --config configs/gate_catalog.yaml 2>&1 | tee -a "$LOG"
python -u scripts/plot_gate_cases.py --config configs/gate_catalog.yaml 2>&1 | tee -a "$LOG"

echo "[$(date -Is)] pytest" | tee -a "$LOG"
pytest -q 2>&1 | tee -a "$LOG"

echo "[$(date -Is)] catalog pipeline complete" | tee -a "$LOG"
