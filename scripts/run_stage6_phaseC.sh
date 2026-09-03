#!/usr/bin/env bash
# Stage 6 Phase C orchestration (stops after verdict; no confirm / multistation).
set -euo pipefail
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate PS

LOGDIR=artifacts/results/stage6/logs
mkdir -p "$LOGDIR" artifacts/results/stage6/phaseC artifacts/cache/stage6/phaseC

echo "[PhaseC] confirm sealed?"
test -f artifacts/results/stage6/splits_full/CONFIRM_SEALED
test ! -f artifacts/results/stage6/method_lock_stage6.json

echo "[1] history (if missing)"
if [[ ! -f artifacts/models/stage6/history_picker_train/manifest.json ]]; then
  python scripts/build_stage6_picker_train_history.py 2>&1 | tee "$LOGDIR/phaseC_history.log"
fi

echo "[2] wait/merge ranker_train candidate caches"
for SRC in stead ida; do
  OUT=artifacts/cache/stage6/phaseC/ranker_train_${SRC}_top10
  if [[ ! -f "$OUT/${SRC}_top10.parquet" ]]; then
    echo "missing $OUT — run cache first"; exit 2
  fi
done

echo "[3] build dataset"
python scripts/build_stage6_ranker_dataset.py 2>&1 | tee "$LOGDIR/phaseC_dataset.log"

echo "[4] baselines"
python scripts/evaluate_stage6_candidate_ranker.py --baselines-only 2>&1 | tee "$LOGDIR/phaseC_baselines.log"

echo "[5] R1 seed42"
python scripts/train_stage6_candidate_ranker.py --variant R1 --seed 42 --beta 0.2 --gamma 1.0 --device cuda:0 \
  2>&1 | tee "$LOGDIR/phaseC_train_R1_s42.log"

echo "[6] R2 seed42 (beta/gamma grid small)"
BEST_R2=""
BEST_SCORE=-1
for BETA in 0.0 0.2; do
  for GAMMA in 0.5 1.0; do
    python scripts/train_stage6_candidate_ranker.py --variant R2 --seed 42 --beta $BETA --gamma $GAMMA --device cuda:0 \
      2>&1 | tee "$LOGDIR/phaseC_train_R2_s42_b${BETA}_g${GAMMA}.log"
  done
done

echo "[PhaseC] orchestration partial — continue with eval/ablation/verdict scripts after selecting best ckpt"
echo "STOP: do not auto-start multistation or confirm"
