#!/usr/bin/env bash
# Watch Phase C caches: merge STEAD when done, then launch IDA; then optionally build dataset.
set -euo pipefail
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate PS

STEAD_OUT=artifacts/cache/stage6/phaseC/ranker_train_stead_top10
IDA_OUT=artifacts/cache/stage6/phaseC/ranker_train_ida_top10
WORLD=4
GPUS=(0 4 6 7)
LOGDIR=artifacts/results/stage6/logs

echo "[watch] waiting for STEAD shards DONE"
while true; do
  n=0
  for r in 0 1 2 3; do
    [[ -f "$STEAD_OUT/shard_0${r}.DONE" ]] && n=$((n+1))
  done
  if [[ $n -eq 4 ]]; then
    break
  fi
  echo "[watch] stead done_shards=$n/4 $(date -Is)"
  sleep 120
done

if [[ ! -f "$STEAD_OUT/stead_top10.parquet" ]]; then
  python scripts/cache_stage6_phaseB_candidates.py --source stead --out-dir "$STEAD_OUT" --world-size "$WORLD" --merge-only
fi
echo "[watch] STEAD merged"

if [[ ! -f "$IDA_OUT/ida_top10.parquet" ]]; then
  mkdir -p "$IDA_OUT"
  PIDS=()
  for RANK in 0 1 2 3; do
    GPU=${GPUS[$RANK]}
    LOG=$LOGDIR/phaseC_cache_ida_rank${RANK}.log
    CUDA_VISIBLE_DEVICES=$GPU python scripts/cache_stage6_phaseB_candidates.py \
      --source ida \
      --manifest artifacts/results/stage6/phaseC/ranker_train_s_manifest.csv \
      --manifest-json artifacts/results/stage6/phaseC/ranker_train_s_manifest.json \
      --out-dir "$IDA_OUT" --rank $RANK --world-size $WORLD --device cuda:0 \
      --ida-ckpt artifacts/models/stage6/phasenet_ida_full_seed42/checkpoints/best.pt \
      >"$LOG" 2>&1 &
    PIDS+=($!)
  done
  printf '%s\n' "${PIDS[@]}" > "$IDA_OUT/LAUNCH.PIDS"
  echo "[watch] IDA launched ${PIDS[*]}"
  fail=0
  for pid in "${PIDS[@]}"; do
    wait "$pid" || fail=1
  done
  [[ $fail -eq 0 ]]
  python scripts/cache_stage6_phaseB_candidates.py --source ida --out-dir "$IDA_OUT" --world-size "$WORLD" --merge-only
fi
echo "[watch] IDA merged"

# wait history
while [[ ! -f artifacts/models/stage6/history_picker_train/manifest.json ]]; do
  echo "[watch] waiting history $(date -Is)"
  sleep 60
done

python scripts/build_stage6_ranker_dataset.py 2>&1 | tee "$LOGDIR/phaseC_dataset.log"
echo "[watch] dataset built — ready for train/eval"
echo "CACHES_AND_DATASET_DONE"
