#!/usr/bin/env bash
# Launch Phase B multi-GPU candidate cache (resume-safe).
set -euo pipefail
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate PS

SOURCE="${1:?source stead|ida}"
MANIFEST="${2:-artifacts/results/stage6/phaseB_eval_manifest.csv}"
MANIFEST_JSON="${3:-artifacts/results/stage6/phaseB_eval_manifest.json}"
NOISE_FLAG="${4:-}"
WORLD=8
LOGDIR="artifacts/results/stage6/logs"
mkdir -p "$LOGDIR" "artifacts/cache/stage6/phaseB"

if [[ "$SOURCE" == "ida" ]]; then
  CKPT="artifacts/models/stage6/phasenet_ida_full_seed42/checkpoints/best.pt"
  if [[ "$(basename "$CKPT")" != "best.pt" ]]; then
    echo "refuse non-best ckpt"; exit 2
  fi
fi

OUT_BASE="artifacts/cache/stage6/phaseB"
if [[ "$NOISE_FLAG" == "--noise" ]]; then
  OUT="$OUT_BASE/noise_${SOURCE}_top10"
else
  OUT="$OUT_BASE/${SOURCE}_top10"
fi
mkdir -p "$OUT"

PIDS=()
for RANK in $(seq 0 $((WORLD-1))); do
  LOG="$LOGDIR/phaseB_cache_${SOURCE}${NOISE_FLAG:+_noise}_rank${RANK}.log"
  CMD=(python scripts/cache_stage6_phaseB_candidates.py
    --source "$SOURCE"
    --manifest "$MANIFEST"
    --manifest-json "$MANIFEST_JSON"
    --out-dir "$OUT"
    --rank "$RANK"
    --world-size "$WORLD"
    --device "cuda:0"
  )
  if [[ "$SOURCE" == "ida" ]]; then
    CMD+=(--ida-ckpt artifacts/models/stage6/phasenet_ida_full_seed42/checkpoints/best.pt)
  fi
  if [[ "$NOISE_FLAG" == "--noise" ]]; then
    CMD+=(--noise)
  fi
  echo "launch rank=$RANK log=$LOG"
  CUDA_VISIBLE_DEVICES=$RANK "${CMD[@]}" >"$LOG" 2>&1 &
  PIDS+=($!)
done

echo "PIDS=${PIDS[*]}"
echo "$SOURCE" > "$OUT/LAUNCH.PIDS"
printf '%s\n' "${PIDS[@]}" >> "$OUT/LAUNCH.PIDS"

fail=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    echo "shard failed pid=$pid"
    fail=1
  fi
done
if [[ $fail -ne 0 ]]; then
  exit 1
fi

python scripts/cache_stage6_phaseB_candidates.py --source "$SOURCE" --out-dir "$OUT" --world-size "$WORLD" --merge-only ${NOISE_FLAG}
echo "DONE $OUT"
