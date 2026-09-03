#!/usr/bin/env bash
# Stage 10 pipeline controller: status | resume | audit | dkpn-smoke | dkpn-seed42
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PY:-/home/yehang/miniconda3/envs/PS/bin/python}"
STATE="$ROOT/artifacts/results/stage10/stage10_state.json"
LOGDIR="$ROOT/artifacts/results/stage10/logs"
CACHE="/data/mnt_data/yehang/PSdetec/Earthquake_stage10"
mkdir -p "$LOGDIR" "$CACHE"

cmd="${1:-status}"

free_gpus() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk -F',' '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($2+0 < 500) print $1}'
}

case "$cmd" in
  status)
    echo "=== stage10_state.json ==="
    if [[ -f "$STATE" ]]; then cat "$STATE"; else echo "(missing)"; fi
    echo "=== GPU ==="
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv 2>/dev/null || true
    echo "=== free GPUs (<500MiB) ==="
    free_gpus || true
    echo "=== DKPN jobs ==="
    ls -la "$CACHE/dkpn_clean" 2>/dev/null || true
    ls -la "$ROOT/artifacts/results/stage10/dkpn_clean" 2>/dev/null || true
    for m in TRAIN.RUNNING TRAIN.DONE TRAIN.FAILED TRAIN.PID; do
      find "$CACHE/dkpn_clean" -name "$m" 2>/dev/null || true
    done
    ;;
  audit)
    "$PY" scripts/audit_stage10A.py
    ;;
  dkpn-overfit)
    "$PY" scripts/train_stage10_dkpn_clean.py --mode overfit --epochs 40 --max-traces 64 --seed 42 --workers 0
    ;;
  dkpn-smoke)
    # pick first free GPU if any
    g=$(free_gpus | head -1 || true)
    if [[ -n "${g:-}" ]]; then
      echo "Using free GPU $g"
      "$PY" scripts/train_stage10_dkpn_clean.py --mode smoke --max-traces 128 --seed 42 --gpu "$g" --workers 2
      "$PY" scripts/train_stage10_dkpn_clean.py --mode throughput --max-traces 256 --seed 42 --gpu "$g" --workers 2 --batch-size 8
    else
      echo "No free GPU; running CPU smoke (slow CF)"
      "$PY" scripts/train_stage10_dkpn_clean.py --mode smoke --max-traces 32 --seed 42 --device cpu --workers 0 --batch-size 2
      "$PY" scripts/train_stage10_dkpn_clean.py --mode throughput --max-traces 16 --seed 42 --device cpu --workers 0 --batch-size 2
    fi
    ;;
  dkpn-seed42)
    g=$(free_gpus | head -1 || true)
    if [[ -z "${g:-}" ]]; then
      echo "No free GPU — refusing to launch seed42 (would contend). Write WAITING marker."
      mkdir -p "$CACHE/dkpn_clean/train_seed42"
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) waiting_for_free_gpu" > "$CACHE/dkpn_clean/train_seed42/WAITING_FOR_GPU"
      exit 2
    fi
    out="$CACHE/dkpn_clean/train_seed42"
    mkdir -p "$out" "$LOGDIR"
    log="$LOGDIR/dkpn_seed42_$(date +%Y%m%d_%H%M%S).log"
    nohup "$PY" scripts/train_stage10_dkpn_clean.py --mode train --seed 42 --epochs 30 \
      --batch-size 8 --workers 4 --gpu "$g" --out-dir "$out" \
      >"$log" 2>&1 &
    echo $! | tee "$out/TRAIN.PID"
    echo "$log" | tee "$out/TRAIN.LOGPATH"
    echo "Launched seed42 PID=$(cat "$out/TRAIN.PID") log=$log gpu=$g"
    ;;
  resume)
    if [[ -f "$CACHE/dkpn_clean/train_seed42/TRAIN.DONE" ]]; then
      echo "seed42 already DONE"
      exit 0
    fi
    if [[ -f "$CACHE/dkpn_clean/train_seed42/TRAIN.RUNNING" ]]; then
      echo "seed42 still RUNNING pid=$(cat "$CACHE/dkpn_clean/train_seed42/TRAIN.PID" 2>/dev/null || echo '?')"
      exit 0
    fi
    exec bash "$0" dkpn-seed42
    ;;
  wait-gpu-seed42)
    # poll until a free GPU appears; do not kill others
    while true; do
      g=$(free_gpus | head -1 || true)
      if [[ -n "${g:-}" ]]; then
        echo "GPU $g free — launching"
        exec bash "$0" dkpn-seed42
      fi
      echo "$(date -u +%H:%M:%S) waiting for free GPU..."
      sleep 120
    done
    ;;
  *)
    echo "Usage: $0 {status|audit|dkpn-overfit|dkpn-smoke|dkpn-seed42|resume|wait-gpu-seed42}"
    exit 1
    ;;
esac
