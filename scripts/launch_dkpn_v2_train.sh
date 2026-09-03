#!/usr/bin/env bash
# Launch DKPN v2: never 1-GPU full train. MIN_GPUS=4. Never kill other jobs.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PY:-/home/yehang/miniconda3/envs/PS/bin/python}"
OUT="/data/mnt_data/yehang/PSdetec/Earthquake_stage10/dkpn_clean_v2_seed42"
LOGDIR="$ROOT/artifacts/results/stage10/logs"
mkdir -p "$OUT" "$LOGDIR" "$ROOT/artifacts/results/stage10"
MODE="${1:-status}"

"$PY" - "$MODE" "$OUT" "$ROOT" <<'PY'
import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[3]) / "src"))
from earthquake.stage10.gpu_policy import (
    MIN_GPUS, PREFERRED_GPUS, THROUGHPUT_GPUS, THROUGHPUT_STEPS, idle_gpu_indices, launch_block_reason,
    snapshot_gpus, status_dict, training_process_running,
)
mode, out, root = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
st = status_dict(out_dir=out)
(root / "artifacts/results/stage10/dkpn_v2_launch_policy.json").write_text(json.dumps({
    "MIN_GPUS": MIN_GPUS,
    "PREFERRED_GPUS": PREFERRED_GPUS,
    "THROUGHPUT_GPUS": THROUGHPUT_GPUS,
    "THROUGHPUT_STEPS": THROUGHPUT_STEPS,
    "mem_used_max_mib": 500,
    "util_near_zero": True,
    "no_other_compute_procs": True,
    "never_preempt": True,
    "one_gpu_full_train_forbidden": True,
    "status": st,
}, indent=2) + "\n")
print(json.dumps(st, indent=2))
if mode == "status":
    sys.exit(0)
idle = idle_gpu_indices()
already = bool(training_process_running()) or (out / "TRAIN.RUNNING").is_file()
reason = launch_block_reason(mode=mode, n_idle=len(idle), already_training=already, train_running_flag=(out / "TRAIN.RUNNING").is_file())
if mode == "throughput" and reason == "need_min_gpus_1_have_0":
    sys.exit(3)
if reason:
    print("REFUSE", reason, file=sys.stderr)
    sys.exit(4 if reason == "training_already_running" else 3)
n_take = THROUGHPUT_GPUS if mode == "throughput" else PREFERRED_GPUS
open(out / "SELECTED_GPUS.txt", "w").write(",".join(str(i) for i in idle[:n_take]) + "\n")
sys.exit(0)
PY
rc=$?
if [[ "$MODE" == "status" ]]; then exit 0; fi
if [[ $rc -ne 0 ]]; then
  echo waiting > "$OUT/WAITING_FOR_GPU"
  exit $rc
fi

if [[ -f "$OUT/TRAIN.DONE" && "$MODE" == "train" ]]; then echo already DONE; exit 0; fi
if [[ -f "$ROOT/artifacts/results/stage10/dkpn_v2/TRAIN_BLOCKED" ]]; then echo BLOCKED; exit 2; fi
for f in overfit32.json mix128.json smoke1000.json; do
  ok=$($PY -c "import json;print(json.load(open('$ROOT/artifacts/results/stage10/dkpn_v2/$f')).get('ok',False))" 2>/dev/null || echo False)
  if [[ "$ok" != "True" ]]; then echo "gate fail $f"; exit 2; fi
done

gpus=$(cat "$OUT/SELECTED_GPUS.txt")
if [[ "$MODE" == "throughput" ]]; then
  gpus="${gpus%%,*}"
fi
export CUDA_VISIBLE_DEVICES="$gpus"
rm -f "$OUT/WAITING_FOR_GPU"
log="$LOGDIR/dkpn_v2_${MODE}_$(date +%Y%m%d_%H%M%S).log"
extra=()
if [[ "$MODE" == "throughput" ]]; then
  extra=(--gpu "$gpus")
fi
nohup "$PY" -u scripts/train_stage10_dkpn_v2.py --mode "$MODE" --workers 4 "${extra[@]}" \
  >"$log" 2>&1 &
echo $! | tee "$OUT/LAUNCH.PID"
echo "$log" | tee "$OUT/TRAIN.LOGPATH"
echo "launched pid=$(cat $OUT/LAUNCH.PID) gpus=$gpus mode=$MODE log=$log"
