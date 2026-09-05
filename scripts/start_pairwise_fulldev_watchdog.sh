#!/usr/bin/env bash
# Start scalar_pairwise full-dev GPU watchdog in background-friendly way.
# Does not kill other jobs, does not sudo, does not touch confirm.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p artifacts/results/pairwise_fulldev/logs

# Prefer project conda env if present
if [[ -x "/home/yehang/miniconda3/envs/PS/bin/python" ]]; then
  PY="/home/yehang/miniconda3/envs/PS/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY="$(command -v python3)"
else
  echo "python not found" >&2
  exit 1
fi

exec "$PY" scripts/watch_pairwise_fulldev_gpu.py \
  --min-free-mb "${MIN_FREE_MB:-4000}" \
  --max-util "${MAX_UTIL:-10}" \
  --max-used-mb "${MAX_USED_MB:-1500}" \
  --poll-seconds "${POLL_SECONDS:-60}" \
  "$@"
