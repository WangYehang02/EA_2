#!/usr/bin/env bash
# Stage 6 Finalization orchestrator: C.2 → lock → preconfirm → authorize → confirm
set -euo pipefail
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate PS

LOG=artifacts/results/stage6/logs
mkdir -p "$LOG" artifacts/results/stage6/phaseC2 artifacts/results/stage6/final_confirm

echo "[$(date -Is)] Phase C.2 closeout"
python -u scripts/run_stage6_phaseC2_closeout.py | tee "$LOG/phaseC2_closeout.log"
test -f artifacts/results/stage6/phaseC2/PHASEC2.DONE

echo "[$(date -Is)] pytest pre-lock"
conda run -n PS pytest -q | tee "$LOG/finalization_pytest1.log"
# mark
echo OK > artifacts/results/stage6/final_confirm/PYTEST_PRECONFIRM.OK

echo "[$(date -Is)] method lock"
python -u scripts/lock_stage6_final_method.py | tee "$LOG/method_lock.log"
test -f artifacts/results/stage6/final_confirm/method_lock.sha256

echo "[$(date -Is)] confirm protocol"
python -u scripts/write_stage6_confirm_protocol.py

echo "[$(date -Is)] preconfirm audit"
python -u scripts/audit_stage6_preconfirm.py | tee "$LOG/preconfirm_audit.log"

echo "[$(date -Is)] pytest pre-authorize"
conda run -n PS pytest -q | tee "$LOG/finalization_pytest2.log"

echo "[$(date -Is)] authorize confirm"
python -u scripts/authorize_stage6_confirm.py

echo "[$(date -Is)] launch confirm (background)"
nohup python -u scripts/run_stage6_final_confirm.py > "$LOG/confirm_final_nohup.log" 2>&1 &
echo "CONFIRM_PID=$!"
echo "$!" > artifacts/results/stage6/final_confirm/CONFIRM.PID
echo "[$(date -Is)] confirm launched; metrics hidden until CONSUMED"
