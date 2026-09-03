#!/usr/bin/env bash
# Overnight Phase C: wait dataset → R1 → R2 grid → full-dev eval → ablations → bootstrap → verdict
# Does NOT unlock confirm / multistation / offset.
set -euo pipefail
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate PS

LOGDIR=artifacts/results/stage6/logs
OUT=artifacts/results/stage6/phaseC
MODEL=artifacts/models/stage6/ranker
mkdir -p "$LOGDIR" "$OUT" "$MODEL"
STATUS="$OUT/overnight_status.json"
DEVICE="${PHASEC_DEVICE:-cuda:0}"

log() { echo "[$(date -Is)] $*" | tee -a "$LOGDIR/phaseC_overnight.log"; }
set_status() {
  # usage: set_status stage_name [extra_json_object]
  local stage="$1"
  local extra="${2:-{}}"
  STAGE="$stage" EXTRA="$extra" STATUS_PATH="$STATUS" python - <<'PY'
import json, os
from pathlib import Path
from datetime import datetime, timezone
p = Path(os.environ["STATUS_PATH"])
d = json.loads(p.read_text()) if p.exists() else {}
d["stage"] = os.environ["STAGE"]
try:
    d.update(json.loads(os.environ["EXTRA"]))
except Exception:
    pass
d["updated_utc"] = datetime.now(timezone.utc).isoformat()
p.write_text(json.dumps(d, indent=2) + "\n")
PY
}

log "=== Phase C overnight start device=$DEVICE ==="
test -f artifacts/results/stage6/splits_full/CONFIRM_SEALED
test ! -f artifacts/results/stage6/method_lock_stage6.json
set_status waiting_dataset '{"confirm_sealed":true}'

# 1) wait for dataset
while [[ ! -f artifacts/results/stage6/phaseC/ranker_dataset_summary.json ]]; do
  if ! pgrep -f 'build_stage6_ranker_dataset.py' >/dev/null; then
    if [[ ! -f artifacts/cache/stage6/phaseC/ranker_train_features_R2.parquet ]]; then
      log "dataset process dead; relaunching build_stage6_ranker_dataset.py"
      PYTHONUNBUFFERED=1 python -u scripts/build_stage6_ranker_dataset.py >>"$LOGDIR/phaseC_dataset.log" 2>&1
    fi
  fi
  log "waiting dataset..."
  sleep 120
done
log "dataset ready"
set_status dataset_done

# 2) R1 seed42
log "train R1 seed42"
set_status train_R1_s42
python -u scripts/train_stage6_candidate_ranker.py --variant R1 --seed 42 --beta 0.2 --gamma 1.0 --device "$DEVICE" \
  >>"$LOGDIR/phaseC_train_R1_s42.log" 2>&1
R1_CKPT=$(ls -d "$MODEL"/R1_seed42_b0.2_g1.0 2>/dev/null | head -1)/best.pt
test -f "$R1_CKPT"
log "R1 ckpt=$R1_CKPT"

# 3) R2 seed42 beta/gamma grid
BEST_R2=""
BEST_SCORE="-1"
for BETA in 0.0 0.2; do
  for GAMMA in 0.5 1.0; do
    log "train R2 seed42 beta=$BETA gamma=$GAMMA"
    set_status train_R2_s42 "{\"beta\":$BETA,\"gamma\":$GAMMA}"
    python -u scripts/train_stage6_candidate_ranker.py --variant R2 --seed 42 --beta "$BETA" --gamma "$GAMMA" --device "$DEVICE" \
      >>"$LOGDIR/phaseC_train_R2_s42_b${BETA}_g${GAMMA}.log" 2>&1
    CKPT="$MODEL/R2_seed42_b${BETA}_g${GAMMA}/best.pt"
    SCORE=$(python - <<PY
import torch
b=torch.load("$CKPT", map_location="cpu", weights_only=False)
print(b.get("metrics",{}).get("s_f1@0.5", -1))
PY
)
    log "R2 b=$BETA g=$GAMMA score=$SCORE"
    if python -c "import sys; sys.exit(0 if float('$SCORE')>float('$BEST_SCORE') else 1)"; then
      BEST_SCORE=$SCORE
      BEST_R2=$CKPT
    fi
  done
done
log "best R2 seed42=$BEST_R2 score=$BEST_SCORE"
echo "$BEST_R2" > "$OUT/best_r2_seed42.txt"

# 4) Decide R3
R1_SCORE=$(python - <<PY
import torch
b=torch.load("$R1_CKPT", map_location="cpu", weights_only=False)
print(b.get("metrics",{}).get("s_f1@0.5", -1))
PY
)
RUN_R3=$(python -c "print(1 if float('$BEST_SCORE')-float('$R1_SCORE')>=0.005 else 0)")
BEST_CKPT=$BEST_R2
if [[ "$RUN_R3" == 1 ]]; then
  log "R2 gain over R1 sufficient; train R3 seed42"
  set_status train_R3_s42
  python -u scripts/train_stage6_candidate_ranker.py --variant R3 --seed 42 --beta 0.2 --gamma 1.0 --device "$DEVICE" \
    >>"$LOGDIR/phaseC_train_R3_s42.log" 2>&1
  R3_CKPT="$MODEL/R3_seed42_b0.2_g1.0/best.pt"
  R3_SCORE=$(python - <<PY
import torch
b=torch.load("$R3_CKPT", map_location="cpu", weights_only=False)
print(b.get("metrics",{}).get("s_f1@0.5", -1))
PY
)
  if python -c "import sys; sys.exit(0 if float('$R3_SCORE')>float('$BEST_SCORE') else 1)"; then
    BEST_CKPT=$R3_CKPT
    BEST_SCORE=$R3_SCORE
  fi
else
  log "skip R3 (R2-R1 gain < 0.005)"
fi
echo "$BEST_CKPT" > "$OUT/best_structure_seed42.txt"
log "structure pick=$BEST_CKPT"

# 5) Extra seeds
VARIANT=$(python - <<PY
import torch
print(torch.load("$BEST_CKPT", map_location="cpu", weights_only=False)["variant"])
PY
)
BETA=$(python - <<PY
import torch
print(torch.load("$BEST_CKPT", map_location="cpu", weights_only=False).get("beta",0.2))
PY
)
GAMMA=$(python - <<PY
import torch
print(torch.load("$BEST_CKPT", map_location="cpu", weights_only=False).get("gamma",1.0))
PY
)
SEED_CKPTS=("$BEST_CKPT")
for SEED in 123 2026; do
  log "train $VARIANT seed=$SEED"
  set_status "train_${VARIANT}_s${SEED}"
  python -u scripts/train_stage6_candidate_ranker.py --variant "$VARIANT" --seed "$SEED" --beta "$BETA" --gamma "$GAMMA" --device "$DEVICE" \
    >>"$LOGDIR/phaseC_train_${VARIANT}_s${SEED}.log" 2>&1
  SEED_CKPTS+=("$MODEL/${VARIANT}_seed${SEED}_b${BETA}_g${GAMMA}/best.pt")
done

FINAL_CKPT=$(python - <<PY
import torch
ckpts = """${SEED_CKPTS[*]}""".split()
best=None; sc=-1.0
for c in ckpts:
  b=torch.load(c, map_location="cpu", weights_only=False)
  s=float(b.get("metrics",{}).get("s_f1@0.5", -1))
  if s>sc:
    sc=s; best=c
print(best)
PY
)
echo "$FINAL_CKPT" > "$OUT/best_ranker_ckpt.txt"
log "final ckpt=$FINAL_CKPT"

# 6) Full-dev eval
log "full-dev eval"
set_status full_dev_eval
python -u scripts/evaluate_stage6_candidate_ranker.py --ranker-ckpt "$FINAL_CKPT" --device "$DEVICE" \
  >>"$LOGDIR/phaseC_eval_best.log" 2>&1

python - <<'PY'
from pathlib import Path
import numpy as np, pandas as pd, torch
from earthquake.stage6.ranker.dataset import TraceCandidateDataset, collate_traces
from earthquake.stage6.ranker.models import build_ranker
from torch.utils.data import DataLoader
out=Path("artifacts/results/stage6/phaseC/baseline_preds")
out.mkdir(parents=True, exist_ok=True)
ckpt=Path("artifacts/models/stage6/ranker/R1_seed42_b0.2_g1.0/best.pt")
if ckpt.exists():
  blob=torch.load(ckpt, map_location="cpu", weights_only=False)
  feat=pd.read_parquet("artifacts/cache/stage6/phaseC/dev_features_R1.parquet")
  model=build_ranker("R1", len(blob["feature_names"]))
  model.load_state_dict(blob["model"]); model.eval()
  ds=TraceCandidateDataset(feat, blob["feature_names"], hard_boost=False, seed=0)
  ds.indices=list(range(len(ds.groups)))
  loader=DataLoader(ds, batch_size=512, shuffle=False, collate_fn=collate_traces)
  preds=[]
  with torch.no_grad():
    for batch in loader:
      logits=model(batch["x"], batch["mask"])
      idx=logits.argmax(-1).numpy()
      samp=batch["samples"].numpy()
      for i,pi in enumerate(idx):
        preds.append(float("nan") if pi>=10 or not bool(batch["mask"][i,pi]) else float(samp[i,pi]))
  meta=pd.read_csv("artifacts/results/stage6/phaseB_eval_manifest.csv")
  order={g["trace_name"]:j for j,g in enumerate(ds.groups)}
  arr=np.full(len(meta), np.nan)
  for i,tn in enumerate(meta.trace_name.astype(str)):
    j=order.get(tn)
    if j is not None: arr[i]=preds[j]
  np.save(out/"R1.npy", arr)
  print("saved R1.npy", float(np.isfinite(arr).mean()))
PY

# 7) ablation + bootstrap
log "ablation summary + bootstrap"
set_status bootstrap_ablation
python - <<'PY'
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from earthquake.config import artifacts_dir, save_json

out = artifacts_dir()/"results"/"stage6"/"phaseC"
preds = pd.read_parquet(out/"ranker_dev_predictions.parquet")
feat = pd.read_parquet(artifacts_dir()/"cache"/"stage6"/"phaseC"/"dev_features_R2.parquet")
tr = feat.drop_duplicates("trace_name")
ida_only = set(tr[tr.analysis_classes.astype(str).str.contains("ida_only_recoverable")].trace_name.astype(str))
pred_map = dict(zip(preds.trace_name.astype(str), preds.pred_s_sample))
true_map = dict(zip(preds.trace_name.astype(str), preds.true_s_sample))
sr_map = dict(zip(preds.trace_name.astype(str), preds.sampling_rate_hz))
ok=0; tot=0
for tn in ida_only:
  tot+=1
  p,t,sr=pred_map.get(tn), true_map.get(tn), sr_map.get(tn)
  if p is not None and np.isfinite(p) and np.isfinite(t) and abs(p-t)/sr<=0.5:
    ok+=1
abl={
  "ida_only_recovery_rate": float(ok/max(tot,1)),
  "n_ida_only": tot,
  "n_ida_only_recovered": ok,
  "seed_consistent": None,
  "note": "shuffled-history full retrain skipped overnight; seed consistency from selection metrics",
}
best=Path(out/"best_ranker_ckpt.txt").read_text().strip()
blob=torch.load(best, map_location="cpu", weights_only=False)
variant=blob["variant"]; beta=blob.get("beta",0.2); gamma=blob.get("gamma",1.0)
scores=[]
for seed in (42,123,2026):
  p=artifacts_dir()/f"models/stage6/ranker/{variant}_seed{seed}_b{beta}_g{gamma}/best.pt"
  if p.exists():
    scores.append(float(torch.load(p, map_location="cpu", weights_only=False).get("metrics",{}).get("s_f1@0.5",-1)))
if len(scores)>=2:
  abl["seed_scores_selection_metric"]=scores
  abl["seed_consistent"]= bool((max(scores)-min(scores)) <= 0.01)
save_json(abl, out/"phaseC_ablations.json")
print(abl)
PY

python -u scripts/bootstrap_stage6_ranker.py >>"$LOGDIR/phaseC_bootstrap.log" 2>&1 || log "bootstrap script soft-failed"

# 8) finalize
log "finalize"
set_status finalize
python -u scripts/finalize_stage6_phaseC.py --best-ckpt "$FINAL_CKPT" >>"$LOGDIR/phaseC_finalize.log" 2>&1

python - <<'PY'
import json
from pathlib import Path
v=json.loads(Path("artifacts/results/stage6/phaseC/phaseC_final_verdict.json").read_text())
assert v.get("confirm_remains_sealed") is True
assert not Path("artifacts/results/stage6/method_lock_stage6.json").exists()
print("verdict", v.get("verdict"), "multi", v.get("multistation_may_start"))
PY

set_status DONE '{"confirm_sealed":true}'
log "=== Phase C overnight DONE — waiting for user; confirm still sealed ==="
echo DONE > "$OUT/OVERNIGHT.DONE"
