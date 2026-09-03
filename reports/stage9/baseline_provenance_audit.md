# Stage 9 — Baseline Provenance Audit (DKPN / SegPhase)

**Created (UTC):** 2026-08-17T23:38:11.472050+00:00  
**Project root:** `/home/yehang/yehang/Earthquake`  
**post_confirm_external_comparator_evaluation:** true  
**sota_claim_allowed:** false  
**Stage-6 method_lock unchanged:** `True`

## Paths / versions

| Repo | realpath | HEAD | Match expected? |
|--|--|--|--|
| SegPhase | `/home/yehang/EARTHQUAKE/baseline/SegPhase` | `27e7e5d9ce02fbc2aea5ec646a569a381a5e7b6e` | prefix `27e7e5d` |
| DKPN | `/home/yehang/EARTHQUAKE/baseline/DKPN` | `cbced5a58282ff6ad2703f9c9f41f8728e334bd0` | prefix `cbced5a` / v0.4.12 |

Third-party repos kept **read-only** (no pull/checkout/modify).

## SegPhase

- **Classification:** `A. valid_external_pretrained_no_INSTANCE_leakage`
- **Preferred checkpoint:** `model/model_100Hz.pth`  
  - SHA256: `ec5d48cdc575b9ffc638a9039669f08de6b2349acd5d9a36748e4c89606d0727`  
  - size: 1222285 (real binary, not LFS pointer)
- **Input:** `(B, 3, 3000)` @ 100 Hz; channels **UD, NS, EW**
- **INSTANCE map:** ENZ → **Z, N, E** (UD=Z, NS=N, EW=E)
- **Normalize:** per-channel z-score (`pred.py`)
- **Peaks:** `find_peaks(..., distance=1/sf, height=0.1)` — P=`pred[0]`, S=`pred[1]`
- **Training:** Japan network / JMA (not INSTANCE) → eligible for main Table A
- Other weights present: `model_250Hz.pth`, `model_M01.pth`, `model_V2_JP/best_model.pth` (Japan V2; not selected as primary)

## DKPN

- **Classification:** `C. possible_INSTANCE_confirm_leakage_diagnostic_only`
- **All official v0.4.12 paper weights** are named `*_TrainDataset_INSTANCE_*`
- Preferred diagnostic ckpt: `models_v0412_paper_sb4/MEDIUM/DKPN_TrainDataset_INSTANCE_Size_MEDIUM_Rnd_50_Epochs_10_LR_0.0010_Batch_64.pt`  
  - SHA256: `0d361e33a2b2cfa0decd017d5b64581149249cae27aa4bc5ff4f9cf864b2aa81`
- **Must NOT** enter valid same-protocol main table or formal Ours−DKPN bootstrap conclusions
- Smoke / diagnostic allowed; clean Stage-6-train retrain is **planned only**, not auto-started
- Official annotate defaults: S_threshold=0.2, component_order=ZNE, SeisBench-oriented API

## Environment plan (do not mutate `PS`)

| Env | Plan |
|--|--|
| `segphase_eval` | Lightweight: python3.11 + pytorch **CUDA** + numpy/scipy/obspy (avoid repo `env.yml` cpuonly pin) |
| `dkpn_eval` | Optional diagnostic: torch + seisbench compatible; full `dkpn_env.yml` (torch 1.11 / sb 0.4) only if needed |

## Next

1. SegPhase smoke + alignment → if pass: dev threshold + lock + confirm (Table A)  
2. DKPN diagnostic smoke only  
3. No simultaneous full HDF5 jobs; SegPhase first for full eval
