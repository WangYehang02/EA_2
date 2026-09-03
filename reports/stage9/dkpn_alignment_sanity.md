# Stage 9 — DKPN Alignment / Diagnostic Smoke

**Role:** `diagnostic_only_possible_leakage`  
**alignment_gate:** `DIAGNOSTIC_ONLY_NO_FULL_EVAL`  
**enter_main_table:** false

## Checkpoint

- path: `/home/yehang/EARTHQUAKE/baseline/DKPN/models_v0412_paper_sb4/MEDIUM/DKPN_TrainDataset_INSTANCE_Size_MEDIUM_Rnd_50_Epochs_10_LR_0.0010_Batch_64.pt`
- sha256: `0d361e33a2b2cfa0decd017d5b64581149249cae27aa4bc5ff4f9cf864b2aa81`
- n_tensors: 111
- n_params: 269805
- loaded (not random init): `True`

## Why not main Table A

All official paper weights are trained on **INSTANCE** (`TrainDataset_INSTANCE` in filenames).
Cannot prove exclusion of Stage-6 confirm events → **no formal Ours−DKPN ranking / bootstrap**.

## Clean retrain (not started)

Would require Stage-6 picker_train event set, same labels/noise, confirm held out, then user approval.
