# Stage 8 — Final Report (LFTNet)

**Final verdict:** `implementation_not_reproducible`  
**sota_claim_allowed:** `false`  
**Replace Stage-6 main method:** `false`  
**Post-confirm comparator executed:** `false`

## Answers

1. **Official & complete?** No. Local drop is an incomplete fragment (`E. incomplete_or_unusable`). `~/baseline` missing; content at `Earthquake/baseline/Tianjiyu1-LFTNet-79994f6` (2 files).
2. **Usable official checkpoint?** No.
3. **Checkpoint training data?** Unknown — no checkpoint.
4. **INSTANCE/confirm leakage risk?** Unknown (no weights). Would be `diagnostic_only_possible_leakage` if INSTANCE-trained weights appear without exclusion proof.
5. **Reproduced paper 0.844?** No (not run).
6. **Same-protocol LFTNet metrics?** Not available.
7. **Who better on F1@0.5?** Cannot compare; Ours remains the only executed same-protocol result among these.
8. **Bootstrap CI includes 0?** N/A.
9. **P/R/coverage/miss tradeoff?** N/A.
10. **P95 better via abstention?** N/A.
11. **LFTNet raises UNION oracle?** Not tested (gate failed).
12. **Worth adding LFTNet now?** No.
13. **Allowed to replace Stage 6?** No.
14. **SOTA claim?** **false**.
15. **pytest:** Stage-8 gate tests (see log).
16. **Stage 2–7 hash unchanged?** method_lock SHA256 `a02dc28e3f485559dd22b0d38c90d75923a159aed43a797bf234d765ada5b303` match declared: `True`.

## What we found in the drop

- `README.md` is a TF2.5 requirements list (not docs).
- `se-tcn-Eqt_utils.py` defines MSSE-TCN-like blocks + `cred2` detector/P/S model, but is **not importable** (no imports; missing custom layers).
- No LICENSE, train/infer scripts, configs, checkpoints, or paper 10k/20k protocol code.

## Dependencies

Do **not** install TensorFlow 2.5 into conda `PS`. After a complete Zenodo release is provided, create `LFTNet_eval`.

## Training (only if later: code-only, no weights)

Automatic training is **disabled**. Rough wall estimate (conservative): ~60 h/epoch → ~3000 h for 50 epochs on current I/O-bound proxy — **not** feasible in one week.

## User action required

Provide full Zenodo package `10.5281/zenodo.15710535` (or complete GitHub tree) **including official pretrained weights**, then re-run `scripts/run_stage8_lftnet_pipeline.sh`.

Frozen start hashes snapshot: `artifacts/results/stage8/stage2_7_frozen_hashes.json`.
