# Stage 10B — Clean DKPN feasibility

**UTC:** 2026-08-18T05:08:19.308406+00:00

## Smoke
- smoke_ok: **True**
- device: `cpu`
- mean_loss (20 steps): 0.4978612810373306
- output_finite: True

## Overfit (32 traces, cached batch, localization)
- overfit_ok: **True**
- loss: 0.3412967920303345 → 0.18974651396274567 (drop 0.444)
- S-peak ±0.2s acc: 0.0625 → 0.9375

## Planned clean train counts
- P+S: **377087**
- P-only (masked): **254708**
- noise @0.2: **126359**
- total: **758154**

## ETA (revised)
CPU measured E2E (CF+train) ≈ 2.8201068322121072 t/s → impractical for full train.
Conservative 1×GPU with parallel CF ≈ 15 t/s → ~421.19666666666666 h / 30 ep.
See `stage10B_dkpn_feasibility.json` for 4/8 GPU scaled ETA.
**Must remount on free GPU before trusting production ETA.**

## Policy
- Official INSTANCE DKPN weights: **not** for main table
- Confirm: **not** used for Stage10 tuning
- SOTA claim: **forbidden** (`exact_lftnet_protocol_reproducible=false`)

## Next stop gate
DKPN seed42 vs strongest waveform-only Stage6/7/9 DEV baseline: ΔF1@0.5≥+0.01 OR (F1 not down & P95≥0.15s better) OR UNION oracle +0.01; else rejected_candidate_source
