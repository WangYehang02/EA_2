# MULTISTATION_MOVEOUT.NO_GO

**Status:** FROZEN  
**Date:** 2026-09-03  
**Scope:** Stage-6 phaseB exploratory multi-station geometry (hard ring gate, soft same-ring, event-wise moveout residual)

## Verdict

**NO-GO for further multi-station geometry development** (same-ring / moveout / GNN / graph transformer).

This freeze does **not** modify:

- `artifacts/results/stage6/method_lock_stage6.json`
- confirm locks / confirm predictions / confirm metrics
- prior `artifacts/results/multistation/` or `multistation_soft/` or `multistation_moveout/` numeric artifacts

## Accurate claims (allowed)

1. same-ring / moveout work in this repo is **exploratory** and **catalog-assisted**; it is **not** a blind picker and is **not** part of the locked Stage-6 method.
2. The current **moveout-rescoring form** did **not** demonstrate independent multi-station information beyond a **single-station** theoretical residual control (`|τ_S(c) − T̂_S|`).
3. Moveout **oracle** diagnostics (neighbor catalog residuals) are a **current moveout-rescorer diagnostic ceiling**, **not** a theoretical upper bound on all multi-station methods.
4. Among **2173 recoverable** traces (baseline top1 wrong, but some UNION candidate within 0.5 s of label):
   - **83.8%** have the closest correct candidate at **fixed-score rank 2**.
5. This does **not** mean most errors are rank-2 problems overall:
   - baseline wrong @0.5 ≈ **11,627** traces (1 − 0.8668) × 87,293
   - only **2,173** are recoverable by re-ranking the existing UNION pool
   - remaining errors are largely **candidate-missing / beyond-pool**.
6. **Do not continue** geometry tuning, ring sweeps, moveout elaborations, or GNN.

## Key numbers (phaseB full-dev; already reported)

| Method | ΔF1@0.5 vs fixed_rescore_UNION | Notes |
|--|--:|--|
| soft same-ring best | ≈ +0.0024 | absolute-S soft |
| legal moveout best | ≈ +0.0056 | weighted-median residual soft |
| single-station \|resid_s\| control | ≈ +0.0069 | **no neighbors**; ≥ moveout |
| moveout neighbor-label oracle | ≈ +0.0069 | leakage diagnostic only |

Moveout ↔ single-station control prediction agreement ≈ **98.3%**.

## Next direction (not started by this freeze)

Waveform-level **top1-vs-top2 pairwise** ranking on recoverable near-miss pairs, under a separate pairwise pilot protocol. That pilot must not read confirm and must not alter this freeze or Stage-6 locks.
