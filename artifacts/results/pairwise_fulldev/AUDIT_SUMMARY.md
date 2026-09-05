# Scalar pairwise → phaseB full-dev: pre-start audit summary

Generated before METHOD LOCK / evaluator / watchdog.

## Provenance (must keep)
- `PAIRWISE.PILOT.FAILED` — incomplete_io (waveform not finished first attempt)
- `PAIRWISE.PILOT.RESUMED_AFTER_IO_FIX` — cache IO fix
- `PAIRWISE.PILOT.FAILED_AFTER_RESUME` — `waveform_increment_not_material`

## Frozen pilot numbers (held-out)
| method | F1@0.5 | Δ vs fixed |
|--|--:|--:|
| fixed | 0.9196 | — |
| resid_s control | 0.9350 | +0.0155 |
| scalar_pairwise | 0.9403 | +0.0207 |
| waveform_plus_scalar | 0.9408 | +0.0212 (+0.00046 vs scalar) |

Waveform branch: **NO-GO**. Multi-station / moveout: **NO-GO**.

## Artifacts to use
- Checkpoint: `artifacts/results/pairwise_pilot/ckpt_scalar_pairwise_seed42.pt` (scalar-only, ~769 params)
- Split lock: `SPLIT.LOCK.json` (train/cal/heldout event hashes)
- τ = **0.50** (pilot cal; no retune)
- Features: prob, stead_prob, ida_prob, fixed_score, resid_s, resid_sp, delta_sp, src_*
- c1/c2: argmax / 2nd `fixed_score` (lw=0.5, lh=2, lp=0), tie→candidate_index
- resid control: λ=1, σ=0.5 on `|resid_s|` Gaussian bonus
- phaseB: `phaseB_eval_manifest.csv` (87293) + `phaseB_enriched_candidate_table.parquet` (157659 cands)
- Stage-6 method lock untouched; confirm untouched

## This stage
Lock scalar method → evaluate fixed vs resid vs scalar on **full phaseB** → GPU watchdog → **hard stop** (no confirm).
