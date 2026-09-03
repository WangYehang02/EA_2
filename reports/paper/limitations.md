# Limitations and non-claims

---

## Scope of the primary method

`fixed_rescore_UNION` is **catalog-assisted**. It uses origin time, a travel-time baseline, and a frozen path-history residual. It is not a blind picker. Locks set `blind_picker_claim_allowed=false` and `sota_claim_allowed=false`.

The system outputs one S pick from a small peak list. It does not do event detection, P picking as the primary task, or multi-station association.

---

## Confirm is consumed

Internal confirm was run once (2026-08-17) and marked `CONFIRM.CONSUMED`. Those 2700 events / 74753 traces (43090 S-labelled) **must not** be used again to choose models, K, λ, thresholds, or subgroups. The paper table copies frozen metrics; it does not constitute a second blind test.

There is no preregistered UNION confirm subgroup. This package does not invent one.

---

## Oracle is not a result column

UNION candidate oracle (full-dev F1@0.5 = 0.892, confirm = 0.868) is a **ceiling**: the closest peak in the already-proposed set. It uses the human S label to pick among candidates. It is not deployable and is omitted from `main_results.csv`.

---

## IDA is not a standalone picker

IDA top-1 matches STEAD F1@0.5 on full-dev (0.839 vs 0.840) but **inflates the error tail** (P95 24.95 s vs 5.92 s). It is retained only as a second peak source. Simple max-prob union does not fix the tail.

---

## Learned alternatives failed or were closed

- **Ranker (Stage 6C):** corrected F1@0.5 = 0.832 vs fixed UNION 0.867; miss_rate ≈ 0.23. Negative ablation. Forced-choice / none-fallback variants were post-hoc diagnostics, forbidden as confirm methods.
- **LFTNet (Stage 8):** official code/weights incomplete; not reproducible; not in the main table.
- **SegPhase 100 Hz (Stage 9):** same protocol, below the primary; does not replace it.
- **Multi-station / GNN:** not started; remains blocked.

---

## DKPN is supplementary material, not a contribution

Status: `rejected_candidate_source`. `FULLDEV.STOP_GATE.FAILED`. Do not describe DKPN as a successful picker or as part of UNION.

Facts for an appendix / failure section only:

1. **Softmax then log-softmax (early seed42).** `DKPN.forward()` already returns softmax probabilities; the train loop applied `log_softmax` again. Verdict: `implementation_bug`. Those weights are not official full-dev numbers.
2. **Partial-label / mixed-crop repair.** Later clean runs used `current_partial_label_nll` (logits + log_softmax once; complete labels reduce to CE) and mixed-crop `(stable_hash(trace_id)+virtual_epoch)%3` instead of v2’s homogeneous crop cycle, which collapsed toward the noise class.
3. **FP16 activation overflow (v3).** 4-GPU AMP FP16 overflowed internal activations (`up_branch.1.2` train_forward / `down_branch.3.2` eval). Marker: `V3.FAILED_FP16_ACTIVATION_OVERFLOW`. Do not report the mixed-precision trajectory as a paper result.
4. **FP32 v4: small pilot ≠ full-dev.** After switching to FP32, the 3-virtual-epoch pilot still failed the preregistered “epoch2 ≥ epoch1” gate (`PILOT.FAILED`). Frozen full-dev of the declared primary (`epoch_1.pt`): F1@0.5 = **0.322** vs STEAD **0.840** (Δ ≈ **−0.518**); miss_rate ≈ 0.66. Adding DKPN top-5 to the union oracle raised the ceiling by only **0.0037** (&lt; +0.01 gate). `best_metric.pt` on disk stores epoch-2 weights because the saver overwrote on every improvement vs init — registered, not rewritten.
5. **Rejection.** DKPN is not in the deployable UNION. No v5, no extra seeds, no confirm use.

Pilot-subset F1 near 0.96 (small windows / unofficial thresholds) is **not** a main result.

---

## Evaluation caveats

- Metrics are for **S-labelled** traces. Events without S labels are excluded by population definition, not silently skipped at inference.
- Detected P95 excludes NaN misses; here miss_rate=0 for the primary and STEAD, so the number is well-defined.
- Full-dev ΔF1 CI in the paper figure is a **read-only recompute** from frozen npy (seed 20260815). Confirm CI is the frozen 2026-08-17 object.
- History coverage on confirm was 81.6% of traces; unavailable history zeros λh rather than imputing another event’s residual.
- INSTANCE HDF5 is Italy-centric. Transfer claims need a new external set that is not this consumed confirm.

---

## What would be required for a stronger claim (not done)

A new **independent external** test set, a fail-closed runner that loads `FINAL_METHOD.LOCK`, and no access to confirm for tuning. That is future work. This package does not start it.
