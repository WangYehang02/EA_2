# Experiment protocol (paper package)

This document restates the **already frozen** protocol. It does not authorize new runs.

---

## Hard constraints for this package

- No training.
- No confirm inference rerun.
- No edits to `FINAL_METHOD.LOCK`, `CONFIRM_ANALYSIS.LOCK`, `CONFIRM_READINESS`, `PRECONFIRM_AUDIT.ADDENDUM`, `CONFIRM.CONSUMED`, Stage 6 method lock, or confirm gates.
- No confirm-time ablation or parameter selection.
- Oracle is diagnostic / ceiling only.
- DKPN remains `rejected_candidate_source`.

---

## Primary comparison

- **Method:** `fixed_rescore_UNION`
- **Baseline:** `STEAD_top1`
- **Phase:** S
- **Endpoint:** F1@0.5 s (match window 0.5 s on sample index / sampling rate)
- **Secondary:** F1@0.1 s, Precision/Recall at 0.1 and 0.5 s, miss, coverage, detected AE median/MAE/P95, wrong-peak rate
- **P95 definition:** absolute error on traces with finite pred and label; report miss_rate separately. Frozen name: `finite_pred_and_label_only_excludes_none_miss`

Always-predict on the S-labelled eval populations used here: miss=0 and Precision=Recall=F1.

---

## Data hashes

### Full-dev

- Manifest CSV: `artifacts/results/stage6/phaseB_eval_manifest.csv`
- SHA256: `2cc1e7a3376525ed46ff12d182f947602714a5b353e651e4d59542b56894c4e7`
- n_traces=87293, n_events=5341
- UNION npy SHA256: `a462b0a87dc7a6ab9fc983fff74965e740931a52393b5db0803615d91d31d27a`
- STEAD_top1 npy SHA256: `f0f202e965fbdd6ae8143c039979852a36957a74e1ebabae5eb1934b573fd4ca`

### Historical confirm

- Protocol SHA256: `b68261e63894eb94b9e21bb19737cdd6274dd8e44c099df3f3339bd8b98ca863`
- Method lock SHA256: `a02dc28e3f485559dd22b0d38c90d75923a159aed43a797bf234d765ada5b303`
- Predictions SHA256: `0ce3e3b9e04401e16a4d9bb912796096b2adbfd292aeab7d53e55b34e22bbf20`
- Metrics SHA256: `48a7b3a0d4cc28fe9d08597d2074a76d39958de84081e22f34ea31dac51163a7`
- Bootstrap SHA256: `92a9d715d7e31aac30c2e17818d1738afafe53656ff9cdbb7f7baa61d2450476`
- Integrity marker: `HISTORICAL_CONFIRM.INTEGRITY.PASSED` SHA256 `203ba9fc1add7e319dd43b188639378d2b88b040c1ab11202422eb7eb32b7de3`
- Verdict: `strong_confirmed` SHA256 `f03e0a04267ef669e12fb926b4d7b4f221ab30a6c9b9cabe073f75a7228beb93`
- CONSUMED: 2026-08-17T01:39:12.235375Z
- Method/protocol lock **before** AUTHORIZED/RUNNING

### Method / analysis locks (Stage 10, unchanged)

- `FINAL_METHOD.LOCK.json` SHA256: `a6264618105f372b548ea5e21fb92505fbddaed5a1bc034d9219d832ac286908`
- `CONFIRM_ANALYSIS.LOCK.json` SHA256: `696618462551bed207fadade4c38dc0a23378e210f25b9dbc88327f381796b08`

### Checkpoints

- STEAD: `761c46496139e9917a7480ff56f660914124c806511ea794f0005518358b09ec`
- IDA best.pt epoch 14: `b03fc6ff6c8429d877e2d69f77a31f0fd426ba285d901bdc5682bdb26b158fc4`

---

## Statistics

Paired **event** bootstrap, n=5000.

| Split | Seed | Status |
|---|---|---|
| Historical confirm | 20260817 | frozen in `confirm_bootstrap.json`; figure histogram reproduced and CI matched |
| Full-dev | 20260815 (Phase B convention) | read-only recompute from frozen npy for the paper figure; **not** a new experiment; **not** used to change the method |

Trace bootstrap is forbidden by `CONFIRM_ANALYSIS.LOCK`.

---

## Confirm gate (verbatim, 2026-08-17)

```
strong_confirmed =
  dF1_05 >= 0.01
  and CI_lo > 0
  and dF1_01 >= -0.003
  and dP95 <= 0
  and no selective recall collapse
```

Operationalization of collapse in the frozen runner: ΔRecall@0.5 < −0.02.

Forbidden on confirm: R1/R2/R3, forced_choice, none_fallback_fixed, new_gate, new_ranker, threshold/K/model selection.

Subgroups: UNION protocol list is **empty**. Do not import DKPN grouping. Do not add groups after seeing confirm.

---

## Full-dev grouping (allowed for paper figures)

Bins were preregistered in the DKPN full-dev stop-gate `METHOD.LOCK` (dev only):

- P–S interval: &lt;10 s, 10–30 s, ≥30 s
- Distance km: 0–50, 50–100, 100–200, 200–400, ≥400
- SNR dB: &lt;0, 0–10, 10–20, ≥20
- station / channel / network: all values in the frozen manifest

This package only **aggregates frozen UNION vs STEAD predictions** on those bins. It does not select groups, does not apply them to confirm, and does not change λ or K.

---

## Ablation rule

Ablations use **existing full-dev artifacts only**:

- STEAD_top1, IDA_top1, `prob_heuristic_UNION` (simple UNION), `fixed_rescore_STEAD`, `fixed_rescore_UNION`
- K ceilings: STEAD K5/K10, IDA K5, UNION K10 oracle

No confirm re-ablation. No new K grid. No new λ grid.

---

## Claims permitted by the locks

- Catalog-assisted S-phase candidate re-picking/refinement
- Internal confirm `strong_confirmed` vs STEAD_top1 under the frozen gate

**Not permitted:** SOTA vs LFTNet/INSTANCE official numbers; blind-picker claim; treating oracle or DKPN as the system; treating confirm as a development set.
