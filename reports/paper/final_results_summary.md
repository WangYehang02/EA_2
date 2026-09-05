# Final Results Summary (evidence-driven)

Primary method: **scalar_pairwise**  
Task: **catalog-assisted S-phase candidate reranking / re-picking**  
Evidence status: **FINAL_EVIDENCE.LOCKED**

## 1. Task definition

Given frozen UNION S candidates from base pickers and catalog-assisted source/origin metadata, select between the two highest `fixed_score` candidates (c1/c2) using a scalar pairwise model (τ=0.50). No new arrivals; no abstain; no rank≥3 selection.

## 2. Candidate oracle / motivation

Candidate sets contain recoverable headroom: when a correct candidate exists but is not selected, fixed selection leaves ranking gap. Candidate oracle is diagnostic only—not a deployable method.

## 3. Ranking-gap forensic evidence

On ranking-recoverable cases (n=2173):

| Rank | Count | Fraction | Cumulative |
| --- | ---: | ---: | ---: |
| 2 | 1820 | 0.838 | 0.838 |
| 3 | 307 | 0.141 | 0.979 |
| 4 | 40 | 0.018 | 0.997 |
| 5 | 6 | 0.003 | 1.000 |

Supports: among recoverable errors, ranking (especially rank-2 near-miss) is the dominant structure—not a claim that all errors are ranking errors.

## 4. Residual control

`resid_s` control reweights using train-only propagation/history geometry (corr≈0.998 with expected_s residual). It is a strong scalar control, not multi-station info and not label leakage.

## 5. Scalar pairwise pilot (held-out)

| Method | F1@0.5 |
| --- | ---: |
| fixed | 0.9196 |
| resid | 0.9350 |
| scalar | 0.9403 |

Δ scalar−fixed = +0.0207; Δ scalar−resid = +0.0053.

## 6. Full-dev validation

| Method | F1@0.5 |
| --- | ---: |
| fixed | 0.8668 |
| resid | 0.8732 |
| scalar | 0.8805 |

Δ scalar−fixed = +0.0137; Δ scalar−resid = +0.0073. Status: `SCALAR_PAIRWISE.FULLDEV.PASSED` (method lock hash frozen).

## 7. Confirmatory result

n=43090 traces / 2669 events / ≥2 candidates=27560; c1 agreement with frozen fixed UNION = 1.0.

| Method | F1@0.1 | F1@0.5 | P95 (s) | Δ vs fixed | Δ vs resid |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed_rescore_UNION | 0.5035 | 0.8373 | 2.47 | 0 | −0.0085 |
| resid_s control | 0.5108 | 0.8459 | 2.12 | +0.0085 | 0 |
| scalar_pairwise | 0.5277 | 0.8535 | 1.85 | **+0.0162** | **+0.0076** |

Event bootstrap 5000:

- scalar vs fixed: mean +0.0162; 95% CI [+0.0146, +0.0178]  
- scalar vs resid: mean +0.0076; 95% CI [+0.0065, +0.0088]  

Status: **CONFIRMED**.

## 8. Mechanism analysis

Confirm switches: fixes=907, breaks=210, net=+697; switch_precision≈0.812; Q2_recovery≈0.773.  
Margin bins are **post-hoc** mechanism views only (not gate design).

## 9. Negative-result ablations (supplement)

| Branch | ΔF1@0.5 | Takeaway |
| --- | ---: | --- |
| hard same-ring | +0.0078 | Abstain≈5.8% breaks always-output |
| soft same-ring | +0.0024 | Small |
| moveout legal best | +0.0056 | ≈ single-station resid; NO-GO |
| waveform-only | +0.0082 vs fixed (pilot) | Informative alone; below scalar |
| waveform+scalar | +0.00046 vs scalar | CI crosses 0; not material |

## 10. Scope / limitations

- Catalog-assisted (uses current-event source/origin metadata).  
- Requires a candidate set; does not fix candidate-missing errors.  
- Not blind continuous detection / not multi-station final method.  
- Evidence domain = current INSTANCE-based splits; not universal generalization.
