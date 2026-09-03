# Paper results summary (frozen; no new experiments)

**Deployable primary:** `fixed_rescore_UNION` (`UNION_STEAD5_IDA5` + fixed history rescore).  
**Baseline:** `STEAD_top1`.  
**Oracle:** ceiling only — not a system, not in the main-result columns.  
**DKPN:** `rejected_candidate_source` — supplementary failure analysis only.  
**New inference this package:** false. Method locks and thresholds were not changed.

Machine index: `artifacts/paper/PAPER.RESULTS.LOCK.json`

---

## Datasets

| Split | Population | Events | S-labelled traces | Manifest SHA256 |
|---|---|---:|---:|---|
| Stage 6 full-dev | all S-labelled traces in `stage6_dev` | 5341 | 87293 | `2cc1e7a3376525ed46ff12d182f947602714a5b353e651e4d59542b56894c4e7` |
| Historical confirm | all S-labelled traces in consumed internal confirm | 2669 | 43090 | CSV `729681037c4d28d1208c29a13c05ffb987cc4cbfa7361a897287fe1227f63543` |

Full confirm partition is 2700 events / 74753 traces; 31 events have no S label and are outside the preregistered eval population.

Integrity of the 2026-08-17 confirm run: `HISTORICAL_CONFIRM.INTEGRITY.PASSED`.  
Gate verdict: `strong_confirmed` (`HISTORICAL_CONFIRM.VERDICT.json`).

---

## Main table (oracle omitted)

Source CSV: `artifacts/paper/main_results.csv`

Always-predict protocol on S-labelled traces: Precision = Recall = F1. Miss = 0, coverage = 1.

### Full-dev

| Method | P/R/F1@0.1 | P/R/F1@0.5 | miss | coverage | AE median / MAE / P95 (s) |
|---|---:|---:|---:|---:|---|
| STEAD_top1 | 0.530260 | **0.840262** | 0 | 1 | 0.09 / 2.328 / **5.92** |
| **fixed_rescore_UNION** | 0.553962 | **0.866805** | 0 | 1 | 0.09 / 1.536 / **1.66** |

ΔF1@0.5 (UNION − STEAD) = **0.026543**.  
Paired event bootstrap (read-only from frozen npy, seed=20260815, n=5000): 95% CI **[0.024836, 0.028242]**.  
This CI was not used to change the method; the primary was locked before confirm.

### Historical confirm (consumed 2026-08-17)

| Method | P/R/F1@0.1 | P/R/F1@0.5 | miss | coverage | AE median / MAE / P95 (s) |
|---|---:|---:|---:|---:|---|
| STEAD_top1 | 0.485171 | **0.817568** | 0 | 1 | 0.11 / 2.292 / **6.10** |
| **fixed_rescore_UNION** | 0.503458 | **0.837317** | 0 | 1 | 0.10 / 1.918 / **2.465** |

ΔF1@0.5 = **0.019749**.  
Frozen paired event bootstrap (seed=20260817, n=5000): 95% CI **[0.017345, 0.022289]**.  
ΔP95 = **−3.635 s**. No selective recall collapse (ΔRecall@0.5 = +0.019749).

Preregistered confirm gate, all true:

- dF1_05 ≥ 0.01
- CI_lo > 0
- dF1_01 ≥ −0.003
- dP95 ≤ 0
- no selective recall collapse

**Verdict: `strong_confirmed`.** Confirm was not reused to pick K, λ, threshold, or checkpoint.

---

## Ablation (full-dev only; no confirm re-ablation)

Source: `artifacts/paper/ablation_results.csv`

| Method | Role | F1@0.1 | F1@0.5 | P95 (s) |
|---|---|---:|---:|---:|
| STEAD_top1 | waveform-only baseline | 0.530 | 0.840 | 5.92 |
| IDA_top1 | second source, not primary | 0.553 | 0.839 | 24.95 |
| simple UNION (`prob_heuristic_UNION`) | max-prob on union pool | 0.535 | 0.842 | 6.57 |
| fixed-rescore STEAD | history rescore, STEAD peaks only | 0.540 | 0.854 | 2.15 |
| **fixed-rescore UNION** | **deployable primary** | **0.554** | **0.867** | **1.66** |
| STEAD K=5 oracle | ceiling, not deployable | 0.544 | 0.862 | 1.94 |
| STEAD K=10 oracle | same as K=5 on this cache | 0.544 | 0.862 | 1.94 |
| IDA K=5 oracle | ceiling | 0.575 | 0.872 | 2.12 |
| UNION K=10 oracle | ceiling only | 0.643 | 0.892 | 1.35 |

Reading:

- Merging candidates without history (`simple UNION`) barely beats STEAD F1 and **worsens P95**.
- History rescore on STEAD alone already cuts the tail (P95 5.92 → 2.15) and lifts F1 to 0.854.
- Adding IDA to the pool then rescoring reaches 0.867 / 1.66 s.
- IDA top-1 is **not** a replacement picker (P95 ≈ 25 s).
- STEAD K=10 oracle = STEAD K=5 oracle: extra STEAD peaks do not move the ceiling. Locked K=5 per source is sufficient.

Oracle UNION F1@0.5 = 0.892 is an upper bound (~0.025 above the deployable primary). It is not reported in the main table.

Ranker R3 on the same full-dev set is a **negative** result (F1@0.5 = 0.832, miss_rate ≈ 0.226) and is not a primary or ablation winner.

---

## Figures (data + scripts; no new inference)

| Figure | Data | Script |
|---|---|---|
| Method F1 bars | `figure_data/method_comparison.csv` | `scripts/plot_paper_figures.py` |
| Event-bootstrap ΔF1 histograms | `bootstrap_delta_f1_*.csv` | same |
| Absolute-error CDF | `ae_cdf.csv` | same |
| Full-dev preregistered groups | `grouping_fulldev_*.csv` | same |
| STEAD vs IDA complementarity | `complementarity_fulldev_k5_tol0.5.csv` | (table) |

Rendered copies: `artifacts/paper/figures/*.pdf` and `*.png`.

Confirm has **no** preregistered subgroup. Marker: `figure_data/CONFIRM_NO_PREREGISTERED_SUBGROUP.txt`. DKPN grouping families were **not** imported onto confirm.

Full-dev grouping (DKPN stop-gate bins, applied to frozen UNION vs STEAD predictions; not used to change the method):

**P–S interval:** gain in every bin (largest on `ps_lt_10s`, ΔF1@0.5 = +0.028).  
**Distance:** gain in all populated bins; `dist_ge_400` has n=1.  
**SNR:** gain in all four bins, including `snr_lt_0` (+0.035).  
Network / channel / station CSVs list every observed value; no post-hoc subset.

---

## Complementarity (why UNION exists)

On full-dev, K=5, tolerance 0.5 s (candidate-level, not the deployable picker):

| Cell | Traces | Rate |
|---|---:|---:|
| both correct | 73362 | 0.840 |
| STEAD only | 1858 | 0.021 |
| IDA only | 2732 | 0.031 |
| neither | 9341 | 0.107 |

IDA recovers a non-trivial exclusive set; STEAD remains the stronger single waveform picker. Fixed rescore is what converts the union pool into a deployable pick.

---

## What this package did not do

- No training, no confirm rerun, no new predictions.
- No threshold / K / λ / checkpoint selection on confirm.
- No post-hoc confirm subgroups.
- No promotion of DKPN, ranker, LFTNet, or oracle into the primary.
