# Independent-period validation report (BSI 2021–2022)

- Generated (UTC): `2026-09-07T01:24:57Z`
- Run status: **COMPLETED** (execution complete ≠ positive result)
- Validation label: independent period, same region — **not** cross-region generalization

## Sample & download
- Selected events/records (locked): 2000 / 15741
- Download requested: 15741; ok: 14822; label-outside-window: 137; other-fail: 782
- Formal pairs (READY): n=14253, events=2000
- Independence: event-id overlaps vs ranker_train/phaseB/pairs = {'ranker_train_manifest': {'n_old_events': 8058, 'n_overlap_ids': 0, 'overlap_examples': []}, 'phaseB_eval_manifest': {'n_old_events': 5341, 'n_overlap_ids': 0, 'overlap_examples': []}, 'pairs_ranker_train': {'n_old_events': 8058, 'n_overlap_ids': 0, 'overlap_examples': []}}

## A–E absolute metrics (shared QC-passed pairs)

| method | f1@0.5 | f1@0.1 | detected_ae_mae | detected_ae_median | detected_ae_p95 | frac_err_gt_5s | mech_fixes | mech_breaks | mech_net_fixes | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_fixed | 0.72939 | 0.428752 | 4.91644 | 0.134327 | 39.1431 | 0.129587 | 0 | 0 | 0 | 14253 |
| B_resid | 0.736897 | 0.434715 | 4.6998 | 0.130509 | 37.4306 | 0.124114 | 129 | 22 | 107 | 14253 |
| C_best | 0.741177 | 0.442503 | 4.66523 | 0.126425 | 37.2083 | 0.123132 | 232 | 64 | 168 | 14253 |
| D_linear | 0.734161 | 0.441381 | 3.97577 | 0.127874 | 29.7464 | 0.122711 | 251 | 183 | 68 | 14253 |
| E_scalar | 0.74665 | 0.446082 | 3.4131 | 0.124158 | 25.36 | 0.112959 | 327 | 81 | 246 | 14253 |

## Pre-registered comparisons

### Primary: scalar − C (F1@0.5)
- ΔF1@0.5 = **0.005473** (E=0.746650, C=0.741177)
- Paired event bootstrap 5000: mean=0.005464, 95% CI=[0.003592, 0.007291]
- Tail secondary ΔMAE (E−C) = -1.252127; Δfrac_err_gt_5s (E−C) = -0.010173
- Tail metrics **must not** replace the primary F1 endpoint.

### Secondary (preregistered): C − fixed (F1@0.5)
- ΔF1@0.5 = **0.011787** (C=0.741177, A=0.729390)
- Bootstrap 95% CI=[0.009247, 0.014457]

## Answers to the three questions
1. Does scalar F1 gain vs C retain? point=True, CI_excludes_zero_positive=True. Practical F1 discussion scale remains |ΔF1@0.5|≈0.003; do not claim material F1 gain from a barely-positive CI alone.
2. Do scalar MAE / >5s tail benefits retain? mae_better=True, gt5_better=True (lower is better for both).
3. Does C retain gain vs fixed? True (Δ=0.011787, CI=[0.009246902817373611, 0.014457423642714858]).


## Population notes (pre-metric)
- Selected rows → unique keys after sha collision resolve: 15741 → 15141 (dropped 600 duplicate-key rows; **not** nearest-to-model)
- Download unique ok / outside / other-fail: 14253 / 133 / 755
- Formal shared A–E pairs: **14253** on **2000** events (size frozen before metrics)
- Outside-window (137 requests / 133 unique) counted separately; not used to retune windows

## Download reason counts
```json
{
  "ok": 14822,
  "station_meta:HTTP_204": 646,
  "FAIL_LABEL_OUTSIDE_WINDOW": 137,
  "HTTP_204_NO_DATA": 96,
  "HTTP_403": 29,
  "missing_3C": 9,
  "dataselect:transient:RetryError:HTTPSConnectionPool(host='webservices.ingv.it', port=443): Max retries exceeded with url: /fdsnws/dataselect/1/query?network=IV&station=BOZZ&location=%2A&channel=HH%3F%2CEH%3F%2CBH%3F%2CHN%3F&starttime=2022-01-14T23%3A33%3A02.175584Z&endtime=2022-01-14T23%3A35%3A02.175584Z (Caused by ResponseError('too many 502 error responses'))": 1,
  "dataselect:transient:RetryError:HTTPSConnectionPool(host='webservices.ingv.it', port=443): Max retries exceeded with url: /fdsnws/dataselect/1/query?network=IV&station=CAR1&location=%2A&channel=HH%3F%2CEH%3F%2CBH%3F%2CHN%3F&starttime=2022-07-19T14%3A38%3A30.786570Z&endtime=2022-07-19T14%3A40%3A30.786570Z (Caused by ResponseError('too many 502 error responses'))": 1
}
```

## Artifact paths
- Data root: `/data/yehang/Earthquake_paper_strengthening_v1/independent_period`
- READY: `/data/yehang/Earthquake_paper_strengthening_v1/independent_period/READY.json`
- Predictions lock: `/data/yehang/Earthquake_paper_strengthening_v1/independent_period/locks/PREDICTIONS.LOCK.json`
- Metrics: `/data/yehang/Earthquake_paper_strengthening_v1/independent_period/results/metrics_A_to_E.csv`
- Comparisons: `/data/yehang/Earthquake_paper_strengthening_v1/independent_period/results/comparisons.json`
- Logs: `/data/yehang/Earthquake_paper_strengthening_v1/independent_period/logs`
