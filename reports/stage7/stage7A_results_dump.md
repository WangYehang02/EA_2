# Stage 7A — Consolidated Results Dump

本文件汇总下列产物原文/表格化抄录（便于一处查阅）。

- `artifacts/results/stage7/comparator_metrics_confirm.json`（由 csv 导出）
- `artifacts/results/stage7/comparator_bootstrap.json`
- `artifacts/results/stage7/stage7A_final_verdict.json`
- `reports/stage7/stage7A_same_protocol_benchmark.md`
- `artifacts/results/stage7/comparator_registry.json`
- `artifacts/results/stage7/runtime_benchmark.json`
- `artifacts/results/stage7/runtime_scaling.json`（由 csv 导出）

---

# 1. stage7A_same_protocol_benchmark.md（原文）

# Stage 7A — Same-Protocol External Baseline Benchmark

**Disclosure:** This is *post-confirm external comparator evaluation*. Stage-6 confirm is already `CONFIRM.CONSUMED`. Comparators were **not** co-preregistered with the Stage-6 main method.

- Main method (unchanged): `fixed_rescore_UNION`
- Stage-6 method_lock SHA256: `a02dc28e3f485559dd22b0d38c90d75923a159aed43a797bf234d765ada5b303`
- `sota_claim_allowed`: **false**

## Locked thresholds (dev-only)

- **PhaseNet-SCEDC**: threshold=0.05 (dev F1@0.5=0.7584)
- **PhaseNet-ETHZ**: threshold=0.05 (dev F1@0.5=0.8355)

## Confirm metrics (S-labelled)

| Method | F1@0.1 | F1@0.5 | P95 | miss | coverage |
|--|--:|--:|--:|--:|--:|
| PhaseNet-ETHZ | 0.4776 | 0.8122 | 1.927 | 0.1198 | 0.8802 |
| PhaseNet-SCEDC | 0.3430 | 0.7263 | 58.657 | 0.0683 | 0.9317 |
| STEAD_top1 | 0.4852 | 0.8176 | 6.100 | 0.0000 | 1.0000 |
| fixed_rescore_STEAD | 0.4900 | 0.8254 | 3.510 | 0.0000 | 1.0000 |
| fixed_rescore_UNION | 0.5035 | 0.8373 | 2.465 | 0.0000 | 1.0000 |
| oracle_UNION | 0.6017 | 0.8676 | 1.780 | 0.0000 | 1.0000 |

## Bootstrap (fixed_UNION − comparator)

- **fixed_UNION_vs_PhaseNet-ETHZ**: ΔF1@0.5=+0.0251 95% CI=[+0.0215, +0.0288]
- **fixed_UNION_vs_PhaseNet-SCEDC**: ΔF1@0.5=+0.1110 95% CI=[+0.1060, +0.1161]
- **fixed_UNION_vs_STEAD_top1_frozen**: ΔF1@0.5=+0.0198 95% CI=[+0.0173, +0.0223]

## Interpretation rule

If fixed UNION's ΔF1@0.5 CI is entirely above 0 vs all valid external comparators, we may state it *outperformed the evaluated external pretrained baselines under our Stage 6 protocol*. We do **not** claim SOTA / best on INSTANCE / surpasses all phase pickers.

Generated UTC: 2026-08-17T07:13:05.582650+00:00

---

# 2. stage7A_final_verdict.json

```json
{
  "stage": "7A",
  "sota_claim_allowed": false,
  "post_confirm_external_comparator_evaluation": true,
  "stage6_main_method_unchanged": "fixed_rescore_UNION",
  "stage6_method_lock_sha256": "a02dc28e3f485559dd22b0d38c90d75923a159aed43a797bf234d765ada5b303",
  "new_comparators": {
    "PhaseNet-ETHZ": {
      "threshold": 0.05
    },
    "PhaseNet-SCEDC": {
      "threshold": 0.05
    }
  },
  "excluded": [
    "EQTransformer-* (download failed)",
    "LFTNet (not reproducible)",
    "instance weights (leakage)"
  ],
  "timestamp_utc": "2026-08-17T06:30:38.145249+00:00",
  "fixed_UNION_vs_PhaseNet-ETHZ": {
    "mean_delta": 0.025122382921583763,
    "ci95": [
      0.021528559071963216,
      0.0288428024399646
    ],
    "fixed_better_stable": true
  },
  "fixed_UNION_vs_PhaseNet-SCEDC": {
    "mean_delta": 0.1109922343172051,
    "ci95": [
      0.10598497215100565,
      0.11613820616423204
    ],
    "fixed_better_stable": true
  }
}
```

---

# 3. comparator_metrics_confirm

## 3.1 主表

| model | thr | F1@0.1 | F1@0.5 | P@0.5 | R@0.5 | miss | wrong-peak | coverage | AE med | AE MAE | AE P90 | AE P95 | AE P99 | denom | n_pred |
|--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| PhaseNet-ETHZ | 0.05 | 0.4776 | 0.8122 | 0.8675 | 0.7635 | 0.1198 | 0.1167 | 0.8802 | 0.100 | 1.997 | 0.720 | 1.927 | 68.844 | 37927 | 37927 |
| PhaseNet-SCEDC | 0.05 | 0.3430 | 0.7263 | 0.7530 | 0.7015 | 0.0683 | 0.2301 | 0.9317 | 0.170 | 5.886 | 11.810 | 58.657 | 83.696 | 40146 | 40146 |
| STEAD_top1 |  | 0.4852 | 0.8176 | 0.8176 | 0.8176 | 0.0000 | 0.1824 | 1.0000 | 0.110 | 2.292 | 1.170 | 6.100 | 68.494 | 43090 | 43090 |
| fixed_rescore_STEAD |  | 0.4900 | 0.8254 | 0.8254 | 0.8254 | 0.0000 | 0.1746 | 1.0000 | 0.110 | 2.003 | 1.070 | 3.510 | 62.424 | 43090 | 43090 |
| fixed_rescore_UNION |  | 0.5035 | 0.8373 | 0.8373 | 0.8373 | 0.0000 | 0.1627 | 1.0000 | 0.100 | 1.918 | 0.940 | 2.465 | 66.362 | 43090 | 43090 |
| oracle_UNION |  | 0.6017 | 0.8676 | 0.8676 | 0.8676 | 0.0000 | 0.1324 | 1.0000 | 0.070 | 1.153 | 0.740 | 1.780 | 35.314 | 43090 | 43090 |

## 3.2 TP/FP/FN @0.1 / @0.5

| model | tp@0.1 | fp@0.1 | fn@0.1 | tp@0.5 | fp@0.5 | fn@0.5 | matched@0.5 |
|--|--:|--:|--:|--:|--:|--:|--:|
| PhaseNet-ETHZ | 19347 | 18580 | 23743 | 32900 | 5027 | 10190 | 32900 |
| PhaseNet-SCEDC | 14276 | 25870 | 28814 | 30229 | 9917 | 12861 | 30229 |
| STEAD_top1 | 20906 | 22184 | 22184 | 35229 | 7861 | 7861 | 35229 |
| fixed_rescore_STEAD | 21114 | 21976 | 21976 | 35566 | 7524 | 7524 | 35566 |
| fixed_rescore_UNION | 21694 | 21396 | 21396 | 36080 | 7010 | 7010 | 36080 |
| oracle_UNION | 25928 | 17162 | 17162 | 37386 | 5704 | 5704 | 37386 |

## 3.3 JSON 全文

```json
[
  {
    "model": "PhaseNet-ETHZ",
    "threshold": 0.05,
    "n_traces": 43090,
    "n_labeled": 43090.0,
    "n_predicted": 37927.0,
    "n_finite_prediction": 37927,
    "prediction_coverage": 0.8801810164771409,
    "predicted_pick_count": 37927.0,
    "none_of_k_count": 5163.0,
    "none_of_k_rate": 0.1198189835228591,
    "miss_count": 5163.0,
    "miss_rate": 0.1198189835228591,
    "wrong_peak_count": 5027.0,
    "wrong_peak_rate": 0.1166627987932234,
    "matched_count_0.5": 32900,
    "detected_ae_n": 37927.0,
    "detected_ae_median": 0.1,
    "detected_ae_mae": 1.9971998839876608,
    "detected_ae_p90": 0.72,
    "detected_ae_p95": 1.926999999999968,
    "detected_ae_p99": 68.84439999999988,
    "detected_ae_p95_denominator": 37927,
    "p95_scope": "finite_pred_and_label_only_excludes_none_miss",
    "p95_naming_warning": "detected_ae_p95 / legacy e2e_p95_ae excludes NaN misses; report miss_rate separately; do not call this alone complete end-to-end P95",
    "precision@0.1": 0.5101115300445593,
    "recall@0.1": 0.4489904850313297,
    "f1@0.1": 0.4776034659392473,
    "tp@0.1": 19347,
    "fp@0.1": 18580,
    "fn@0.1": 23743,
    "precision@0.2": 0.7065415139610304,
    "recall@0.2": 0.6218844279415178,
    "f1@0.2": 0.6615154844045077,
    "tp@0.2": 26797.0,
    "fp@0.2": 11130.0,
    "fn@0.2": 16293.0,
    "precision@0.5": 0.8674559021277718,
    "recall@0.5": 0.7635182176839174,
    "f1@0.5": 0.81217522248417,
    "tp@0.5": 32900,
    "fp@0.5": 5027,
    "fn@0.5": 10190
  },
  {
    "model": "PhaseNet-SCEDC",
    "threshold": 0.05,
    "n_traces": 43090,
    "n_labeled": 43090.0,
    "n_predicted": 40146.0,
    "n_finite_prediction": 40146,
    "prediction_coverage": 0.931677883499652,
    "predicted_pick_count": 40146.0,
    "none_of_k_count": 2944.0,
    "none_of_k_rate": 0.0683221165003481,
    "miss_count": 2944.0,
    "miss_rate": 0.0683221165003481,
    "wrong_peak_count": 9917.0,
    "wrong_peak_rate": 0.2301462056161522,
    "matched_count_0.5": 30229,
    "detected_ae_n": 40146.0,
    "detected_ae_median": 0.17,
    "detected_ae_mae": 5.886316444975838,
    "detected_ae_p90": 11.81,
    "detected_ae_p95": 58.6575,
    "detected_ae_p99": 83.69550000000002,
    "detected_ae_p95_denominator": 40146,
    "p95_scope": "finite_pred_and_label_only_excludes_none_miss",
    "p95_naming_warning": "detected_ae_p95 / legacy e2e_p95_ae excludes NaN misses; report miss_rate separately; do not call this alone complete end-to-end P95",
    "precision@0.1": 0.3556020525083445,
    "recall@0.1": 0.3313065676491065,
    "f1@0.1": 0.343024652794464,
    "tp@0.1": 14276,
    "fp@0.1": 25870,
    "fn@0.1": 28814,
    "precision@0.2": 0.5486225277736263,
    "recall@0.2": 0.5111394755163611,
    "f1@0.2": 0.5292181267720697,
    "tp@0.2": 22025.0,
    "fp@0.2": 18121.0,
    "fn@0.2": 21065.0,
    "precision@0.5": 0.7529766352812235,
    "recall@0.5": 0.7015316778834997,
    "f1@0.5": 0.7263443702244221,
    "tp@0.5": 30229,
    "fp@0.5": 9917,
    "fn@0.5": 12861
  },
  {
    "model": "STEAD_top1",
    "threshold": NaN,
    "n_traces": 43090,
    "n_labeled": NaN,
    "n_predicted": NaN,
    "n_finite_prediction": 43090,
    "prediction_coverage": 1.0,
    "predicted_pick_count": NaN,
    "none_of_k_count": NaN,
    "none_of_k_rate": NaN,
    "miss_count": NaN,
    "miss_rate": 0.0,
    "wrong_peak_count": NaN,
    "wrong_peak_rate": 0.1824321188210722,
    "matched_count_0.5": 35229,
    "detected_ae_n": NaN,
    "detected_ae_median": 0.11,
    "detected_ae_mae": 2.29244929217916,
    "detected_ae_p90": 1.1700000000000004,
    "detected_ae_p95": 6.1,
    "detected_ae_p99": 68.49440000000001,
    "detected_ae_p95_denominator": 43090,
    "p95_scope": NaN,
    "p95_naming_warning": NaN,
    "precision@0.1": 0.4851705732188442,
    "recall@0.1": 0.4851705732188442,
    "f1@0.1": 0.4851705732188442,
    "tp@0.1": 20906,
    "fp@0.1": 22184,
    "fn@0.1": 22184,
    "precision@0.2": NaN,
    "recall@0.2": NaN,
    "f1@0.2": NaN,
    "tp@0.2": NaN,
    "fp@0.2": NaN,
    "fn@0.2": NaN,
    "precision@0.5": 0.8175678811789279,
    "recall@0.5": 0.8175678811789279,
    "f1@0.5": 0.8175678811789279,
    "tp@0.5": 35229,
    "fp@0.5": 7861,
    "fn@0.5": 7861
  },
  {
    "model": "fixed_rescore_STEAD",
    "threshold": NaN,
    "n_traces": 43090,
    "n_labeled": NaN,
    "n_predicted": NaN,
    "n_finite_prediction": 43090,
    "prediction_coverage": 1.0,
    "predicted_pick_count": NaN,
    "none_of_k_count": NaN,
    "none_of_k_rate": NaN,
    "miss_count": NaN,
    "miss_rate": 0.0,
    "wrong_peak_count": NaN,
    "wrong_peak_rate": 0.1746112787189603,
    "matched_count_0.5": 35566,
    "detected_ae_n": NaN,
    "detected_ae_median": 0.11,
    "detected_ae_mae": 2.0033246692968203,
    "detected_ae_p90": 1.07,
    "detected_ae_p95": 3.51,
    "detected_ae_p99": 62.42440000000003,
    "detected_ae_p95_denominator": 43090,
    "p95_scope": NaN,
    "p95_naming_warning": NaN,
    "precision@0.1": 0.4899976792759341,
    "recall@0.1": 0.4899976792759341,
    "f1@0.1": 0.4899976792759341,
    "tp@0.1": 21114,
    "fp@0.1": 21976,
    "fn@0.1": 21976,
    "precision@0.2": NaN,
    "recall@0.2": NaN,
    "f1@0.2": NaN,
    "tp@0.2": NaN,
    "fp@0.2": NaN,
    "fn@0.2": NaN,
    "precision@0.5": 0.8253887212810397,
    "recall@0.5": 0.8253887212810397,
    "f1@0.5": 0.8253887212810397,
    "tp@0.5": 35566,
    "fp@0.5": 7524,
    "fn@0.5": 7524
  },
  {
    "model": "fixed_rescore_UNION",
    "threshold": NaN,
    "n_traces": 43090,
    "n_labeled": NaN,
    "n_predicted": NaN,
    "n_finite_prediction": 43090,
    "prediction_coverage": 1.0,
    "predicted_pick_count": NaN,
    "none_of_k_count": NaN,
    "none_of_k_rate": NaN,
    "miss_count": NaN,
    "miss_rate": 0.0,
    "wrong_peak_count": NaN,
    "wrong_peak_rate": 0.1626827570201903,
    "matched_count_0.5": 36080,
    "detected_ae_n": NaN,
    "detected_ae_median": 0.1,
    "detected_ae_mae": 1.917768623810629,
    "detected_ae_p90": 0.94,
    "detected_ae_p95": 2.4654999999999565,
    "detected_ae_p99": 66.36220000000002,
    "detected_ae_p95_denominator": 43090,
    "p95_scope": NaN,
    "p95_naming_warning": NaN,
    "precision@0.1": 0.5034578788582038,
    "recall@0.1": 0.5034578788582038,
    "f1@0.1": 0.5034578788582038,
    "tp@0.1": 21694,
    "fp@0.1": 21396,
    "fn@0.1": 21396,
    "precision@0.2": NaN,
    "recall@0.2": NaN,
    "f1@0.2": NaN,
    "tp@0.2": NaN,
    "fp@0.2": NaN,
    "fn@0.2": NaN,
    "precision@0.5": 0.8373172429798097,
    "recall@0.5": 0.8373172429798097,
    "f1@0.5": 0.8373172429798097,
    "tp@0.5": 36080,
    "fp@0.5": 7010,
    "fn@0.5": 7010
  },
  {
    "model": "oracle_UNION",
    "threshold": NaN,
    "n_traces": 43090,
    "n_labeled": NaN,
    "n_predicted": NaN,
    "n_finite_prediction": 43090,
    "prediction_coverage": 1.0,
    "predicted_pick_count": NaN,
    "none_of_k_count": NaN,
    "none_of_k_rate": NaN,
    "miss_count": NaN,
    "miss_rate": 0.0,
    "wrong_peak_count": NaN,
    "wrong_peak_rate": 0.1323741007194244,
    "matched_count_0.5": 37386,
    "detected_ae_n": NaN,
    "detected_ae_median": 0.07,
    "detected_ae_mae": 1.1532476212578324,
    "detected_ae_p90": 0.74,
    "detected_ae_p95": 1.78,
    "detected_ae_p99": 35.31440000000003,
    "detected_ae_p95_denominator": 43090,
    "p95_scope": NaN,
    "p95_naming_warning": NaN,
    "precision@0.1": 0.6017173358087723,
    "recall@0.1": 0.6017173358087723,
    "f1@0.1": 0.6017173358087723,
    "tp@0.1": 25928,
    "fp@0.1": 17162,
    "fn@0.1": 17162,
    "precision@0.2": NaN,
    "recall@0.2": NaN,
    "f1@0.2": NaN,
    "tp@0.2": NaN,
    "fp@0.2": NaN,
    "fn@0.2": NaN,
    "precision@0.5": 0.8676258992805755,
    "recall@0.5": 0.8676258992805755,
    "f1@0.5": 0.8676258992805755,
    "tp@0.5": 37386,
    "fp@0.5": 5704,
    "fn@0.5": 5704
  }
]
```

---

# 4. comparator_bootstrap.json

定义：`delta = fixed_UNION - comparator`（ΔF1>0 表示 UNION 更好；ΔP95<0 表示 UNION 的 P95 更好）。

## fixed_UNION_vs_PhaseNet-ETHZ

- n_boot: `5000`  seed: `20260817`  definition: `delta = fixed_UNION - comparator`

| metric | point_delta | mean_delta | 95% CI |
|--|--:|--:|--|
| f1@0.5 | 0.025142 | 0.025122 | [+0.021529, +0.028843] |
| f1@0.1 | 0.025854 | 0.025934 | [+0.021041, +0.030950] |
| precision@0.5 | -0.030139 | -0.030181 | [-0.034761, -0.025749] |
| recall@0.5 | 0.073799 | 0.073793 | [+0.069463, +0.078088] |
| miss_rate | -0.119819 | -0.119857 | [-0.125834, -0.113995] |
| wrong_peak_rate | 0.046020 | 0.046065 | [+0.041277, +0.051112] |
| detected_ae_p95 | 0.538500 | 0.553249 | [+0.215925, +0.941563] |

## fixed_UNION_vs_PhaseNet-SCEDC

- n_boot: `5000`  seed: `20260817`  definition: `delta = fixed_UNION - comparator`

| metric | point_delta | mean_delta | 95% CI |
|--|--:|--:|--|
| f1@0.5 | 0.110973 | 0.110992 | [+0.105985, +0.116138] |
| f1@0.1 | 0.160433 | 0.160467 | [+0.154323, +0.166539] |
| precision@0.5 | 0.084341 | 0.084345 | [+0.079132, +0.089562] |
| recall@0.5 | 0.135786 | 0.135816 | [+0.130550, +0.141159] |
| miss_rate | -0.068322 | -0.068362 | [-0.072610, -0.064210] |
| wrong_peak_rate | -0.067463 | -0.067454 | [-0.072833, -0.062312] |
| detected_ae_p95 | -56.192000 | -56.203530 | [-58.410087, -53.979950] |

## fixed_UNION_vs_STEAD_top1_frozen

- n_boot: `5000`  seed: `20260817`  definition: ``

| metric | point_delta | mean_delta | 95% CI |
|--|--:|--:|--|
| f1@0.5 | 0.019749 | 0.019754 | [+0.017345, +0.022289] |
| f1@0.1 | 0.018287 | 0.018279 | [+0.015752, +0.020750] |
| precision@0.5 | 0.019749 | 0.019754 | [+0.017345, +0.022289] |
| recall@0.5 | 0.019749 | 0.019754 | [+0.017345, +0.022289] |
| miss_rate | 0.000000 | 0.000000 | [+0.000000, +0.000000] |
| wrong_peak_rate | -0.019749 | -0.019754 | [-0.022289, -0.017345] |
| detected_ae_p95 | -3.634500 | -3.570241 | [-6.010087, -1.931463] |

## 4.1 JSON 全文

```json
{
  "fixed_UNION_vs_PhaseNet-ETHZ": {
    "n_boot": 5000,
    "seed": 20260817,
    "definition": "delta = fixed_UNION - comparator",
    "point_delta": {
      "f1@0.5": 0.02514202049563974,
      "f1@0.1": 0.025854412918956515,
      "precision@0.5": -0.0301386591479621,
      "recall@0.5": 0.07379902529589233,
      "miss_rate": -0.11981898352285913,
      "wrong_peak_rate": 0.04601995822696682,
      "detected_ae_p95": 0.5384999999999889
    },
    "f1@0.5": {
      "mean_delta": 0.025122382921583763,
      "ci95": [
        0.021528559071963216,
        0.0288428024399646
      ]
    },
    "f1@0.1": {
      "mean_delta": 0.025934060780234545,
      "ci95": [
        0.021041238065820443,
        0.030950278340873435
      ]
    },
    "precision@0.5": {
      "mean_delta": -0.03018054815823099,
      "ci95": [
        -0.034761332684182876,
        -0.025749291136186066
      ]
    },
    "recall@0.5": {
      "mean_delta": 0.07379264922637438,
      "ci95": [
        0.06946294181550902,
        0.07808781911534007
      ]
    },
    "miss_rate": {
      "mean_delta": -0.11985745866074182,
      "ci95": [
        -0.12583421256415397,
        -0.11399498869978089
      ]
    },
    "wrong_peak_rate": {
      "mean_delta": 0.04606480943436746,
      "ci95": [
        0.041276667868675086,
        0.05111167997834141
      ]
    },
    "detected_ae_p95": {
      "mean_delta": 0.553248999999999,
      "ci95": [
        0.21592499999998363,
        0.9415625000000131
      ]
    }
  },
  "fixed_UNION_vs_PhaseNet-SCEDC": {
    "n_boot": 5000,
    "seed": 20260817,
    "definition": "delta = fixed_UNION - comparator",
    "point_delta": {
      "f1@0.5": 0.11097287275538759,
      "f1@0.1": 0.16043322606373983,
      "precision@0.5": 0.0843406076985862,
      "recall@0.5": 0.13578556509631,
      "miss_rate": -0.06832211650034811,
      "wrong_peak_rate": -0.06746344859596193,
      "detected_ae_p95": -56.19200000000004
    },
    "f1@0.5": {
      "mean_delta": 0.1109922343172051,
      "ci95": [
        0.10598497215100565,
        0.11613820616423204
      ]
    },
    "f1@0.1": {
      "mean_delta": 0.16046658398418803,
      "ci95": [
        0.15432289047048592,
        0.1665386713083254
      ]
    },
    "precision@0.5": {
      "mean_delta": 0.08434502984637772,
      "ci95": [
        0.07913188213976542,
        0.08956242382839148
      ]
    },
    "recall@0.5": {
      "mean_delta": 0.1358160923255305,
      "ci95": [
        0.1305495603626926,
        0.1411591352334533
      ]
    },
    "miss_rate": {
      "mean_delta": -0.06836195930284927,
      "ci95": [
        -0.07261005548973067,
        -0.06421049502952894
      ]
    },
    "wrong_peak_rate": {
      "mean_delta": -0.0674541330226812,
      "ci95": [
        -0.07283315400596216,
        -0.062312324798744975
      ]
    },
    "detected_ae_p95": {
      "mean_delta": -56.20353019999997,
      "ci95": [
        -58.410087499999996,
        -53.97995
      ]
    }
  },
  "fixed_UNION_vs_STEAD_top1_frozen": {
    "n_boot": 5000,
    "seed": 20260817,
    "point_delta": {
      "f1@0.5": 0.019749361800881826,
      "f1@0.1": 0.01828730563935954,
      "precision@0.5": 0.019749361800881826,
      "recall@0.5": 0.019749361800881826,
      "miss_rate": 0.0,
      "wrong_peak_rate": -0.01974936180088188,
      "detected_ae_p95": -3.634500000000043
    },
    "f1@0.5": {
      "mean_delta": 0.019753944620016773,
      "ci95": [
        0.01734504340402719,
        0.022288600930271577
      ]
    },
    "f1@0.1": {
      "mean_delta": 0.01827875877342925,
      "ci95": [
        0.01575205414872332,
        0.020750057732412158
      ]
    },
    "precision@0.5": {
      "mean_delta": 0.01975394462001677,
      "ci95": [
        0.017345043404027188,
        0.022288600930271574
      ]
    },
    "recall@0.5": {
      "mean_delta": 0.01975394462001677,
      "ci95": [
        0.017345043404027188,
        0.022288600930271574
      ]
    },
    "miss_rate": {
      "mean_delta": 0.0,
      "ci95": [
        0.0,
        0.0
      ]
    },
    "wrong_peak_rate": {
      "mean_delta": -0.019753944620016766,
      "ci95": [
        -0.022288600930271546,
        -0.017345043404027215
      ]
    },
    "detected_ae_p95": {
      "mean_delta": -3.570240599999985,
      "ci95": [
        -6.010087499999999,
        -1.9314625000000187
      ]
    }
  }
}
```

---

# 5. comparator_registry.json

- SeisBench: `0.12.3`
- new_valid_comparators: `['PhaseNet-ETHZ', 'PhaseNet-SCEDC']`
- n_new_valid: `2`
- proceed_gpu_batch: `True`
- created_utc: `2026-08-17T07:03:23.378824+00:00`

| model | valid | run_7A | dataset | leakage | sampling_rate | component_order | exclusion_reason | weight_sha256 |
|--|--|--|--|--|--|--|--|--|
| PhaseNet-STEAD | True | False | STEAD | False | 100 | ZNE | frozen_stage6_baseline | `761c46496139e991…` |
| PhaseNet-ETHZ | True | True | ETHZ | False | 100 | ZNE | new_external_comparator | `89cb5ff3272ea91d…` |
| PhaseNet-SCEDC | True | True | SCEDC | False | 100 | ZNE | new_external_comparator | `c71b8857cde02cd4…` |
| EQTransformer-STEAD | False | False | None | None | None | None | official_weight_download_failed_SSLError_network; local cache empty |  |
| EQTransformer-ETHZ | False | False | None | None | None | None | official_weight_download_failed_SSLError_network; local cache empty |  |
| PhaseNet-instance | False | False | None | True | None | None | INSTANCE-trained weights may include confirm events; leakage diagnostic only |  |
| EQTransformer-instance | False | False | None | True | None | None | INSTANCE-trained weights may include confirm events; leakage diagnostic only; weight unavailable locally |  |
| LFTNet | False | False | None | None | None | None | no local official package/checkpoint; incomplete reproducibility gates |  |
| PhaseNO | False | None | None | None | None | None | multi-station input; incompatible with single-station Stage-6 protocol |  |
| SegPhase | False | None | None | None | None | None | region/sampling/training protocol incompatible; no fully compatible official checkpoint verified |  |
| GreenPhase | False | None | None | None | None | None | no compatible official public implementation verified in this environment |  |
| PhaseNet-DAS | False | None | None | None | None | None | DAS modality incompatible |  |
| PhaseNet+ | False | None | None | None | None | None | multi-task objectives; not a dedicated picking comparator under our protocol |  |

## 5.1 JSON 全文

```json
{
  "created_utc": "2026-08-17T07:03:23.378824+00:00",
  "seisbench_version": "0.12.3",
  "threshold_grid_preregistered": [
    0.05,
    0.1,
    0.2,
    0.3,
    0.5,
    0.7
  ],
  "models": {
    "PhaseNet-STEAD": {
      "official_source": "seisbench.models.PhaseNet.from_pretrained('stead')",
      "weight_name": "stead",
      "weight_path": "/home/yehang/.seisbench/models/v3/phasenet/stead.pt.v2",
      "weight_sha256": "761c46496139e9917a7480ff56f660914124c806511ea794f0005518358b09ec",
      "training_dataset": "STEAD",
      "possible_confirm_leakage": false,
      "sampling_rate": 100,
      "component_order": "ZNE",
      "phases": "PSN",
      "valid_comparator": true,
      "role": "frozen_stage6_baseline",
      "run_in_stage7A": false,
      "reuse_stage6_confirm_preds": true,
      "exclusion_reason": null,
      "window_length_note": "SeisBench PhaseNet annotate default",
      "preprocessing": "SeisBench annotate + project UTC remap ENZ->ZNE / PSN",
      "license": "SeisBench model zoo / original dataset licenses"
    },
    "PhaseNet-ETHZ": {
      "official_source": "seisbench.models.PhaseNet.from_pretrained('ethz')",
      "weight_name": "ethz",
      "weight_path": "/home/yehang/.seisbench/models/v3/phasenet/ethz.pt.v2",
      "weight_sha256": "89cb5ff3272ea91df7d6a3c347fdd067f5a6c2aab91aa884c448a93a482cd848",
      "training_dataset": "ETHZ",
      "possible_confirm_leakage": false,
      "sampling_rate": 100,
      "component_order": "ZNE",
      "phases": "PSN",
      "valid_comparator": true,
      "role": "new_external_comparator",
      "run_in_stage7A": true,
      "default_S_threshold_official": 0.34493361542513395,
      "threshold_selection": "preregistered_grid_on_stage6_dev_only",
      "exclusion_reason": null,
      "window_length_note": "SeisBench PhaseNet annotate default",
      "preprocessing": "SeisBench annotate + project UTC remap ENZ->ZNE / PSN",
      "license": "SeisBench model zoo / original dataset licenses"
    },
    "PhaseNet-SCEDC": {
      "official_source": "seisbench.models.PhaseNet.from_pretrained('scedc')",
      "weight_name": "scedc",
      "weight_path": "/home/yehang/.seisbench/models/v3/phasenet/scedc.pt.v2",
      "weight_sha256": "c71b8857cde02cd4d0ab6e3d5e429b16ae82cfd9e39928bef9ec229a4a64502c",
      "training_dataset": "SCEDC",
      "possible_confirm_leakage": false,
      "sampling_rate": 100,
      "component_order": "ZNE",
      "phases": "PSN",
      "valid_comparator": true,
      "role": "new_external_comparator",
      "run_in_stage7A": true,
      "default_S_threshold_official": 0.4039903966026159,
      "threshold_selection": "preregistered_grid_on_stage6_dev_only",
      "exclusion_reason": null,
      "window_length_note": "SeisBench PhaseNet annotate default",
      "preprocessing": "SeisBench annotate + project UTC remap ENZ->ZNE / PSN",
      "license": "SeisBench model zoo / original dataset licenses"
    },
    "EQTransformer-STEAD": {
      "official_source": "seisbench.models.EQTransformer.from_pretrained('stead')",
      "weight_name": "stead",
      "valid_comparator": false,
      "run_in_stage7A": false,
      "exclusion_reason": "official_weight_download_failed_SSLError_network; local cache empty"
    },
    "EQTransformer-ETHZ": {
      "official_source": "seisbench.models.EQTransformer.from_pretrained('ethz')",
      "valid_comparator": false,
      "run_in_stage7A": false,
      "exclusion_reason": "official_weight_download_failed_SSLError_network; local cache empty"
    },
    "PhaseNet-instance": {
      "valid_comparator": false,
      "role": "diagnostic_only_data_leakage",
      "run_in_stage7A": false,
      "possible_confirm_leakage": true,
      "exclusion_reason": "INSTANCE-trained weights may include confirm events; leakage diagnostic only"
    },
    "EQTransformer-instance": {
      "valid_comparator": false,
      "role": "diagnostic_only_data_leakage",
      "run_in_stage7A": false,
      "possible_confirm_leakage": true,
      "exclusion_reason": "INSTANCE-trained weights may include confirm events; leakage diagnostic only; weight unavailable locally"
    },
    "LFTNet": {
      "valid_comparator": false,
      "LFTNet_status": "not_reproducible_from_official_release",
      "run_in_stage7A": false,
      "exclusion_reason": "no local official package/checkpoint; incomplete reproducibility gates"
    },
    "PhaseNO": {
      "valid_comparator": false,
      "exclusion_reason": "multi-station input; incompatible with single-station Stage-6 protocol"
    },
    "SegPhase": {
      "valid_comparator": false,
      "exclusion_reason": "region/sampling/training protocol incompatible; no fully compatible official checkpoint verified"
    },
    "GreenPhase": {
      "valid_comparator": false,
      "exclusion_reason": "no compatible official public implementation verified in this environment"
    },
    "PhaseNet-DAS": {
      "valid_comparator": false,
      "exclusion_reason": "DAS modality incompatible"
    },
    "PhaseNet+": {
      "valid_comparator": false,
      "exclusion_reason": "multi-task objectives; not a dedicated picking comparator under our protocol"
    }
  },
  "n_new_valid_comparators": 2,
  "new_valid_comparators": [
    "PhaseNet-ETHZ",
    "PhaseNet-SCEDC"
  ],
  "proceed_gpu_batch": true
}
```

---

# 6. runtime_scaling.json

| setting | n_gpus | traces/s median | min | max | status | gpus |
|--|--:|--:|--:|--:|--|--|
| 1gpu | 1 | 6.458 | 6.409 | 6.463 | nan | [0] |
| 4gpu | 4 | 24.404 | 24.312 | 24.770 | nan | [0, 1, 2, 3] |
| 8gpu |  |  |  |  | insufficient_free_gpus | [0, 1, 2, 3, 4, 5, 6] |

```json
[
  {
    "setting": "1gpu",
    "n_gpus": 1.0,
    "gpus": "[0]",
    "traces_per_s_median": 6.457563288972597,
    "traces_per_s_min": 6.409167314625249,
    "traces_per_s_max": 6.463484995872554,
    "weight": "ethz",
    "n_timed_per_rep": 400.0,
    "warmup": 200.0,
    "reps": 3.0,
    "requested": NaN,
    "actual_gpus": NaN,
    "status": NaN
  },
  {
    "setting": "4gpu",
    "n_gpus": 4.0,
    "gpus": "[0, 1, 2, 3]",
    "traces_per_s_median": 24.40367963941658,
    "traces_per_s_min": 24.311795259389285,
    "traces_per_s_max": 24.76973528542771,
    "weight": "ethz",
    "n_timed_per_rep": 400.0,
    "warmup": 200.0,
    "reps": 3.0,
    "requested": NaN,
    "actual_gpus": NaN,
    "status": NaN
  },
  {
    "setting": "8gpu",
    "n_gpus": NaN,
    "gpus": "[0, 1, 2, 3, 4, 5, 6]",
    "traces_per_s_median": NaN,
    "traces_per_s_min": NaN,
    "traces_per_s_max": NaN,
    "weight": NaN,
    "n_timed_per_rep": NaN,
    "warmup": NaN,
    "reps": NaN,
    "requested": 8.0,
    "actual_gpus": 7.0,
    "status": "insufficient_free_gpus"
  }
]
```

---

# 7. runtime_benchmark.json（摘要 + 全文）

- free_gpus_at_start: `[0, 1, 2, 3, 4, 5, 6]`
- scaling_ratios: `{"scale_1_to_4": 3.77908485714574, "scale_4_to_8": null, "scale_4_to_8_note": "8 free GPUs unavailable", "scale_4_to_7_proxy": 1.6514350404994786}`
- projections: `{"reference_setting": "4gpu", "traces_per_s": 24.40367963941658, "eta_100k_traces_h": 1.1382618600234118, "eta_confirm_74753_traces_h": 0.8508848882233009, "eta_full_INSTANCE_1159249_traces_h": 13.1952892297028}`
- hdf5_io_bottleneck_likely: `False`
- near_8_gpu_probe: `{"n_gpus": 7, "gpus": [0, 1, 2, 3, 4, 5, 6], "traces_per_s": 40.301091673656224, "hdf5_read_fraction_median": 0.5267954928571148, "note": "GPU7 busy; 8 free GPUs unavailable. 7-GPU probe for I/O saturation."}`
- sota_claim_allowed: `False`

## 7.1 actual_settings

```json
[
  {
    "setting": "1gpu",
    "n_gpus": 1,
    "gpus": [
      0
    ],
    "traces_per_s_median": 6.457563288972597,
    "traces_per_s_min": 6.409167314625249,
    "traces_per_s_max": 6.463484995872554,
    "weight": "ethz",
    "n_timed_per_rep": 400,
    "warmup": 200,
    "reps": 3
  },
  {
    "setting": "4gpu",
    "n_gpus": 4,
    "gpus": [
      0,
      1,
      2,
      3
    ],
    "traces_per_s_median": 24.40367963941658,
    "traces_per_s_min": 24.311795259389285,
    "traces_per_s_max": 24.76973528542771,
    "weight": "ethz",
    "n_timed_per_rep": 400,
    "warmup": 200,
    "reps": 3
  },
  {
    "setting": "8gpu",
    "requested": 8,
    "actual_gpus": 7,
    "status": "insufficient_free_gpus",
    "gpus": [
      0,
      1,
      2,
      3,
      4,
      5,
      6
    ]
  }
]
```

## 7.2 JSON 全文（含 runs 明细）

```json
{
  "created_utc": "2026-08-17T07:03:21.357348+00:00",
  "runtime_manifest_sha256": "70635b5ddba325428aba3af471c9ac0f50443a4f545fb7982db07d5cdf7c3ff7",
  "free_gpus_at_start": [
    0,
    1,
    2,
    3,
    4,
    5,
    6
  ],
  "requested_gpu_counts": [
    1,
    4,
    8
  ],
  "actual_settings": [
    {
      "setting": "1gpu",
      "n_gpus": 1,
      "gpus": [
        0
      ],
      "traces_per_s_median": 6.457563288972597,
      "traces_per_s_min": 6.409167314625249,
      "traces_per_s_max": 6.463484995872554,
      "weight": "ethz",
      "n_timed_per_rep": 400,
      "warmup": 200,
      "reps": 3
    },
    {
      "setting": "4gpu",
      "n_gpus": 4,
      "gpus": [
        0,
        1,
        2,
        3
      ],
      "traces_per_s_median": 24.40367963941658,
      "traces_per_s_min": 24.311795259389285,
      "traces_per_s_max": 24.76973528542771,
      "weight": "ethz",
      "n_timed_per_rep": 400,
      "warmup": 200,
      "reps": 3
    },
    {
      "setting": "8gpu",
      "requested": 8,
      "actual_gpus": 7,
      "status": "insufficient_free_gpus",
      "gpus": [
        0,
        1,
        2,
        3,
        4,
        5,
        6
      ]
    }
  ],
  "scaling_ratios": {
    "scale_1_to_4": 3.77908485714574,
    "scale_4_to_8": null,
    "scale_4_to_8_note": "8 free GPUs unavailable",
    "scale_4_to_7_proxy": 1.6514350404994786
  },
  "projections": {
    "reference_setting": "4gpu",
    "traces_per_s": 24.40367963941658,
    "eta_100k_traces_h": 1.1382618600234118,
    "eta_confirm_74753_traces_h": 0.8508848882233009,
    "eta_full_INSTANCE_1159249_traces_h": 13.1952892297028
  },
  "hdf5_io_bottleneck_likely": false,
  "runs": [
    {
      "n_gpus": 1,
      "gpus": [
        0
      ],
      "wall_clock_s_includes_warmup": 95.68913374701515,
      "n_timed_total": 400,
      "traces_per_s_aggregate": 6.457563288972597,
      "traces_per_s_wall_includes_warmup": 4.180202958650749,
      "per_gpu": [
        {
          "physical_gpu": 0,
          "weight": "ethz",
          "mode": "e2e",
          "warmup": 200,
          "n_timed": 400,
          "wall_clock_s": 61.94286948500667,
          "traces_per_s_e2e": 6.457563288972597,
          "sections": {
            "hdf5_read_s": 35.435696262982674,
            "preprocess_s": NaN,
            "model_annotate_s": 25.22412528214045,
            "candidate_extraction_s": 25.22412528214045,
            "cache_serialization_s": 0.031159859616309404,
            "hdf5_read_tps": 11.288052505909233,
            "candidate_extraction_tps": 15.857834336210415
          },
          "gpu_util_median": 0.0,
          "gpu_util_p95": 0.0,
          "gpu_mem_peak_mib_smi": 545.0,
          "gpu_mem_peak_mib_torch": 5.31494140625,
          "batch_size_note": "SeisBench annotate default (single-stream per call; official path)"
        }
      ],
      "events_per_s_approx": 7.161807714202511,
      "hdf5_read_fraction_median": 0.5841707964245009,
      "label": "1gpu",
      "rep": 0
    },
    {
      "n_gpus": 1,
      "gpus": [
        0
      ],
      "wall_clock_s_includes_warmup": 95.468248951016,
      "n_timed_total": 400,
      "traces_per_s_aggregate": 6.463484995872554,
      "traces_per_s_wall_includes_warmup": 4.189874690225405,
      "per_gpu": [
        {
          "physical_gpu": 0,
          "weight": "ethz",
          "mode": "e2e",
          "warmup": 200,
          "n_timed": 400,
          "wall_clock_s": 61.88611875101924,
          "traces_per_s_e2e": 6.463484995872554,
          "sections": {
            "hdf5_read_s": 35.40261814126279,
            "preprocess_s": NaN,
            "model_annotate_s": 25.204304945655167,
            "candidate_extraction_s": 25.204304945655167,
            "cache_serialization_s": 0.029303858638741076,
            "hdf5_read_tps": 11.298599397477563,
            "candidate_extraction_tps": 15.870304730182763
          },
          "gpu_util_median": 0.0,
          "gpu_util_p95": 0.0,
          "gpu_mem_peak_mib_smi": 545.0,
          "gpu_mem_peak_mib_torch": 5.31494140625,
          "batch_size_note": "SeisBench annotate default (single-stream per call; official path)"
        }
      ],
      "events_per_s_approx": 7.168375226476031,
      "hdf5_read_fraction_median": 0.5841348865457331,
      "label": "1gpu",
      "rep": 1
    },
    {
      "n_gpus": 1,
      "gpus": [
        0
      ],
      "wall_clock_s_includes_warmup": 96.00826460903045,
      "n_timed_total": 400,
      "traces_per_s_aggregate": 6.409167314625249,
      "traces_per_s_wall_includes_warmup": 4.166307990555809,
      "per_gpu": [
        {
          "physical_gpu": 0,
          "weight": "ethz",
          "mode": "e2e",
          "warmup": 200,
          "n_timed": 400,
          "wall_clock_s": 62.41060349403415,
          "traces_per_s_e2e": 6.409167314625249,
          "sections": {
            "hdf5_read_s": 35.72580802068114,
            "preprocess_s": NaN,
            "model_annotate_s": 25.399334496003576,
            "candidate_extraction_s": 25.399334496003576,
            "cache_serialization_s": 0.029812399297952652,
            "hdf5_read_tps": 11.196387770108542,
            "candidate_extraction_tps": 15.748444120177144
          },
          "gpu_util_median": 0.0,
          "gpu_util_p95": 0.0,
          "gpu_mem_peak_mib_smi": 545.0,
          "gpu_mem_peak_mib_torch": 5.31494140625,
          "batch_size_note": "SeisBench annotate default (single-stream per call; official path)"
        }
      ],
      "events_per_s_approx": 7.108133805499352,
      "hdf5_read_fraction_median": 0.5844699341343773,
      "label": "1gpu",
      "rep": 2
    },
    {
      "n_gpus": 4,
      "gpus": [
        0,
        1,
        2,
        3
      ],
      "wall_clock_s_includes_warmup": 52.76584503205959,
      "n_timed_total": 400,
      "traces_per_s_aggregate": 24.76973528542771,
      "traces_per_s_wall_includes_warmup": 7.580661311440519,
      "per_gpu": [
        {
          "physical_gpu": 0,
          "weight": "ethz",
          "mode": "e2e",
          "warmup": 200,
          "n_timed": 100,
          "wall_clock_s": 16.238901191041805,
          "traces_per_s_e2e": 6.158052125790693,
          "sections": {
            "hdf5_read_s": 8.856901498278603,
            "preprocess_s": NaN,
            "model_annotate_s": 7.0417670557508245,
            "candidate_extraction_s": 7.0417670557508245,
            "cache_serialization_s": 0.01288207178004086,
            "hdf5_read_tps": 11.29063025251389,
            "candidate_extraction_tps": 14.200980976548017
          },
          "gpu_util_median": 0.0,
          "gpu_util_p95": 0.0,
          "gpu_mem_peak_mib_smi": 545.0,
          "gpu_mem_peak_mib_torch": 5.31494140625,
          "batch_size_note": "SeisBench annotate default (single-stream per call; official path)"
        },
        {
          "physical_gpu": 1,
          "weight": "ethz",
          "mode": "e2e",
          "warmup": 200,
          "n_timed": 100,
          "wall_clock_s": 16.33817211096175,
          "traces_per_s_e2e": 6.120635730903283,
          "sections": {
            "hdf5_read_s": 8.833976121735759,
            "preprocess_s": NaN,
            "model_annotate_s": 7.170375798945315,
            "candidate_extraction_s": 7.170375798945315,
            "cache_serialization_s": 0.010656876023858786,
            "hdf5_read_tps": 11.319930982601674,
            "candidate_extraction_tps": 13.94627043323293
          },
          "gpu_util_median": 0.0,
          "gpu_util_p95": 0.0,
          "gpu_mem_peak_mib_smi": 506.0,
          "gpu_mem_peak_mib_torch": 5.31494140625,
          "batch_size_note": "SeisBench annotate default (single-stream per call; official path)"
        },
        {
          "physical_gpu": 2,
          "weight": "ethz",
          "mode": "e2e",
          "warmup": 200,
          "n_timed": 100,
          "wall_clock_s": 16.233471236075275,
          "traces_per_s_e2e": 6.16011194067799,
          "sections": {
            "hdf5_read_s": 8.825641137664206,
            "preprocess_s": NaN,
            "model_annotate_s": 7.0661216175649315,
            "candidate_extraction_s": 7.0661216175649315,
            "cache_serialization_s": 0.011918811709620059,
            "hdf5_read_tps": 11.330621587732718,
            "candidate_extraction_tps": 14.152034936876897
          },
          "gpu_util_median": 0.0,
          "gpu_util_p95": 0.0,
          "gpu_mem_peak_mib_smi": 506.0,
          "gpu_mem_peak_mib_torch": 5.31494140625,
          "batch_size_note": "SeisBench annotate default (single-stream per call; official path)"
        },
        {
          "physical_gpu": 3,
          "weight": "ethz",
          "mode": "e2e",
          "warmup": 200,
          "n_timed": 100,
          "wall_clock_s": 15.795453956001438,
          "traces_per_s_e2e": 6.3309354880557445,
          "sections": {
            "hdf5_read_s": 8.69756364019122,
            "preprocess_s": NaN,
            "model_annotate_s": 6.77179322775919,
            "candidate_extraction_s": 6.77179322775919,
            "cache_serialization_s": 0.009723330964334309,
            "hdf5_read_tps": 11.497472641407596,
            "candidate_extraction_tps": 14.76713724661235
          },
          "gpu_util_median": 0.0,
          "gpu_util_p95": 0.0,
          "gpu_mem_peak_mib_smi": 506.0,
          "gpu_mem_peak_mib_torch": 5.31494140625,
          "batch_size_note": "SeisBench annotate default (single-stream per call; official path)"
        }
      ],
      "events_per_s_approx": 27.471055769420754,
      "hdf5_read_fraction_median": 0.5562219810006469,
      "label": "4gpu",
      "rep": 0
    },
    {
      "n_gpus": 4,
      "gpus": [
        0,
        1,
        2,
        3
      ],
      "wall_clock_s_includes_warmup": 53.16413718101103,
      "n_timed_total": 400,
      "traces_per_s_aggregate": 24.311795259389285,
      "traces_per_s_wall_includes_warmup": 7.523868931383138,
      "per_gpu": [
        {
          "physical_gpu": 0,
          "weight": "ethz",
          "mode": "e2e",
          "warmup": 200,
          "n_timed": 100,
          "wall_clock_s": 16.46705006493721,
          "traces_per_s_e2e": 6.072733100686136,
          "sections": {
            "hdf5_read_s": 8.703958938713185,
            "preprocess_s": NaN,
            "model_annotate_s": 7.439002021332271,
            "candidate_extraction_s": 7.439002021332271,
            "cache_serialization_s": 0.011777457664720714,
            "hdf5_read_tps": 11.489024787929921,
            "candidate_extraction_tps": 13.4426633724843
          },
          "gpu_util_median": 0.0,
          "gpu_util_p95": 0.0,
          "gpu_mem_peak_mib_smi": 545.0,
          "gpu_mem_peak_mib_torch": 5.31494140625,
          "batch_size_note": "SeisBench annotate default (single-stream per call; official path)"
        },
        {
          "physical_gpu": 1,
          "weight": "ethz",
          "mode": "e2e",
          "warmup": 200,
          "n_timed": 100,
          "wall_clock_s": 16.41171985899564,
          "traces_per_s_e2e": 6.093206614490663,
          "sections": {
            "hdf5_read_s": 8.685862259357236,
            "preprocess_s": NaN,
            "model_annotate_s": 7.374845048994757,
            "candidate_extraction_s": 7.374845048994757,
            "cache_serialization_s": 0.009607083746232092,
            "hdf5_read_tps": 11.512961754864406,
            "candidate_extraction_tps": 13.559606925386277
          },
          "gpu_util_median": 0.0,
          "gpu_util_p95": 0.0,
          "gpu_mem_peak_mib_smi": 506.0,
          "gpu_mem_peak_mib_torch": 5.31494140625,
          "batch_size_note": "SeisBench annotate default (single-stream per call; official path)"
        },
        {
          "physical_gpu": 2,
          "weight": "ethz",
          "mode": "e2e",
          "warmup": 200,
          "n_timed": 100,
          "wall_clock_s": 16.51491205196362,
          "traces_per_s_e2e": 6.055133668611334,
          "sections": {
            "hdf5_read_s": 8.688394205993973,
            "preprocess_s": NaN,
            "model_annotate_s": 7.50403852830641,
            "candidate_extraction_s": 7.50403852830641,
            "cache_serialization_s": 0.010190301458351314,
            "hdf5_read_tps": 11.509606680945915,
            "candidate_extraction_tps": 13.326157591380204
          },
          "gpu_util_median": 0.0,
          "gpu_util_p95": 0.0,
          "gpu_mem_peak_mib_smi": 506.0,
          "gpu_mem_peak_mib_torch": 5.31494140625,
          "batch_size_note": "SeisBench annotate default (single-stream per call; official path)"
        },
        {
          "physical_gpu": 3,
          "weight": "ethz",
          "mode": "e2e",
          "warmup": 200,
          "n_timed": 100,
          "wall_clock_s": 16.418415097985417,
          "traces_per_s_e2e": 6.090721875601152,
          "sections": {
            "hdf5_read_s": 8.762858371832408,
            "preprocess_s": NaN,
            "model_annotate_s": 7.31132810364943,
            "candidate_extraction_s": 7.31132810364943,
            "cache_serialization_s": 0.011503987363539636,
            "hdf5_read_tps": 11.411801464399216,
            "candidate_extraction_tps": 13.6774056070723
          },
          "gpu_util_median": 0.0,
          "gpu_util_p95": 0.0,
          "gpu_mem_peak_mib_smi": 506.0,
          "gpu_mem_peak_mib_torch": 5.31494140625,
          "batch_size_note": "SeisBench annotate default (single-stream per call; official path)"
        }
      ],
      "events_per_s_approx": 26.963174039988118,
      "hdf5_read_fraction_median": 0.5399971294897818,
      "label": "4gpu",
      "rep": 1
    },
    {
      "n_gpus": 4,
      "gpus": [
        0,
        1,
        2,
        3
      ],
      "wall_clock_s_includes_warmup": 53.06808759307023,
      "n_timed_total": 400,
      "traces_per_s_aggregate": 24.40367963941658,
      "traces_per_s_wall_includes_warmup": 7.5374866165750625,
      "per_gpu": [
        {
          "physical_gpu": 0,
          "weight": "ethz",
          "mode": "e2e",
          "warmup": 200,
          "n_timed": 100,
          "wall_clock_s": 16.555015010992065,
          "traces_per_s_e2e": 6.040465679650716,
          "sections": {
            "hdf5_read_s": 8.810750396340154,
            "preprocess_s": NaN,
            "model_annotate_s": 7.412147805094719,
            "candidate_extraction_s": 7.412147805094719,
            "cache_serialization_s": 0.012983551830984652,
            "hdf5_read_tps": 11.349771075292113,
            "candidate_extraction_tps": 13.491366150478715
          },
          "gpu_util_median": 0.0,
          "gpu_util_p95": 0.0,
          "gpu_mem_peak_mib_smi": 545.0,
          "gpu_mem_peak_mib_torch": 5.31494140625,
          "batch_size_note": "SeisBench annotate default (single-stream per call; official path)"
        },
        {
          "physical_gpu": 1,
          "weight": "ethz",
          "mode": "e2e",
          "warmup": 200,
          "n_timed": 100,
          "wall_clock_s": 16.543603939935565,
          "traces_per_s_e2e": 6.044632134755366,
          "sections": {
            "hdf5_read_s": 8.7681159817148,
            "preprocess_s": NaN,
            "model_annotate_s": 7.446828856365755,
            "candidate_extraction_s": 7.446828856365755,
            "cache_serialization_s": 0.01113775942940265,
            "hdf5_read_tps": 11.404958626065389,
            "candidate_extraction_tps": 13.428534739927215
          },
          "gpu_util_median": 0.0,
          "gpu_util_p95": 0.0,
          "gpu_mem_peak_mib_smi": 506.0,
          "gpu_mem_peak_mib_torch": 5.31494140625,
          "batch_size_note": "SeisBench annotate default (single-stream per call; official path)"
        },
        {
          "physical_gpu": 2,
          "weight": "ethz",
          "mode": "e2e",
          "warmup": 200,
          "n_timed": 100,
          "wall_clock_s": 16.238840749952942,
          "traces_per_s_e2e": 6.158075046107573,
          "sections": {
            "hdf5_read_s": 8.721435221727006,
            "preprocess_s": NaN,
            "model_annotate_s": 7.194225041079335,
            "candidate_extraction_s": 7.194225041079335,
            "cache_serialization_s": 0.009455821826122701,
            "hdf5_read_tps": 11.466002722909423,
            "candidate_extraction_tps": 13.900037798233402
          },
          "gpu_util_median": 0.0,
          "gpu_util_p95": 0.0,
          "gpu_mem_peak_mib_smi": 506.0,
          "gpu_mem_peak_mib_torch": 5.31494140625,
          "batch_size_note": "SeisBench annotate default (single-stream per call; official path)"
        },
        {
          "physical_gpu": 3,
          "weight": "ethz",
          "mode": "e2e",
          "warmup": 200,
          "n_timed": 100,
          "wall_clock_s": 16.232430803007446,
          "traces_per_s_e2e": 6.160506778902923,
          "sections": {
            "hdf5_read_s": 8.715469123679213,
            "preprocess_s": NaN,
            "model_annotate_s": 7.193061173311435,
            "candidate_extraction_s": 7.193061173311435,
            "cache_serialization_s": 0.0104003029409796,
            "hdf5_read_tps": 11.473851674640006,
            "candidate_extraction_tps": 13.902286883230202
          },
          "gpu_util_median": 0.0,
          "gpu_util_p95": 0.0,
          "gpu_mem_peak_mib_smi": 506.0,
          "gpu_mem_peak_mib_torch": 5.31494140625,
          "batch_size_note": "SeisBench annotate default (single-stream per call; official path)"
        }
      ],
      "events_per_s_approx": 27.065079082532254,
      "hdf5_read_fraction_median": 0.5454773055132214,
      "label": "4gpu",
      "rep": 2
    },
    {
      "n_gpus": 7,
      "gpus": [
        0,
        1,
        2,
        3,
        4,
        5,
        6
      ],
      "wall_clock_s_includes_warmup": 49.57822017907165,
      "n_timed_total": 400,
      "traces_per_s_aggregate": 40.301091673656224,
      "traces_per_s_wall_includes_warmup": 8.06805888866602,
      "per_gpu": [
        {
          "physical_gpu": 0,
          "weight": "ethz",
          "mode": "e2e",
          "warmup": 200,
          "n_timed": 58,
          "wall_clock_s": 10.280123557895422,
          "traces_per_s_e2e": 5.641955534226471,
          "sections": {
            "hdf5_read_s": 5.166141907218844,
            "preprocess_s": NaN,
            "model_annotate_s": 4.872754485113546,
            "candidate_extraction_s": 4.872754485113546,
            "cache_serialization_s": 0.0085771051235497,
            "hdf5_read_tps": 11.226946731555017,
            "candidate_extraction_tps": 11.902918601212567
          },
          "gpu_util_median": 0.0,
          "gpu_util_p95": 0.0,
          "gpu_mem_peak_mib_smi": 545.0,
          "gpu_mem_peak_mib_torch": 5.31494140625,
          "batch_size_note": "SeisBench annotate default (single-stream per call; official path)"
        },
        {
          "physical_gpu": 1,
          "weight": "ethz",
          "mode": "e2e",
          "warmup": 200,
          "n_timed": 57,
          "wall_clock_s": 9.867716624983586,
          "traces_per_s_e2e": 5.776412331874681,
          "sections": {
            "hdf5_read_s": 5.054294073139317,
            "preprocess_s": NaN,
            "model_annotate_s": 4.569867939222604,
            "candidate_extraction_s": 4.569867939222604,
            "cache_serialization_s": 0.007870826520957053,
            "hdf5_read_tps": 11.277539291376495,
            "candidate_extraction_tps": 12.47300813898278
          },
          "gpu_util_median": 0.0,
          "gpu_util_p95": 0.0,
          "gpu_mem_peak_mib_smi": 506.0,
          "gpu_mem_peak_mib_torch": 5.31494140625,
          "batch_size_note": "SeisBench annotate default (single-stream per call; official path)"
        },
        {
          "physical_gpu": 2,
          "weight": "ethz",
          "mode": "e2e",
          "warmup": 200,
          "n_timed": 57,
          "wall_clock_s": 9.82158900599461,
          "traces_per_s_e2e": 5.803541561880672,
          "sections": {
            "hdf5_read_s": 5.045437103253789,
            "preprocess_s": NaN,
            "model_annotate_s": 4.5321640183683485,
            "candidate_extraction_s": 4.5321640183683485,
            "cache_serialization_s": 0.006364038796164095,
            "hdf5_read_tps": 11.29733635233325,
            "candidate_extraction_tps": 12.576773428539974
          },
          "gpu_util_median": 0.0,
          "gpu_util_p95": 0.0,
          "gpu_mem_peak_mib_smi": 506.0,
          "gpu_mem_peak_mib_torch": 5.31494140625,
          "batch_size_note": "SeisBench annotate default (single-stream per call; official path)"
        },
        {
          "physical_gpu": 3,
          "weight": "ethz",
          "mode": "e2e",
          "warmup": 200,
          "n_timed": 57,
          "wall_clock_s": 9.830335502978414,
          "traces_per_s_e2e": 5.798377886769992,
          "sections": {
            "hdf5_read_s": 5.103780649835244,
            "preprocess_s": NaN,
            "model_annotate_s": 4.477987816208042,
            "candidate_extraction_s": 4.477987816208042,
            "cache_serialization_s": 0.0054547470062971115,
            "hdf5_read_tps": 11.16819156439257,
            "candidate_extraction_tps": 12.728931461959084
          },
          "gpu_util_median": 0.0,
          "gpu_util_p95": 0.0,
          "gpu_mem_peak_mib_smi": 506.0,
          "gpu_mem_peak_mib_torch": 5.31494140625,
          "batch_size_note": "SeisBench annotate default (single-stream per call; official path)"
        },
        {
          "physical_gpu": 4,
          "weight": "ethz",
          "mode": "e2e",
          "warmup": 200,
          "n_timed": 57,
          "wall_clock_s": 10.006708692060784,
          "traces_per_s_e2e": 5.69617860917878,
          "sections": {
            "hdf5_read_s": 5.109411076176912,
            "preprocess_s": NaN,
            "model_annotate_s": 4.6579986901488155,
            "candidate_extraction_s": 4.6579986901488155,
            "cache_serialization_s": 0.006367696914821863,
            "hdf5_read_tps": 11.155884533496947,
            "candidate_extraction_tps": 12.237015034063683
          },
          "gpu_util_median": 0.0,
          "gpu_util_p95": 0.0,
          "gpu_mem_peak_mib_smi": 506.0,
          "gpu_mem_peak_mib_torch": 5.31494140625,
          "batch_size_note": "SeisBench annotate default (single-stream per call; official path)"
        },
        {
          "physical_gpu": 5,
          "weight": "ethz",
          "mode": "e2e",
          "warmup": 200,
          "n_timed": 57,
          "wall_clock_s": 9.84812013793271,
          "traces_per_s_e2e": 5.787906646309991,
          "sections": {
            "hdf5_read_s": 5.158713999553584,
            "preprocess_s": NaN,
            "model_annotate_s": 4.424721390008926,
            "candidate_extraction_s": 4.424721390008926,
            "cache_serialization_s": 0.0074032609118148685,
            "hdf5_read_tps": 11.049265379885874,
            "candidate_extraction_tps": 12.882167028348197
          },
          "gpu_util_median": 0.0,
          "gpu_util_p95": 0.0,
          "gpu_mem_peak_mib_smi": 506.0,
          "gpu_mem_peak_mib_torch": 5.31494140625,
          "batch_size_note": "SeisBench annotate default (single-stream per call; official path)"
        },
        {
          "physical_gpu": 6,
          "weight": "ethz",
          "mode": "e2e",
          "warmup": 200,
          "n_timed": 57,
          "wall_clock_s": 9.833148541976698,
          "traces_per_s_e2e": 5.796719103415643,
          "sections": {
            "hdf5_read_s": 5.06893430580385,
            "preprocess_s": NaN,
            "model_annotate_s": 4.527241978095844,
            "candidate_extraction_s": 4.527241978095844,
            "cache_serialization_s": 0.008162632118910551,
            "hdf5_read_tps": 11.244967198477182,
            "candidate_extraction_tps": 12.590446959933468
          },
          "gpu_util_median": 0.0,
          "gpu_util_p95": 0.0,
          "gpu_mem_peak_mib_smi": 506.0,
          "gpu_mem_peak_mib_torch": 5.31494140625,
          "batch_size_note": "SeisBench annotate default (single-stream per call; official path)"
        }
      ],
      "events_per_s_approx": 44.69621997078325,
      "hdf5_read_fraction_median": 0.5267954928571148,
      "label": "7gpu_proxy_for_8",
      "rep": 0
    }
  ],
  "sota_claim_allowed": false,
  "near_8_gpu_probe": {
    "n_gpus": 7,
    "gpus": [
      0,
      1,
      2,
      3,
      4,
      5,
      6
    ],
    "traces_per_s": 40.301091673656224,
    "hdf5_read_fraction_median": 0.5267954928571148,
    "note": "GPU7 busy; 8 free GPUs unavailable. 7-GPU probe for I/O saturation."
  }
}
```

