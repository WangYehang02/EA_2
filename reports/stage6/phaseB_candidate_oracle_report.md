# Stage 6 Phase B — Candidate Oracle Report

**Decision:** `keep_ida_for_union`  
**Keep ID-A as secondary source:** `True`  
**Recommended candidate source / K:** `UNION_STEAD5_IDA5` / `10`  
**Ranker may start:** `false`（本阶段结束，等待下一步指令）  
**Confirm sealed:** `true`

## Startup / eval scope

| 项 | 值 |
|--|--|
| ID-A best.pt | epoch 14, finite, BN≈STEAD, smoke OK |
| Eval | **full** `splits_full/stage6_dev` S-labelled |
| Events / S-traces | 5341 / 87293 |
| Manifest SHA256 | `2cc1e7a3376525ed46ff12d182f947602714a5b353e651e4d59542b56894c4e7` |
| GPUs used (events) | 8×4090（此前已缓存） |
| GPUs used (noise) | 4,6,7 |
| Confirm | sealed |

## Oracle（closest candidate；miss≈0 时 F1≈recall≈hit-rate）

| Set | F1@0.1 | F1@0.5 | miss | closest P95 | mean #cand |
|--|--:|--:|--:|--:|--:|
| STEAD K=5 | 0.5438 | 0.8617 | 0.0000 | 1.940 | 1.153 |
| STEAD K=10 | 0.5438 | 0.8617 | 0.0000 | 1.940 | 1.153 |
| ID-A K=5 | 0.5754 | 0.8717 | 0.0000 | 2.120 | 1.195 |
| ID-A K=10 | 0.5754 | 0.8717 | 0.0000 | 2.120 | 1.195 |
| Union K≤10 | 0.6429 | 0.8917 | 0.0000 | 1.350 | 1.806 |

**Fairness：** Union vs **STEAD K=10** ΔF1@0.5 = **+0.0300**（vs STEAD K=5 同为 +0.0300）。

说明：官方 peak 提取下单模型平均候选数约 1.15，故 STEAD K=5 与 K=10 数值几乎相同；Union 增益来自 **跨模型互补峰**，而非单模型扩大 K。

## Complementarity（K=10, ±0.5 s）

| class | traces | rate |
|--|--:|--:|
| both_correct | 73362 | 0.8404 |
| stead_only_correct | 1858 | 0.0213 |
| ida_only_correct | 2732 | 0.0313 |
| neither_correct | 9341 | 0.1070 |

- ID-A-only（相对 STEAD K=10）: {'tol': 0.5, 'k_compared': 10, 'n_ida_only_correct': 2732, 'rate': 0.031296896658380396, 'n_events': 1772}
- STEAD↔ID-A K5 候选 overlap@0.05s: 0.4700
- ida_only 是否过度集中: {'n_events_with_any': 1772, 'frac_traces_in_top5pct_events': 0.15300146412884333, 'frac_traces_in_top10_events': 0.023792093704245974}

## Bootstrap（event-level, 5000 reps）

- Union vs STEAD K=10 recall@0.5: mean Δ=+0.0336, 95% CI=[0.031859108095737096, 0.03532333216773072]
- Union vs STEAD K=10 recall@0.1: mean Δ=+0.0952, CI=[0.09258945565990125, 0.09789420889945916]
- STEAD K=10 vs K=5 recall@0.5: Δ=+0.0000

## Top-1（同列表；非主结论）

| | F1@0.5 | detected_ae_p95 | >10s | >30s |
|--|--:|--:|--:|--:|
| STEAD | 0.8403 | 5.920 | 4008 | 2424 |
| ID-A | 0.8390 | 24.948 | 5400 | 4106 |

- ID-A fixes STEAD@0.5: 3247；breaks: 3360；disagreement@0.05s: 0.5397
- **ID-A top-1 仍不推荐作主 picker**（P95 长尾恶化）；保留仅作 union 第二源。

## Noise audit（held-out 5000，与 train noise 不重叠）

| | S FPR@0.1 | P FPR@0.1 | mean #S cand |
|--|--:|--:|--:|
| STEAD | 0.0712 | 0.0420 | 1.019 |
| ID-A | 0.1168 | 0.0992 | 1.036 |
| Union mean # | — | — | 1.982 |

ID-A 相对 STEAD 候选数几乎不膨胀（+0.017）；S FPR 升幅 <0.2，通过 noise 门。

## Gates

```json
{
  "union_vs_stead_k10_delta_ge_0.01": true,
  "bootstrap_direction_stable": true,
  "ida_only_not_concentrated": true,
  "union_n_le_10": true,
  "noise_ok": true,
  "fine_tol_not_vanished": true
}
```

## Next（stopped）

- 推荐下一阶段候选源：`UNION_STEAD5_IDA5`，K≤10
- **不**自动启动 ranker / multi-station / offset
- **不**启封 confirm
- 等待用户下一步指令

## Artifacts

- `artifacts/results/stage6/phaseB_candidate_oracle.json`
- `artifacts/results/stage6/phaseB_candidate_complementarity.csv`
- `artifacts/results/stage6/phaseB_bootstrap.json`
- `artifacts/results/stage6/phaseB_noise_audit.json`
- `artifacts/results/stage6/phaseB_final_verdict.json`
- `artifacts/results/stage6/phaseA_revised_interpretation.json`
