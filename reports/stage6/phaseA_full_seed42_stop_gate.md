# Stage 6 Phase A stop-gate（full INSTANCE / seed42）

**依据：** 现有 `train_history.json` + `checkpoints/best.pt`（**未**重训、**未**跑满 140k `stage6_dev` annotate）。  
**训练状态：** epoch 22 后崩溃，无 `TRAIN.DONE`；有效最佳 **epoch 14**。

## 评测范围

| 项 | 值 |
|--|--|
| 对照 | 同子集 STEAD pretrained annotate vs ID-A best |
| 子集 | `annotate_eval_max_traces=1024`（`splits_full` / `stage6_dev` 头部） |
| 全量 dev | 140,428 traces — **未完成** |
| 其他 seed | 123 / 2026 — **未跑** |

> 旧 `baseline_stead/stage6_dev_stead_metrics.json` 属于 **pilot 10k splits**，不可与本次 full-split 子集直接比绝对值。

## 数字（ID-A − STEAD，同 1024 子集）

| 指标 | STEAD | ID-A best@14 | Δ |
|--|--:|--:|--:|
| S F1@0.5 | 0.6242 | 0.6349 | +0.0107 |
| S F1@0.1 | 0.4312 | 0.4264 | -0.0048 |
| detected_ae_p95 (s) | 6.047 | 22.577 | +16.530 |
| P F1@0.5 | 0.8291 | 0.8691 | +0.0400 |

命名提醒：`detected_ae_p95` ≠ 含 miss 的完整 e2e P95；本子集 miss_rate≈0。

## 停止门对照（kickoff）

Phase A：相对 STEAD，**dev** 上 F1@0.5 / F1@0.1 / K=5 oracle **任一** +≥0.01，**或** P95 降 ≥0.15 s；且 P/noise 可接受。

| 条件 | 结果 |
|--|--|
| Δ F1@0.5 ≥ 0.01 | **是**（+0.0107） |
| Δ F1@0.1 ≥ 0.01 | 否 |
| K=5 oracle +≥0.01 | **未评** |
| P95 降 ≥0.15 s | 否（反而恶化） |
| P 未明显变差 | 是（P F1 上升） |
| noise FPR | **未测** |

## 判定

**`marginal_pass_f1_0.5_timing_regressed`**

- **形式：** 在 1024 子集上，仅 F1@0.5 刚过 +0.01 线。  
- **实质：** 绝对时差尾部（detected_ae_p95）严重变差；不宜把 ID-A 升为唯一主 picker。  
- **推荐主 base picker：** **`STEAD`**  
- **ID-A 角色：** `optional_second_candidate_source_only_after_phaseB_oracle`

## 下一步（允许 / 禁止）

| 动作 | |
|--|--|
| 启封 `internal_confirm` | **禁止** |
| 启动 ranker / multi-station / offset | **禁止**（仍卡在 Phase A→B） |
| Phase B candidate oracle | **允许**：主源 STEAD；可选并入 ID-A `best.pt` 仅当 union oracle 在同一 full-split dev（或固定大子集）上相对单源 +≥0.01 |
| 补 seed 123/2026 | **不建议**（边际 F1 + 时差恶化，优先做 Phase B oracle，而不是堆 seed） |
| 从 `last.pt` 续训 | **不建议**；若重开应用 `best.pt` 并先查 NaN |

机器可读：`artifacts/results/stage6/phaseA_stop_gate_verdict.json`
