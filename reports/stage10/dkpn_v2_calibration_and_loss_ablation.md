# DKPN v2 阈值校准与 loss 消融（v2 冻结，本轮未创建 v3）

- utc: 2026-08-31T04:52:17Z
- seed=42；8000 step；batch=16；lr=1e-3；mixed-crop + FrozenCycleSampler
- 未训练 full/pilot；未覆盖 v2 冻结目录；未创建 `dkpn_clean_v3_*`
- 选阈协议（init 与 trained 相同）：`max_f1_0p5_on_calibration_picks_cap_3_tie_higher_thr`
- 禁止：trained 最佳阈值 vs init 固定 0.2；evaluation 上重选阈值
- B 与官方逐点 CE **不再数值等价**（受控 loss 消融）

## 1. calibration / evaluation 事件数与 hash

| split | traces | events | event-id SHA256 |
| --- | ---: | ---: | --- |
| train events | 10000 | — | `c001e0730840940bd7477ac271f17388354ab0dbfe833a91435fab2c52ef2688` |
| train noise | 2000 | — | — |
| held-out | 2000 | 1582 | — |
| calibration | 985 | 791 | `5d0976cf309057aa78c50d9cbf7394c831e19f79ca747750f7a95f3393d31705` |
| evaluation | 1015 | 791 | `06fadcab7fb0ecbf682dffa3c5a33b0498b3c68c70e0bbd95b38fc2b89878e68` |

- cal∩eval events = ∅；confirm 仅用 ID 表做不相交，未读波形/指标
- cal traces hash=`a423e0c9708ea7bc179c3d799f26c3688359b29b7b3450168e5e3647c00f3b92`
- eval traces hash=`b8325fde86d839b78b5768d57e2898920745a4b6b1589ecf84ca64855a68d6fe`

## 2. init 与 trained@2000 完整 threshold sweep

mixed-crop trained = **A step 2000**（先前 2000-step gate 未存权重；同 seed/split/协议复现）。

协议在 **calibration** 上选阈（picks/trace≤3 的格子里最大化 F1@0.5，并列取更高阈）。evaluation 只使用冻结阈值。height=0.2 始终单列。

### init（random）

calibration 选出 **thr=0.5**（≤0.3 的格子 picks/trace≈44.8，全部不合格）。

| thr | cal F1@0.1 | cal F1@0.5 | cal picks/tr | cal 无S峰 | 合格 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0.001–0.3 | 0.0112 | 0.1147 | 44.779 | 0.000 | 否 |
| **0.5** | 0.0000 | 0.0000 | 0.000 | 1.000 | 是 |

evaluation reporting-only：thr≤0.3 时 F1@0.5=0.1251、picks/tr=44.801；thr=0.5 时 F1=0、无S峰=1。

### trained A@2000

calibration 选出 **thr=0.01**（合格格子中 F1@0.5 最高）。

| thr | cal F1@0.1 | cal F1@0.5 | P@0.5 | R@0.5 | picks/tr | 无S峰 | 合格 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.001 | 0.5746 | 0.9523 | 0.9523 | 0.9523 | 41.314 | 0.000 | 否 |
| 0.002 | 0.5746 | 0.9523 | 0.9523 | 0.9523 | 13.177 | 0.000 | 否 |
| 0.005 | 0.5746 | 0.9523 | 0.9523 | 0.9523 | 4.310 | 0.000 | 否 |
| **0.010** | 0.5746 | 0.9523 | 0.9523 | 0.9523 | 2.998 | 0.000 | 是 |
| 0.020 | 0.5773 | 0.9475 | 0.9518 | 0.9431 | 2.133 | 0.009 | 是 |
| 0.030 | 0.5796 | 0.9452 | 0.9535 | 0.9371 | 1.713 | 0.017 | 是 |
| 0.050 | 0.5857 | 0.9398 | 0.9617 | 0.9188 | 1.165 | 0.045 | 是 |
| 0.075 | 0.5875 | 0.9301 | 0.9641 | 0.8985 | 1.040 | 0.068 | 是 |
| 0.100 | 0.5890 | 0.9258 | 0.9679 | 0.8873 | 0.984 | 0.083 | 是 |
| 0.150 | 0.5914 | 0.9151 | 0.9726 | 0.8640 | 0.921 | 0.112 | 是 |
| **0.200** | 0.5876 | 0.9032 | 0.9730 | 0.8426 | 0.881 | 0.134 | 是 |
| 0.300 | 0.5861 | 0.8598 | 0.9856 | 0.7624 | 0.777 | 0.226 | 是 |
| 0.500 | 0.4814 | 0.5845 | 0.9927 | 0.4142 | 0.417 | 0.583 | 是 |

## 3. 独立 evaluation（冻结阈值，禁止重选）

| ckpt | 协议选出的 thr | F1@0.1s | F1@0.5s | picks/tr | 无S峰 | true-S p | local S peak p | oracle acc@0.5s | bootstrap F1@0.5 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| init 冻结 0.5 | 0.5 | 0.000 | 0.000 | 0.000 | 1.000 | 0.391 | 0.409 | 0.111 | 0.000–0.000 |
| init 官方 0.2 | 0.2 | 0.0099 | 0.1251 | 44.801 | 0.000 | 0.391 | 0.409 | 0.111 | 0.109–0.141 |
| A@2000 冻结 0.01 | 0.01 | 0.622 | 0.957 | 2.974 | 0.004 | 0.333 | 0.486 | 0.961 | 0.948–0.967 |
| A@2000 官方 0.2 | 0.2 | 0.629 | 0.922 | 0.907 | 0.107 | 0.333 | 0.486 | 0.961 | 0.912–0.934 |

结论：**不是“只缺阈值校准”。** init 在合格阈值下 F1=0；trained@2000 在冻结 thr=0.01 和官方 0.2 上 evaluation 都大幅高于 init。低阈值不是唯一通路。

## 4. A：current partial-label control，8000 step

- 1.83 h；NaN=false；全N塌缩=false
- oracle eval 0.111 → **0.982**（train 0.080 → 0.985，同时升）
- F1@0.5 height=0.2：0.125 → **0.966**（boot 0.959–0.973）
- F1@0.5 校准冻结阈 0.005：0 → **0.982**（boot 0.976–0.988）；picks/tr=2.03
- phase/N 梯度比：0.026 → **2.25**
- 2000 step 已明显高于 init，故先前 2000-step 小门失败**不是**“原 loss 不可用”，而是步数/抽样/batch 顺序与本复现不可等同；延长到 8000 后稳定。
- 小型 gate：**通过**

## 5. B：phase-balanced partial-label，8000 step

- 2.39 h；NaN=false；全N塌缩=false
- **明确：与官方逐点 CE 不再数值等价**
- oracle eval 0.111 → **0.963**（train 0.080 → 0.978）
- F1@0.5 height=0.2：0.125 → **0.963**（boot 0.953–0.972）
- F1@0.5 校准冻结 thr=0.3：0 → **0.963**；picks/tr=1.90
- true-S p 终值 0.963（A 终值 0.500）；S 峰更锐
- 早期 500–1000 step 曾出现 picks/tr>3，终态回到 ≤3
- 小型 gate：**通过**

## 6. phase/N 梯度比是否改善

是。终态 A=2.25，B=3.37。A 从 0.026 升到 >1，说明当前 token-mean partial-label 在 mixed-crop + 事件背景零 loss 下，8000 step **不再被 N 全局淹没**。B 的组归一化进一步抬高比值，但不是唯一能过门的修复。

## 7. 是否仍发生全 N 塌缩

否。A、B 全程 `all_n_collapse=false`；后段 S max mean：A≈0.64，B≈0.99。

## 8. height=0.2 是否为硬门槛

本轮 **是 v3 硬门槛**：官方 DKPN 默认 height=0.2，Stage6 method lock **未**写入 DKPN 峰阈。本轮不允许用低阈值替代 0.2 来追认 v2。A/B 均在 **0.2 下通过**，因此不需要把 dev-calibrated threshold 写入 method lock。校准协议仅作为诊断预注册，confirm 前若将来采用，须再冻结。

## 9. 是否获准创建 v3

- 小型 8 条 gate：A 通过、B 通过
- 结论码：`A_and_B_pass_prefer_original_loss_longer_training`（原 loss 可用，2000 step 不足；phase/N 归一化非必要）
- **本轮未创建 v3 目录**；未启动 full/pilot
- 若要开工 v3：须你明示。建议沿用 A 的 current partial-label + mixed-crop + 本 checkpoint 规则（`epoch_N.pt`、valid_best 须超过 init+min_delta、F1=0 不得当有效 best、fixed-0.2 与 calibrated 分名）

## 10. confirm 未读取证明

- `confirm_read=false`；`confirm_waveforms_read=false`；`confirm_metrics_read=false`
- 仅加载 `stage6_internal_confirm` 的 **event-id 列表**（n=2700）做不相交
- 未调用 `assert_full_confirm_access_allowed`
- 未打开 confirm 波形 HDF5 键
- v2 冻结目录未覆盖；无 full pilot

## 11. pytest

`185 passed, 2 skipped`（PS 环境，`tests/`，2026-08-31）。

完整网格与逐步诊断：

- `artifacts/results/stage10/dkpn_threshold_calibration.json`
- `artifacts/results/stage10/dkpn_loss_ablation.json`
