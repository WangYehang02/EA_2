# DKPN v2 seed42 corrected-pilot 失败归因

**判定：** 接受 `PILOT.FAILED`。不续训、不跑 30 epoch、不覆盖 v2。  
**冻结名：** `dkpn_clean_v2_seed42_failed_homogeneous_crop_cycle`  
路径：`/data/mnt_data/yehang/PSdetec/Earthquake_stage10/dkpn_clean_v2_seed42_failed_homogeneous_crop_cycle`  
manifest：`artifacts/results/stage10/dkpn_v2_frozen_manifest.json`

confirm **波形未读**。仅用 confirm **event id** 做 train/dev 不相交断言。

---

## 1. Checkpoint 冻结

| 文件 | 存在 | epoch | 备注 |
|--|--|--|--|
| init.pt | 是 | -1 | 随机初始化 |
| epoch0/1/2.pt | **否** | — | **从未保存，不重构** |
| best_metric.pt | 是 | 0 | `val_s_f1=0.0` |
| best_loss.pt | 是 | 1 | val_loss≈0.088 |
| last.pt | 是 | 2 | 94770 steps；含 opt |
| PILOT.FAILED | 是 | | `dev_s_f1_not_above_init` |

**best_metric 无效：** 三个 virtual epoch 的官方 thr=0.2 F1@0.5 全是 0。保存条件是 `0 > 初始 best_metric=-1`，因此指向 **epoch 0 的 0 分权重**。不得称为有效 best。

代理（因缺 per-epoch 文件）：epoch0≈best_metric.pt，epoch1≈best_loss.pt，epoch2=last.pt。

---

## 2. 训练失败 vs 验证失败

固定协议：S-centered crop，同一 `extract_picks`，train 诊断 256 条 / dev 512 条（val_subset）。

### epoch0（best_metric.pt）

- train / dev **oracle argmax acc@0.5s ≈ 0.285 / 0.277**（明显高于 init 的 0.004）
- S true-loc 概率约 **0.053**，S max 约 **0.12**
- **thr=0.2 时 95–96% 无 S 峰 → F1@0.5=0**
- **thr=0.01–0.05 时 train/dev F1@0.5 ≈ 0.25 / 0.24**

结论：不是「train 会、dev 不会」。是 **校准/峰值阈值**：模型已能粗定位，但 S 峰值远低于官方 0.2。

### epoch1（best_loss.pt）

oracle@0.5s 降至 ~0.11；thr=0.2 仍无峰；N 占比 ~0.995。

### epoch2（last.pt）

train/dev 几乎 **全部像 N**（frac_allN=1，S max≈4e-4）。oracle 仍 ~0.09–0.11（对塌缩通道的 argmax 噪声）。**epoch2 覆盖了 epoch1/0 的 S 能力。**

init：train 与 dev 都接近随机（oracle 0.004），不是偏移对齐。

**分类：**

1. 门控用 thr=0.2 → 看起来像「完全失败」  
2. 实际 epoch0 是 **oracle 有效 + extract_picks@0.2 为 0**  
3. 随后 **N 塌缩**，尤其 homogeneous background epoch  
4. train/dev 同向，**不是**单纯 held-out 分布差异

---

## 3. 协议一致性

| 项 | 结果 |
|--|--|
| 5ch CF、ZNE、100 Hz、3001 | 训练与本审计同一 `compute_cf_5ch` / `normalize_cf_window` |
| PSN = 0/1/2 | 是 |
| eval 只 softmax 一次 | `softmax(logits)`，logits=True |
| crop→label 经 `label_coord` | 是 |
| pad 不进峰值 | 是 |
| extract_picks | height=0.2, distance=50, 三点平滑；与 cbced5a 一致 |
| F1@0.5 | **时间容差 50 samples / 0.5 s**，与概率阈值分开扫 |

20 张 dev 图：`artifacts/results/stage10/v2_pilot_failure_plots/dev_00.png` … `dev_19.png`。

**固定时间偏移：** 未见 ~400 sample（fstab）系统偏差。epoch0 符号中位约 **-51 ~ -81 sample（~0.5–0.8 s）**，不是整窗错位。init 的大偏移是随机 argmax。

---

## 4. 有效监督（homogeneous v2 调度）

8000 行 catalog 抽样（旧：epoch%3 全员同一 crop）：

| epoch | P / S / bg / noise | 说明 |
|--|--|--|
| 0 | 6734 / 0 / 0 / 1266 | 全 P-centered + noise |
| 1 | 0 / 4001 / 2733 / 1266 | 全 S-centered；P-only→background |
| 2 | 2733 / 0 / **4001** / 1266 | P+S 全 background |

v2 当时：P+S 且两相位都不在窗内 → **整窗 `n_pos=1`（伪 N）**。  
新规则：这些窗 **零 loss**；N 只来自认证 noise。

epoch2 在新规则下：`zero_loss` 比例 **0.50**，P+S background **4001** 窗（约 59% event）。v2 实际把它们当成 N 训了，与 epoch2 塌缩一致。

相位 vs 非相位权重比约 **0.006–0.011**（N/not_p 占绝对多数）。last.pt 上 true-S 处 |grad| / N 区 ≈ **105**（S 点仍可回传，但全局已被 N 淹没）。

---

## 5. Mixed-crop 修正（代码，尚未用于 v2 权重）

`crop_type = (stable_hash(trace_id) + virtual_epoch) % 3`

- 每 epoch 同时有 P/S/background  
- 3 epoch 每条 P+S trace 三类各一次  
- DDP：全局定 crop 再 shard  
- event 未知 background **不再**当 N  

测试：`tests/test_stage10_online_crop_and_launch.py`、`test_event_background_both_phases_outside_is_zero_loss_not_n`

---

## 6. 小型 held-out gate（已跑完，未通过）

`artifacts/results/stage10/dkpn_v3_small_generalization_gate.json`

- 12000 train 窗（含 20% noise）/ 2000 event-disjoint dev；mixed-crop；seed42 随机 init；**2000 step**；~18 min（1×4090）
- 每 epoch 内已出现 P/S/background/noise
- 无 NaN；S 真值处梯度非零（~3.7e-5）
- **oracle F1/acc@0.5s：train 0.057→0.161，dev 0.035→0.163**（train 与 dev 同向，不是只过拟合）
- **官方 extract_picks thr=0.2 的 F1@0.5：train 0.030→0，dev 0.023→0**（S max 0.52→0.14，峰掉到 0.2 以下）

门控要求 thresholded F1 也高于 init → **`ok=false`，不准启动 v3**。  
不得用加长 epoch 掩盖：这是峰值校准/N 质量压过 S，与 v2 epoch0 同一机制。

---

## 7. v3

**不准启动** `dkpn_clean_v3_seed42_mixedcrop`。v2 未覆盖。
