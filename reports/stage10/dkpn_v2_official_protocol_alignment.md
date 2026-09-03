# DKPN v2 官方输入/标签协议对照

对照仓库：`/home/yehang/EARTHQUAKE/baseline/DKPN`  
commit：**`cbced5a58282ff6ad2703f9c9f41f8728e334bd0`**（v0.4.12）  
主脚本：`dkpn/train.py`、`dkpn/core.py`（`PreProc` / `DKPN.forward`）、`dkpn/eval_utils.py`。

判定：**CNN 输入协议已与官方对齐（5 通道 CF，不是生三分量直接进网）**。  
small-overfit 成功 **不能**单独当作协议证据；下表逐项来自源码对照。  
与官方 **训练采样策略**（RandomWindow、INSTANCE P–S&lt;3000 过滤）的差异是任务设计（partial-label / 长 P–S），不是输入张量定义。

| 项 | 官方 cbced5a | 本仓库 v2 | 对齐？ |
|--|--|--|--|
| 模型看到的是什么 | 3C 波形经 `PreProc` 变成 **5 通道 CF**（3× filterbank CF + incidence + modulus），再 STD 归一化后进 CNN | `enz_to_zne` → `PreProc.__matrix_cfs__` → `normalize_cf_window` 得到 (5, 3001) | **是** |
| 是否把生波形直接喂给 DKPN CNN | 否 | 否 | **是** |
| 波形归一化 | SeisBench `Normalize(demean, amp std)` 之后 `PreProc` **再次** demean + 按通道 std；CF 算出后对 3 条 CF 与 modulus 再 std | 不重复 SeisBench 那一层；`PreProc` 内部 demean+std 与官方第二段相同，随后 `normalize_cf_window` 对齐 `annotate_window_pre` | **实质对齐**（第一层 SeisBench amp-norm 被 PreProc 覆盖） |
| 滤波 | 无单独的全局 Butterworth 作为 CNN 输入；CF 的 `FBSummary` 使用 `freqmin=0.5`, `corner=1`, `t_long=4`, `mode=rms` | 同一套 `DKPN().default_args` | **是** |
| 输入长度 | `final_windowlength=3001`；raw 窗 `3001+fp_stabilization`，`fp_stabilization=4s=400` samples | `IN_SAMPLES=3001`, `FSTAB=400`, `WIN_RAW=3401` | **是** |
| 通道顺序 | `component_order: ZNE` | INSTANCE ENZ → `enz_to_zne` | **是** |
| 标签 Gaussian 宽度 | `ProbabilisticLabeller(sigma=10)`，单位 **sample**（100 Hz → 0.10 s） | `sigma_s=0.1` → `sigma=10` samples | **是** |
| Softmax / 通道顺序 | `phases="PSN"`：ch0=P, ch1=S, ch2=N；`forward` 默认 softmax；训练 loss 为 `-y log(p+eps)` | `DKPN(..., phases="PSN")`；训练用 **logits** + `partial_label_nll` / `log_softmax`（禁止 softmax 后再 log_softmax） | **通道顺序是**；loss 为 partial-label 扩展，完整标签时与 CE 一致（已测） |
| 推理峰值 | `classify_aggregate` → SeisBench `picks_from_annotations`；默认 `P_threshold=0.2`, `S_threshold=0.2`；评估脚本 `extract_picks`：平滑 + `find_peaks(height=0.2, distance=50)` | 窗内评测现用同一 `extract_picks`（`src/earthquake/stage10/dkpn_picks.py`）；流式 `picks_from_annotations` 仅在后续 stop-gate 全迹推理时使用 | **窗内峰值已对齐**；全迹 overlap 推理须走官方 `annotate` |
| INSTANCE 过滤 | 官方训练曾要求 P 与 S 都在、且 P–S&lt;3000 samples | 保留 P-only 与 P–S&gt;30s，用 partial-label，**不把窗外相位标成 N** | **有意不同**（任务），不是 CF 输入错误 |
| 窗采样 | `WindowAroundSample` + `RandomWindow(3401)` | 每个 virtual epoch 一条 trace 一个 **确定性** crop（P/S/background 轮换） | **有意不同** |

## 训练前必须保持的约束

- 不得改已通过的 partial-label 与 crop 可见性语义。  
- 不得把 event 迹的未知时段直接当 noise。  
- 正式 30 virtual-epoch 前必须：官方表无「输入张量」项为否，且 corrected pilot 过门。  
- confirm 禁止读取。
