# Stage 6 一次性盲测就绪审计（方法冻结，不跑 confirm）

**日期：** 2026-09-02  
**判决：** `CONFIRM.BLOCKED`  
**本轮未执行 confirm，也不会自动进入下一阶段。**

本轮只读 train/dev 产物、代码、配置和已有 method lock / UNION confirm 协议。  
未打开 confirm 波形；未读取 confirm predictions / metrics 文件内容；未跑 inference；未训练；未重启 DKPN；未改权重；未启动 DiTing。

机器可读产物：

- `artifacts/results/stage10/FINAL_METHOD.LOCK.json`  
  SHA256 `a6264618105f372b548ea5e21fb92505fbddaed5a1bc034d9219d832ac286908`（已 `chmod a-w`）
- `artifacts/results/stage10/CONFIRM_ANALYSIS.LOCK.json`  
  SHA256 `696618462551bed207fadade4c38dc0a23378e210f25b9dbc88327f381796b08`（已 `chmod a-w`）
- `artifacts/results/stage10/CONFIRM_READINESS.json`  
  SHA256 `ff3e363eb5d4eb1f1ec53ff72b27e91bd34a19ebddfa567347f05741c35d6512`（已 `chmod a-w`）

---

## 1. 预注册候选路线盘点

没有「启动前已预注册、且 confirm 前必须完成」的未完成实验。因此**不因候选缺口写 `CONFIRM.BLOCKED`**。  
SegPhase 域内重训、DKPN 额外 seed、DiTing 均**不得**因为 confirm 前还有时间而补跑。

| 路线 | 状态 | 最终 checkpoint | 未完成的预注册 dev 实验 | confirm 前必须完成 |
|---|---|---|---|---|
| STEAD | completed | `PhaseNet.from_pretrained('stead')`；`~/.seisbench/models/v3/phasenet/stead.pt.v2` SHA256 `761c46496139e9917a7480ff56f660914124c806511ea794f0005518358b09ec` | 无 | 否 |
| IDA | completed | `artifacts/models/stage6/phasenet_ida_full_seed42/checkpoints/best.pt` epoch 14 SHA256 `b03fc6ff6c8429d877e2d69f77a31f0fd426ba285d901bdc5682bdb26b158fc4` | 无 | 否 |
| UNION_STEAD5_IDA5 | completed | `fixed_rescore_UNION`（STEAD K=5 ∪ IDA K=5） | 无 | 否（已是 primary） |
| SegPhase | completed | Stage9 `model_100Hz.pth`；外部对照，不替换主方法 | INSTANCE 10C 重训 never_started，**不是** confirm 前提 | 否 |
| DKPN | **rejected** | v4 `epoch_1.pt` 仅事后评估，不进 UNION | 额外 seed 仅在 stop-gate 通过后才允许；门已失败 | 否；**禁止补救** |
| ranker | rejected | — | 无 | 否 |
| fixed-rescore | completed | `lw=0.5, lh=2.0, lp=0.0` | 无 | 否（已是 primary 规则） |
| LFTNet | rejected | 官方发布不完整 | 无 | 否 |
| PhaseNet-ETHZ / SCEDC | completed | Stage7 外部对照 | 无 | 否 |
| EQTransformer | never_started | 权重下载失败 | 无 | 否 |
| DiTing | never_started | — | 无 | 否；本轮禁止启动 |
| 多台站 / GNN | never_started | 已封锁 | 无 | 否 |

**DKPN 强制标记（本轮保持）：**

- `rejected_candidate_source`
- `FULLDEV.STOP_GATE.FAILED`
- `route_closed=true`

DKPN 失败标记路径（仅 stat，未改）：  
`/data/mnt_data/yehang/PSdetec/Earthquake_stage10/dkpn_clean_v4_seed42_fp32/FULLDEV.STOP_GATE.FAILED`

说明：`UNION_STEAD5_IDA5` 的「5」是**同一模型的 top-K 峰**，不是 5 个独立 STEAD/IDA checkpoint。锁文件里 5 个 slot 指向同一权重，避免被理解成五模型集成。

---

## 2. 最终可部署方法（已冻结）

**Primary：** `UNION_STEAD5_IDA5` **fixed-rescore**（锁名 `fixed_rescore_UNION`）

- 可部署 full-dev F1@0.5 = **0.867**（精确值 0.8668048984454653，Phase C.1 全量 S 标注 dev）
- STEAD_top1 full-dev F1@0.5 = **0.840**
- UNION oracle F1@0.5 = **0.892** 仅为上限，**不能**作为 confirm primary
- 不允许根据 confirm 改 primary
- **不含 DKPN**

复制自 Stage6 方法锁 SHA256  
`a02dc28e3f485559dd22b0d38c90d75923a159aed43a797bf234d765ada5b303`  
（`locked_before_confirm=true`，`confirm_seen_before_lock=false`）

### 输入与峰值

- 采样率 100 Hz
- 窗口：SeisBench PhaseNet `annotate` 默认（`in_samples=3001`）
- 分量：`ENZ_HDF5 → ZNE_SeisBench`；标签 `PSN`
- 归一化 / 滤波：仅 SeisBench PhaseNet.annotate 默认；项目侧无额外带通
- 峰值：`find_peaks`，`min_distance=50`，`min_prominence=0.05`，`min_probability=0.1`；无峰则全局 argmax fallback
- 基础模型阈值：STEAD 与 IDA 共用上述峰值门；STEAD_top1 使用 annotate argmax（阈值 0.0）
- UNION：`K=5` / `K=5`，`dedup=0.05s`，`max_union=10`，双边支持时保留 STEAD 代表时刻

### fixed-rescore 公式（冻结）

\[
\mathrm{score}=\lambda_w\log(p+\varepsilon)+\lambda_h\log(\mathrm{hist}+\varepsilon)+\lambda_p\cdot\mathrm{norm\_prominence}
\]

\[
\mathrm{hist}=\exp\bigl(-0.5((c-s_{\mathrm{expected}})/\sigma)^2\bigr)
\]

- \(\lambda_w=0.5,\ \lambda_h=2.0\)（无历史则 0）, \(\lambda_p=0.0\)
- \(\varepsilon=10^{-8}\)
- \(p=\max(p_{\mathrm{STEAD}},p_{\mathrm{IDA}},10^{-6})\)
- shrinkage \(k=50\)，`min_history=5`
- \(\sigma_s=\mathrm{clip}(1.4826\cdot\mathrm{MAD}, 0.05, 1.0)\) 秒，再乘采样率
- 打分平局：严格大于才替换；相等保留迭代中先出现的候选  
  UNION 池截断前排序：`(-both_support, -max_prob, stead_rank)`

### 缺失通道 / 损坏 trace（预注册意图 vs 历史实现）

预注册意图（reader）：缺 E/N/Z 或无法识别形状 → `KeyError` / `ValueError`，不填补。  
Stage6 方法锁：`no_candidate_behavior=nan_prediction`。  
历史 cache 脚本：捕获 `Exception` 后**跳过该迹**（**不是 fail closed**）。这是本轮脚本阻塞项之一。

### 环境与哈希

- Python 3.11.15 / PyTorch 2.5.1+cu121 / SeisBench 0.12.3
- full-dev manifest SHA256 `2cc1e7a3376525ed46ff12d182f947602714a5b353e651e4d59542b56894c4e7`
- confirm split **只引用既有记录、未重读内容**：  
  events `987fc9271d2b58ebd3786d9124cdda4b2fe64ab934ccdcc8e76d2244c296a7d6`  
  traces `67abd5272774baea14b620ea9b53de004a4ef3f4e5a2f5d6ab8ab78dd9daae56`
- 当前树无 git HEAD（`rev-parse` 失败）；Stage6 锁记录 `git_diff_hash=55094e77…`
- `candidate_schema.json` 文件哈希 `ec8baf5f…` 与锁内自列 `e226239c…` 不同，是因为 JSON **把自身 sha 写进文件后再哈希**；以 Stage6 方法锁中的 `e226239c…` 为规范 schema 指纹，不视为内容漂移。

---

## 3. Confirm 分析方法（已冻结）

主比较：`UNION_STEAD5_IDA5 fixed-rescore` vs `STEAD_top1`  
主指标：**F1@0.5s**（协议原文 `S F1@0.5`）

次指标：Precision / Recall / F1@0.1s；miss / coverage；picks per trace；no-S-peak rate；detected AE median / MAE / P95；wrong-peak rate；inference runtime。  
noise FPR：**不纳入**。UNION 协议总体是 `all_s_labelled_internal_confirm`，协议里没有预注册 confirm noise 集。

统计：

- 配对单位 = **event**
- paired event bootstrap，**5000** 次，seed **20260817**
- 报告 ΔF1@0.5 点估计与 95% CI
- **禁止**用 trace bootstrap 代替 event bootstrap

Subgroup：UNION confirm 协议**没有**列出任何分组。  
P–S 间隔 / distance / SNR / station / channel / network 只出现在 **DKPN full-dev stop-gate** 锁里，**不是** UNION confirm 预注册分组。冻结为 `[]`，禁止看过 confirm 后加有利分组。

Oracle 仅允许诊断，不允许作为每条迹的实际输出。

---

## 4. 既有 confirm 成功门槛（原样引用，未改）

**存在预注册的最终 UNION confirm gate。不写 `CONFIRM.GATE.MISSING`。**  
不以本轮建议门槛替换，也不把 DKPN stop-gate 当成 UNION confirm 门槛。

| 项目 | UNION 盲测门槛 | DKPN 候选 stop-gate |
|---|---|---|
| 文件 | `artifacts/results/stage6/final_confirm/confirm_evaluation_protocol.json` | `artifacts/results/stage10/data_cache/dkpn_clean_v4_seed42_fp32/full_dev_stop_gate/METHOD.LOCK.json` |
| 生成脚本 | `scripts/write_stage6_confirm_protocol.py` | Stage10 v4 full-dev 锁 |
| mtime | **2026-08-17 08:48:42 +08**（UTC 00:48:42） | 2026-09-02（DKPN 事后门） |
| SHA256 | `b68261e63894eb94b9e21bb19737cdd6274dd8e44c099df3f3339bd8b98ca863` | 本轮未改该文件 |
| 用途 | `fixed_rescore_UNION vs STEAD_top1` | 是否把 DKPN 加入候选源 |

UNION 协议 `decision_rules` **原文**：

- `strong_confirmed`: `dF1_05>=0.01 and CI_lo>0 and dF1_01>=-0.003 and dP95<=0 and no selective recall collapse`
- `modest_confirmed`: `dF1_05>0 and CI_lo>0 and gain<0.01 and safety ok`
- `direction_only_underpowered`: `point positive but CI includes 0`
- `not_confirmed`: `dF1_05<=0 or safety failure`

Bootstrap：`n=5000`, `seed=20260817`, `unit=event`。

DKPN 门槛（**仅对照，不用于 UNION confirm**）：A ΔF1@0.5≥+0.01 vs 最强波形对照；B F1 不降且 P95 改善≥0.15s 且 miss 不恶化；C UNION+DKPN5 oracle ΔF1≥+0.01 且 CI 下界>0。该门已失败，路线关闭。

建议门槛（ΔF1≥+0.01 且 CI 下界>0 等）与 UNION `strong_confirmed` **方向一致，但不是新的预注册**；以 2026-08-17 协议为准。

---

## 5. Confirm 是否仍盲（仅元数据）

本进程 `/proc/self/fd` **没有**打开 confirm 内容文件。

| 标志 | 本轮审计进程 | 项目历史 |
|---|---|---|
| confirm_waveforms_read | **false** | INSTANCE HDF5 容器存在；未打开波形数组 |
| confirm_predictions_read | **false** | `confirm_predictions.parquet` **文件存在**（stat） |
| confirm_metrics_read | **false** | `confirm_method_metrics.json` **文件存在**（stat） |
| confirm_threshold_tuned | **false** | 本轮未调 |
| confirm_model_selected | **false** | 本轮未选 |

`CONFIRM.CONSUMED` **文件名存在**（2026-08-17 一次性内部 confirm 已消耗）。  
因此：**对本进程仍未读内容；对项目而言已不是第一次盲看。**  
这是 `CONFIRM.BLOCKED` 的主因之一。不得为核对格式再读一条 confirm 波形。

---

## 6. Confirm 执行脚本静态审计（未运行）

脚本：`scripts/run_stage6_final_confirm.py` + `scripts/cache_stage6_phaseB_candidates.py`

| 检查 | 结果 |
|---|---|
| 只加载 `FINAL_METHOD.LOCK` 中的 checkpoint | **否**；加载 Stage6 `final_confirm/method_lock.json` |
| 扫描多个 checkpoint | 否 |
| 在 confirm 上搜 threshold | 否 |
| 在 confirm 上选 K | 否（硬编码 `stead_k=5, ida_k=5`；cache 抽 top-10 再截断） |
| 根据 confirm 替换 primary | 否（写死 `fixed_rescore_UNION`） |
| 调用 oracle 作为实际输出 | 计算并保存 `oracle_UNION`（协议允许诊断）；**实际主输出是 fixed-rescore**，但同一次 confirm 会看到 oracle |
| 每条迹实际预测来自 fixed-rescore | 是（主列） |
| 报错 fail closed | **否**：cache 捕获异常后跳过；无候选写 NaN 继续 |
| 输出 per-trace / per-event | 是（历史脚本会写 parquet + event bootstrap JSON） |
| inference 与统计可分开复现 | 函数拆分存在；未改造成独立入口 |

Dry-run 单测：`tests/test_stage10_preconfirm_readiness.py`，只用 synthetic / 锁文件 / 脚本源码。  
**pytest：9 passed。** 未读 confirm。

---

## 7. 为何是 `CONFIRM.BLOCKED`（不是 `CONFIRM.GATE.MISSING`）

方法可以冻结，UNION 门槛也已存在。阻塞的是「再做一次**第一次**盲 confirm」：

1. `CONFIRM.CONSUMED` 已存在 → 同一内部 confirm 集不能再当一次性盲测。
2. 现有 runner 不加载本轮 `FINAL_METHOD.LOCK.json`。
3. 缺候选 / 读波失败时 **NaN 跳过**，不是 fail closed。
4. cache 脚本静默跳过失败迹。
5. runner 在 confirm 上计算 oracle（诊断列）。

未完成且**必要**的预注册候选：**无**。  
未擅自把 DKPN 门槛写成 UNION 门槛。  
即使将来批准，也必须先处理消耗状态与 fail-closed；**本轮不跑 confirm。**
