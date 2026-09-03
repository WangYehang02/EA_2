# CONFIRM.CONSUMED 溯源审计

**日期：** 2026-09-02  
**判决保持：** `CONFIRM.BLOCKED`  
**未删除、未覆盖** `CONFIRM.CONSUMED`。  
未打开 confirm 波形；未解析 `confirm_predictions.parquet`；未把 confirm F1 用于选方法。  
Grep 误扫见 `PRECONFIRM_AUDIT.ADDENDUM.json`（不得记成 false）。

---

## 1. 首次消耗时间线（stat + 标记文件）

| 事件 | UTC | 本地 +08 | 证据 |
|---|---|---|---|
| Stage6 method lock | 2026-08-17T00:48:41Z | 08:48:41 | `method_lock.json` mtime；SHA `a02dc28e…` |
| 协议写入 | 2026-08-17T00:48:42Z | 08:48:42 | `confirm_evaluation_protocol.json` |
| AUTHORIZED | 2026-08-17T00:49:03Z | 08:49:03 | `CONFIRM_EVAL.AUTHORIZED` |
| RUNNING 启动 | 2026-08-17T00:49:34Z | 08:49:34 | `CONFIRM_EVAL.RUNNING`；日志 `CONFIRM_EVAL.RUNNING` |
| PID 落盘 | 2026-08-17T00:49:46Z | 08:49:46 | `CONFIRM.PID` 内容 `4029591` |
| predictions parquet mtime | 2026-08-17T01:34:59Z | 09:34:59 | **仅 stat，未读内容** |
| metrics json mtime | 2026-08-17T01:34:59Z | 09:34:59 | **仅 stat，未读内容** |
| bootstrap / verdict mtime | 2026-08-17T01:39:12Z | 09:39:12 | **仅 stat** |
| **CONFIRM.CONSUMED** | **2026-08-17T01:39:12.235375Z** | **09:39:12** | 标记 `completed_utc` 与文件 mtime 一致 |

`CONFIRM.CONSUMED` 文件 SHA256 `79f038e5ba04be8d15f948dee7e342d19eaa15ea57d6e0c4be05d5aa8992e239`（哈希标记文件本身，不是指标表）。

标记内方法指纹（非 F1）：

- `method_lock_sha256` = `a02dc28e3f485559dd22b0d38c90d75923a159aed43a797bf234d765ada5b303`
- `one_shot` = true
- `confirm_may_not_be_reused_for_model_selection` = true
- predictions/metrics/bootstrap 的 **文件哈希** 已记录在标记里，本轮未打开那些文件核对内容

AUTHORIZED 同时锁定协议 SHA `b68261e63894eb94b9e21bb19737cdd6274dd8e44c099df3f3339bd8b98ca863`（与现文件一致）。

---

## 2. 执行命令

编排脚本 `scripts/run_stage6_finalization.sh`（现盘 mtime 2026-09-02，属工作区拷贝；命令文本）：

```bash
python -u scripts/authorize_stage6_confirm.py
nohup python -u scripts/run_stage6_final_confirm.py > artifacts/results/stage6/logs/confirm_final_nohup.log 2>&1 &
echo $! > artifacts/results/stage6/final_confirm/CONFIRM.PID
```

环境：conda `PS`。PID **4029591**。  
日志 head 显示 08:49:34 进入 RUNNING，随后 history query 与 STEAD/IDA candidate cache（进度计数，无 F1 表）。

---

## 3. 消耗时的方法版本

消耗时锁定的可部署方法即 Stage6 **`fixed_rescore_UNION`**（`UNION_STEAD5_IDA5` fixed-rescore），对照 **STEAD_top1**。

| 物件 | 消耗时 | 现在 | 是否在消耗后改变 |
|---|---|---|---|
| `method_lock.json` | SHA `a02dc28e…` mtime 00:48:41Z | 相同 SHA | **否** |
| IDA `best.pt` epoch14 | SHA `b03fc6ff…` mtime 2026-08-14 | 相同 | **否** |
| STEAD `stead.pt.v2` | SHA `761c4649…` mtime 2026-08-12 | 相同 | **否** |
| 协议 | SHA `b68261e6…` | 相同 | **否** |
| `candidate_rescorer.py` 等打分源码 | 无 git；mtime 2026-08-22（消耗后 5 天） | 哈希仍为 `65545a1a…` / `17622ce8…` 等，与 `FINAL_METHOD.LOCK` 及更早 Stage4 记录一致 | **内容未改；mtime 像整树拷贝** |
| `run_stage6_final_confirm.py` | 无 git | mtime 2026-09-02；SHA `fcf95686…` 与昨日 FINAL_METHOD.LOCK 一致 | **无内容漂移证据** |
| K / λ / 峰值门 | 锁内 K=5，lw=0.5 lh=2.0 lp=0.0 | 相同 | **否** |

**结论：UNION 权重、K、阈值、rescore 公式没有在消耗之后被改成另一种方法。**  
mtime 晚于 CONSUMED 的 `.py` 不能单独证明逻辑变更；checkpoint 与 method lock 字节未变。

Stage7 曾在消耗后 **复用已冻结的 confirm 预测** 做 ETHZ/SCEDC 外部对照，**没有**据此改 UNION primary。这使该内部 confirm 不能再当第一次盲测，但不构成用 confirm 训练/选主方法。

---

## 4. 旧 confirm 身份

**`previously_consumed_internal_test`**

不是 `contaminated_development_data`，因为：

- method lock 在 AUTHORIZED/RUNNING/CONSUMED **之前**；
- `confirm_seen_before_lock=false`；
- IDA/STEAD 权重未在消耗后替换；
- 没有证据表明 UNION 的 K、阈值或 λ 按 confirm 指标重选；
- DKPN/SegPhase 未加入 primary。

它**已经消耗**：同一 `stage6_internal_confirm`（2700 events / 74753 traces 的既有哈希）不能再做一次性盲测。后续对照若再碰该集，只是已看过的内部测试，不是新的外部盲测。

---

## 5. Grep 误扫

见 `artifacts/results/stage10/PRECONFIRM_AUDIT.ADDENDUM.json`。

**确实读取了 confirm 产物字节**（工具输出里出现了指标/清单字段），包括 `n_stations=460`、`n_events=2669`、清单 CSV 中的 SNR/到时样本。  
**不得**写成 `confirm_metrics_read=false`。三份已冻结 lock 未改，因此旧 READY 文件里的 false 保持原样，以本 addendum 更正。

未打开波形数组，未把 confirm F1 用于改 primary。

---

## 6. Future blind-test runner（仅 synthetic/dev）

模块：`src/earthquake/stage10/future_blind_runner.py`  
锁：`artifacts/results/stage10/FUTURE_BLIND_RUNNER.LOCK.json`  
SHA256 `1aa7d957b173c3387489d4c2d7c9583cfa95d09743a19fd816bd8ed13cba3c90`

- 强制校验 `FINAL_METHOD.LOCK` sidecar + IDA/STEAD/config SHA  
- 只允许 `fixed_rescore_UNION` 与冻结 `STEAD_top1`  
- 禁止 threshold / K / checkpoint 搜索  
- NaN / Inf / 缺通道 / 损坏 trace / 缺候选 → `BlindRunnerError`（fail closed，无静默 skip）  
- 主输出原子冻结（npy replace + sha256）之后才允许诊断钩子；传入 `oracle_fn` 直接拒绝  
- 拒绝 `final_confirm` / DiTing 路径  
- **dev 87293 条**：新 rescore 与冻结 `fixed_rescore_UNION.npy`、`STEAD_top1.npy` **逐条相等**；未跑波形推理、未跑 confirm、未跑 DiTing  

pytest：`tests/test_stage10_future_blind_runner.py` **9 passed**。

---

## 7. 可用的新外部盲测候选（本轮不启动）

INSTANCE 54008 个事件已全部分到 picker / ranker / dev / confirm，**没有**剩余未用 INSTANCE 事件可当新盲集。

| 候选 | 状态 | 可否当新的 UNION 一次性盲测 |
|---|---|---|
| `stage6_internal_confirm` | 已 CONSUMED | 否 |
| Stage6 full-dev | 已用于选方法 | 否 |
| ETHZ / SCEDC PhaseNet 对照 | Stage7 已在消耗后评过 | 否（不是新 UNION 盲测） |
| DiTing | never_started | 本轮禁止启动 |
| 新的、预注册的独立目录/台网/时段（从未参与 UNION 选择） | 尚未存在 | **唯一合法方向**；需新协议与新 seal |

**本轮未启动任何外部测试。**
