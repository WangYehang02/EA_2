# 数据需求清单 — 独立时段验证（已完成）

状态：**COMPLETED**（见 `INDEPENDENT_PERIOD_VALIDATION_REPORT.md`）

- 来源：INGV BSI QuakeML 2021–2022 + INGV dataselect（直连、TLS verify、进程内绕过代理）
- 抽样锁：2000 events / 15741 selected rows → unique trace_key 15141（碰撞消解见 audits）
- 正式 pairs：14253（2000 events），READY 已校验
- 数据根：`/data/yehang/Earthquake_paper_strengthening_v1/independent_period`
