# Independent-period validation — actual commands & status

Generated: 2026-09-07T01:24:57Z

## Status: COMPLETED

## Commands (executed)

```bash
# sample + audits
python scripts/paper_strengthening_v1/build_bsi_eval_sample.py
python scripts/paper_strengthening_v1/build_independence_and_access_audits.py

# download (process-local proxy bypass)
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
  python -W ignore scripts/paper_strengthening_v1/download_bsi_waveforms.py

# candidates (8-GPU shards) then merge
python -W ignore scripts/paper_strengthening_v1/build_independent_candidates_pairs.py
# after empty-CSV merge fix:
python -W ignore scripts/paper_strengthening_v1/build_independent_candidates_pairs.py --merge-only

# READY + formal eval + report
python scripts/paper_strengthening_v1/build_ready_and_data_lock.py
python scripts/paper_strengthening_v1/run_formal_independent_eval.py
python scripts/paper_strengthening_v1/write_independent_validation_report.py
```

## Logs
- `/data/yehang/Earthquake_paper_strengthening_v1/independent_period/logs/download.log`
- `/data/yehang/Earthquake_paper_strengthening_v1/independent_period/logs/candidates.log` + `cand_shard_*.log`
- `/data/yehang/Earthquake_paper_strengthening_v1/independent_period/logs/ready.log`
- `/data/yehang/Earthquake_paper_strengthening_v1/independent_period/logs/eval.log`
- `/data/yehang/Earthquake_paper_strengthening_v1/independent_period/logs/report.log`

## Key artifacts
- READY: `/data/yehang/Earthquake_paper_strengthening_v1/independent_period/READY.json`
- DATA lock: `/data/yehang/Earthquake_paper_strengthening_v1/independent_period/locks/DATA.LOCK.json`
- Predictions: `/data/yehang/Earthquake_paper_strengthening_v1/independent_period/predictions/` + `PREDICTIONS.LOCK.json`
- Metrics: `/data/yehang/Earthquake_paper_strengthening_v1/independent_period/results/metrics_A_to_E.csv`
- Comparisons: `/data/yehang/Earthquake_paper_strengthening_v1/independent_period/results/comparisons.json`
- Report: `reports/paper_strengthening_v1/INDEPENDENT_PERIOD_VALIDATION_REPORT.md`
