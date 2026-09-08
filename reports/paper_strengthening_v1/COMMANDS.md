# Reproducible commands

```bash
conda activate PS
cd /path/to/Earthquake

python scripts/paper_strengthening_v1/run_strong_controls_dev.py
python scripts/paper_strengthening_v1/probe_ingv.py
python scripts/paper_strengthening_v1/estimate_power_proxy.py
python scripts/paper_strengthening_v1/run_independent_period_eval.py
pytest -q tests/test_paper_strengthening_v1.py
```
