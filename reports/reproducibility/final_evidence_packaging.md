# Reproducibility notes (engineering; not paper contributions)

The following are **engineering / packaging** details and must **not** be listed as scientific contributions:

- GPU watchdog / process locking  
- HDF5 / memmap / crop caches  
- CRLF / path fixes  
- Sentinel files (`*.PASSED`, `*.CONFIRMED`, `FINAL_EVIDENCE.LOCKED`)  
- Pytest counts  

## Final evidence packaging commands

```bash
conda activate PS
cd /path/to/Earthquake
python scripts/audit_final_evidence.py
python scripts/build_final_paper_tables.py
python scripts/build_final_paper_figures.py
```

Primary evidence roots:

- `artifacts/results/pairwise_pilot/`  
- `artifacts/results/pairwise_fulldev/`  
- `artifacts/results/pairwise_confirm/`  
- `artifacts/results/final_evidence/`  
- `reports/paper/`
