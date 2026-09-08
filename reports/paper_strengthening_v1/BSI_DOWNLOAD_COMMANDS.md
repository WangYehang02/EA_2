# BSI download & offline import commands

## Important: bypass local proxy

This machine’s `http_proxy=http://127.0.0.1:7890` causes **TLS EOF** to `bsi.ingv.it` / `webservices.ingv.it`.  
**Direct connection works** with certificate verification enabled. Do **not** disable TLS verify.

```bash
conda activate PS
cd /path/to/Earthquake

# Discover + download all 6 QuakeML ZIPs from official archive pages
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  python scripts/paper_strengthening_v1/download_bsi_quakeml.py

# Offline import (laptop → server)
# On laptop (where download works):
#   env -u http_proxy -u https_proxy python scripts/paper_strengthening_v1/download_bsi_quakeml.py
#   rsync -av artifacts/results/paper_strengthening_v1/bsi_catalog/raw/ server:/path/.../bsi_catalog/incoming/
# On server:
env -u http_proxy -u https_proxy \
  python scripts/paper_strengthening_v1/download_bsi_quakeml.py \
    --import-dir artifacts/results/paper_strengthening_v1/bsi_catalog/incoming

# Label audit (no model scores)
python scripts/paper_strengthening_v1/audit_bsi_quakeml_labels.py

# Waveform diagnosis + tiny smoke
env -u http_proxy -u https_proxy python scripts/paper_strengthening_v1/diagnose_waveform_access.py
env -u http_proxy -u https_proxy python scripts/paper_strengthening_v1/smoke_waveform_download.py
```

## Official DOIs (2021–2022)

- 10.13127/BSI/202101 … 202203  
- Pages: `https://bsi.ingv.it/en/archivio-dati?doi=<DOI>`  
- URLs are **scraped from each page**, not invented.

## Outputs

- `artifacts/results/paper_strengthening_v1/bsi_catalog/discovery_manifest.json`  
- `artifacts/results/paper_strengthening_v1/bsi_catalog/download_manifest.json` (URL, DOI, time, SHA256, zip validation)  
- `artifacts/results/paper_strengthening_v1/bsi_catalog/raw/*QML*.zip`
