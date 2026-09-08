# Waveform access diagnosis

## Failure-mode separation (this server)

With **default proxy** (`127.0.0.1:7890`): station/dataselect/event → **TLS EOF**; ObsPy → **SERVICE_DISCOVERY** (downstream of TLS).

With **proxy bypass** (`env -u http_proxy -u https_proxy ...`), certificate verification **kept on**:

| Probe | Class |
| --- | --- |
| DNS `webservices.ingv.it` | OK |
| FDSN station | OK |
| FDSN dataselect tiny | OK |
| FDSN event text | OK |
| ObsPy `Client("INGV")` | OK |
| EIDA routing | HTTP_403 |

Tiny smoke download: `waveform_diag/imports/IV.ACER.HHZ.20220615T000000_30s.mseed` (sha256 recorded).

**204 vs connect failure:** smoke returned HTTP 200 with parseable MiniSEED — not a no-data case.

Artifacts:

- `artifacts/results/paper_strengthening_v1/waveform_diag/waveform_access_diagnosis.json`  
- `artifacts/results/paper_strengthening_v1/waveform_diag/waveform_smoke_download.json`
