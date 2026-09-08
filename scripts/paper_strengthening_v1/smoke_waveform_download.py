#!/usr/bin/env python
"""Fixed small-scale INGV waveform download smoke (bypass broken local proxy)."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import requests

OUT = Path("artifacts/results/paper_strengthening_v1/waveform_diag")
OUT.mkdir(parents=True, exist_ok=True)
IMP = OUT / "imports"
IMP.mkdir(parents=True, exist_ok=True)

for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
    os.environ.pop(k, None)

PROXIES = {"http": None, "https": None}
# Fixed tiny request — not selected by model confidence.
URL = (
    "https://webservices.ingv.it/fdsnws/dataselect/1/query"
    "?net=IV&sta=ACER&loc=*&cha=HHZ&starttime=2022-06-15T00:00:00&endtime=2022-06-15T00:00:30"
)


def main() -> int:
    t0 = time.time()
    row = {"url": URL, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        r = requests.get(URL, timeout=60, proxies=PROXIES, headers={"User-Agent": "EA2-waveform-smoke/1.0"})
        row["status"] = r.status_code
        row["nbytes"] = len(r.content)
        row["elapsed_s"] = time.time() - t0
        if r.status_code == 204:
            row["failure_class"] = "HTTP_204_NO_DATA"
            row["ok"] = False
        elif r.status_code != 200:
            row["failure_class"] = f"HTTP_{r.status_code}"
            row["ok"] = False
            row["preview"] = r.content[:200].decode("latin1", errors="replace")
        else:
            dest = IMP / "IV.ACER.HHZ.20220615T000000_30s.mseed"
            dest.write_bytes(r.content)
            row["dest"] = str(dest)
            row["sha256"] = hashlib.sha256(r.content).hexdigest()
            row["ok"] = len(r.content) > 0
            row["failure_class"] = "OK" if row["ok"] else "HTTP_200_EMPTY_BODY"
            # parse with obspy if available
            try:
                from obspy import read

                st = read(str(dest))
                row["obspy_ntraces"] = len(st)
                row["obspy_ids"] = [tr.id for tr in st]
                row["obspy_sr"] = [float(tr.stats.sampling_rate) for tr in st]
            except Exception as e:
                row["obspy_parse_error"] = f"{type(e).__name__}: {e}"
    except Exception as e:
        row["ok"] = False
        row["failure_class"] = "TLS_OR_CONNECT"
        row["error"] = f"{type(e).__name__}: {e}"
    (OUT / "waveform_smoke_download.json").write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps(row, indent=2))
    return 0 if row.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
