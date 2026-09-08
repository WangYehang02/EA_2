#!/usr/bin/env python
"""Probe INGV/EIDA availability for independent-period validation (catalog first)."""

from __future__ import annotations

import json
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "results" / "paper_strengthening_v1" / "ingv_probe"
OUT.mkdir(parents=True, exist_ok=True)

URLS = [
    "http://webservices.ingv.it/fdsnws/event/1/query?starttime=2021-01-01&endtime=2021-01-03&minmagnitude=3&format=text&limit=5",
    "https://webservices.ingv.it/fdsnws/event/1/query?starttime=2021-01-01&endtime=2021-01-03&minmagnitude=3&format=text&limit=5",
    "http://webservices.rm.ingv.it/fdsnws/event/1/query?starttime=2021-01-01&endtime=2021-01-03&minmagnitude=3&format=text&limit=5",
    "https://webservices.rm.ingv.it/fdsnws/event/1/query?starttime=2021-01-01&endtime=2021-01-03&minmagnitude=3&format=text&limit=5",
]


def try_url(url: str) -> dict:
    t0 = time.time()
    try:
        ctx = ssl.create_default_context()
        req = urllib.request.Request(url, headers={"User-Agent": "EA2-paper-strengthening/1.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=30) as r:
            body = r.read(2000)
            return {
                "url": url,
                "ok": True,
                "status": getattr(r, "status", None),
                "nbytes": len(body),
                "elapsed_s": time.time() - t0,
                "preview": body[:200].decode("utf-8", errors="replace"),
            }
    except Exception as e:
        return {
            "url": url,
            "ok": False,
            "error_type": type(e).__name__,
            "error": str(e)[:500],
            "elapsed_s": time.time() - t0,
        }


def main() -> int:
    rows = [try_url(u) for u in URLS]
    obspy_info = {"attempted": False}
    try:
        from obspy.clients.fdsn import Client

        obspy_info["attempted"] = True
        Client("INGV")
        obspy_info["client"] = "INGV discovered"
    except Exception as e:
        obspy_info["attempted"] = True
        obspy_info["error_type"] = type(e).__name__
        obspy_info["error"] = str(e)[:500]

    out = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "http_probes": rows,
        "obspy": obspy_info,
        "any_success": any(r.get("ok") for r in rows),
        "conclusion": (
            "INGV FDSN reachable"
            if any(r.get("ok") for r in rows)
            else "INGV FDSN unreachable from this environment (TLS/502). Independent download blocked."
        ),
    }
    (OUT / "ingv_probe.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0 if out["any_success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
