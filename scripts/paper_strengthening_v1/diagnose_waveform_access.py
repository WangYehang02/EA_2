#!/usr/bin/env python
"""Layered diagnosis of INGV/EIDA waveform access (station/dataselect/discovery)."""

from __future__ import annotations

import json
import socket
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path

OUT = Path("artifacts/results/paper_strengthening_v1/waveform_diag")
OUT.mkdir(parents=True, exist_ok=True)

ENDPOINTS = {
    "ingv_station": "https://webservices.ingv.it/fdsnws/station/1/query?net=IV&sta=ACER&level=station&format=text",
    "ingv_dataselect_tiny": "https://webservices.ingv.it/fdsnws/dataselect/1/query?net=IV&sta=ACER&loc=--&cha=HHZ&starttime=2022-01-01T00:00:00&endtime=2022-01-01T00:00:10",
    "ingv_event_text": "https://webservices.ingv.it/fdsnws/event/1/query?starttime=2022-01-01&endtime=2022-01-02&minmagnitude=4&format=text&limit=3",
    "eida_routing": "https://www.orfeus-eu.org/fdsnws/routing/1/query?sta=ACER&format=json",
}


def classify(exc: Exception | None, status: int | None, body: bytes | None) -> str:
    if exc is not None:
        name = type(exc).__name__
        msg = str(exc).lower()
        if isinstance(exc, socket.gaierror) or "name or service not known" in msg or "nodename nor servname" in msg:
            return "DNS_OR_CONNECT"
        if isinstance(exc, ssl.SSLError) or "ssl" in msg or "eof occurred" in msg:
            return "TLS"
        if isinstance(exc, TimeoutError) or "timed out" in msg:
            return "CONNECT_TIMEOUT"
        if isinstance(exc, urllib.error.HTTPError):
            if exc.code in (502, 503, 504):
                return "HTTP_GATEWAY"
            return f"HTTP_{exc.code}"
        if "No FDSN services could be discovered" in str(exc):
            return "SERVICE_DISCOVERY"
        return f"OTHER_{name}"
    if status == 204:
        return "HTTP_204_NO_DATA"
    if status == 200:
        if body is not None and len(body) == 0:
            return "HTTP_200_EMPTY_BODY"
        return "OK"
    if status in (502, 503, 504):
        return "HTTP_GATEWAY"
    return f"HTTP_{status}"


def probe(url: str) -> dict:
    t0 = time.time()
    ctx = ssl.create_default_context()
    # Bypass local proxy that breaks INGV TLS (direct connection works).
    import os

    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
        os.environ.pop(k, None)
    try:
        import requests

        r = requests.get(url, timeout=30, proxies={"http": None, "https": None}, headers={"User-Agent": "EA2-waveform-diag/1.0"})
        body = r.content[:5000]
        status = r.status_code
        return {
            "url": url,
            "ok": status == 200 and len(body) > 0,
            "status": status,
            "nbytes": len(body),
            "elapsed_s": time.time() - t0,
            "failure_class": classify(None, status, body),
            "preview": body[:120].decode("latin1", errors="replace"),
            "via": "requests_noproxy",
        }
    except Exception as e:
        status = getattr(e, "response", None)
        status = getattr(status, "status_code", None) if status is not None else getattr(e, "code", None)
        return {
            "url": url,
            "ok": False,
            "status": status,
            "elapsed_s": time.time() - t0,
            "failure_class": classify(e, status, None),
            "error_type": type(e).__name__,
            "error": str(e)[:400],
            "via": "requests_noproxy",
        }


def probe_obspy() -> dict:
    try:
        from obspy.clients.fdsn import Client
        from obspy.clients.fdsn.header import FDSNNoServiceException

        try:
            Client("INGV")
            return {"ok": True, "failure_class": "OK", "note": "Client('INGV') constructed"}
        except FDSNNoServiceException as e:
            return {"ok": False, "failure_class": "SERVICE_DISCOVERY", "error": str(e)[:400]}
        except Exception as e:
            return {"ok": False, "failure_class": classify(e, None, None), "error_type": type(e).__name__, "error": str(e)[:400]}
    except Exception as e:
        return {"ok": False, "failure_class": "IMPORT_OR_OTHER", "error": str(e)[:400]}


def main() -> int:
    rows = {k: probe(v) for k, v in ENDPOINTS.items()}
    rows["obspy_client_INGV"] = probe_obspy()
    # DNS resolve check
    dns = {}
    for host in ["webservices.ingv.it", "www.orfeus-eu.org", "bsi.ingv.it"]:
        try:
            dns[host] = {"ok": True, "addrs": list({ai[4][0] for ai in socket.getaddrinfo(host, 443)})}
        except Exception as e:
            dns[host] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    out = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dns": dns,
        "probes": rows,
        "summary_classes": sorted({r.get("failure_class") for r in rows.values()}),
        "laptop_offline_import_hint": [
            "# On a machine where INGV works:",
            "python scripts/paper_strengthening_v1/diagnose_waveform_access.py",
            "# If dataselect OK, download a tiny mseed, checksum, then copy to server:",
            "# scp tiny.mseed user@server:/path/artifacts/results/paper_strengthening_v1/waveform_diag/imports/",
            "python scripts/paper_strengthening_v1/download_bsi_quakeml.py --import-dir /path/to/downloaded_zips",
        ],
        "tls_note": "Certificate verification is kept enabled; disabling TLS verify is NOT an accepted formal fix.",
    }
    (OUT / "waveform_access_diagnosis.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v.get("failure_class") for k, v in rows.items()}, indent=2))
    print("dns", {k: v.get("ok") for k, v in dns.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
