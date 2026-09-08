#!/usr/bin/env python
"""Batch download 3C waveforms for frozen independent-period records.

Process-local proxy bypass; TLS verify ON. Resume-safe via per-trace JSON status.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

warnings.filterwarnings("ignore")

for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
    os.environ.pop(k, None)

ROOT = Path(__file__).resolve().parents[2]
OUT = Path("/data/yehang/Earthquake_paper_strengthening_v1/independent_period")
WAVE = OUT / "waveforms"
LOG = OUT / "download"
PROXIES = {"http": None, "https": None}
PRE_S = 40.0
LENGTH = 120.0
VS_FALLBACK = 3.5
DATASELECT = "https://webservices.ingv.it/fdsnws/dataselect/1/query"
STATION = "https://webservices.ingv.it/fdsnws/station/1/query"
UA = "EA2-paper-strengthening-v1-independent/1.0"
MAX_WORKERS = 40
RETRIES = 3


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def make_session() -> requests.Session:
    s = requests.Session()
    s.proxies.update(PROXIES)
    s.headers.update({"User-Agent": UA})
    retry = Retry(total=RETRIES, backoff_factor=0.8, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=["GET"])
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    return s


def load_mlp():
    import pickle
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    path = ROOT / "artifacts/models/stage6/history_picker_train/travel_time_baseline_mlp.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * r * np.arcsin(np.minimum(1.0, np.sqrt(a))))


def key_hash(trace_key: str) -> str:
    return hashlib.sha1(trace_key.encode()).hexdigest()


class StationCache:
    def __init__(self, session: requests.Session):
        self.session = session
        self.lock = threading.Lock()
        self.cache: dict[str, dict] = {}
        self.path = LOG / "station_coords_cache.json"
        if self.path.exists():
            try:
                self.cache = json.loads(self.path.read_text())
            except Exception:
                self.cache = {}

    def get(self, net: str, sta: str) -> dict:
        k = f"{net}.{sta}"
        with self.lock:
            if k in self.cache:
                return self.cache[k]
        # fetch StationXML level=station
        params = {"network": net, "station": sta, "level": "station", "format": "text"}
        try:
            r = self.session.get(STATION, params=params, timeout=60, proxies=PROXIES, verify=True)
            if r.status_code != 200:
                val = {"error": f"HTTP_{r.status_code}"}
            else:
                # FDSN text: Network|Station|Latitude|Longitude|Elevation|...
                lines = [ln for ln in r.text.splitlines() if ln and not ln.startswith("#")]
                if not lines:
                    val = {"error": "empty_station_text"}
                else:
                    parts = lines[0].split("|")
                    val = {
                        "station_latitude": float(parts[2]),
                        "station_longitude": float(parts[3]),
                        "station_elevation_m": float(parts[4]) if parts[4] not in ("", "None") else 0.0,
                    }
        except Exception as e:
            val = {"error": f"{type(e).__name__}:{e}"}
        with self.lock:
            self.cache[k] = val
            if len(self.cache) % 20 == 0:
                self.path.write_text(json.dumps(self.cache))
        return val

    def flush(self):
        with self.lock:
            self.path.write_text(json.dumps(self.cache, indent=2))


def pick_components(st):
    prefer = ["HH", "EH", "BH", "HN"]
    by_pref = {p: {} for p in prefer}
    for tr in st:
        ch = tr.stats.channel
        if len(ch) < 3:
            continue
        pref, comp = ch[:2], ch[-1].upper()
        if pref in by_pref and comp in ("E", "N", "Z"):
            by_pref[pref][comp] = tr
    for pref in prefer:
        if set(by_pref[pref]) >= {"E", "N", "Z"}:
            from obspy import Stream

            return Stream([by_pref[pref]["E"], by_pref[pref]["N"], by_pref[pref]["Z"]]), pref
    return None, None


def process_one(row: dict, session: requests.Session, mlp, station_cache: StationCache) -> dict:
    from obspy import UTCDateTime, read
    from io import BytesIO

    key = row["trace_key"]
    hid = key_hash(key)
    dest = WAVE / f"{hid}.mseed"
    meta_path = WAVE / f"{hid}.json"
    if dest.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            if meta.get("ok") is True or meta.get("reason") == "FAIL_LABEL_OUTSIDE_WINDOW":
                meta["cached"] = True
                return meta
            if meta.get("ok") is False and meta.get("reason") and not str(meta["reason"]).startswith("dataselect:transient"):
                # permanent failure cached
                meta["cached"] = True
                return meta
        except Exception:
            pass

    out = {"trace_key": key, "event_id": row["event_id"], "network": row["network"], "station": row["station"]}
    net, sta = str(row["network"]), str(row["station"])
    sc = station_cache.get(net, sta)
    if "error" in sc:
        out.update({"ok": False, "reason": f"station_meta:{sc['error']}"})
        meta_path.write_text(json.dumps(out) + "\n")
        return out

    try:
        dist = haversine_km(row["source_latitude"], row["source_longitude"], sc["station_latitude"], sc["station_longitude"])
    except Exception as e:
        out.update({"ok": False, "reason": f"distance:{e}"})
        meta_path.write_text(json.dumps(out) + "\n")
        return out

    elev = sc.get("station_elevation_m", 0.0)
    depth = row["source_depth_km"] if row["source_depth_km"] is not None and np.isfinite(float(row["source_depth_km"])) else 10.0
    pred = mlp.predict_row(pd.Series({"distance_km": dist, "source_depth_km": depth, "station_elevation_m": elev}))
    tau = float(pred.get("base_tau_s", np.nan))
    if not np.isfinite(tau):
        tau = max(dist, 0.0) / VS_FALLBACK + 5.0
        out["tau_fallback"] = True

    origin = pd.to_datetime(row["origin_time"], utc=True)
    start = origin + pd.to_timedelta(tau - PRE_S, unit="s")
    end = start + pd.to_timedelta(LENGTH, unit="s")
    start_s = start.strftime("%Y-%m-%dT%H:%M:%S.") + f"{start.microsecond:06d}" + "Z"
    end_s = end.strftime("%Y-%m-%dT%H:%M:%S.") + f"{end.microsecond:06d}" + "Z"

    params = {
        "network": net,
        "station": sta,
        "location": "*",
        "channel": "HH?,EH?,BH?,HN?",
        "starttime": start_s,
        "endtime": end_s,
    }
    try:
        r = session.get(DATASELECT, params=params, timeout=90, proxies=PROXIES, verify=True)
    except Exception as e:
        out.update({"ok": False, "reason": f"dataselect:transient:{type(e).__name__}:{e}", "start": start_s, "end": end_s})
        meta_path.write_text(json.dumps(out) + "\n")
        return out

    if r.status_code == 204:
        out.update({"ok": False, "reason": "HTTP_204_NO_DATA", "start": start_s, "end": end_s})
        meta_path.write_text(json.dumps(out) + "\n")
        return out
    if r.status_code != 200:
        out.update({"ok": False, "reason": f"HTTP_{r.status_code}", "start": start_s, "end": end_s, "preview": r.text[:200]})
        meta_path.write_text(json.dumps(out) + "\n")
        return out

    try:
        st = read(BytesIO(r.content), format="MSEED")
    except Exception as e:
        out.update({"ok": False, "reason": f"mseed_parse:{type(e).__name__}:{e}"})
        meta_path.write_text(json.dumps(out) + "\n")
        return out

    st.merge(method=1, fill_value=0)
    picked, pref = pick_components(st)
    if picked is None:
        out.update({"ok": False, "reason": "missing_3C", "channels": [tr.stats.channel for tr in st]})
        meta_path.write_text(json.dumps(out) + "\n")
        return out

    # align to planned window start
    t0 = UTCDateTime(start.to_pydatetime())
    t1 = t0 + LENGTH
    picked.trim(starttime=t0, endtime=t1, pad=True, fill_value=0)
    for tr in picked:
        if abs(float(tr.stats.sampling_rate) - 100.0) > 1e-6:
            tr.resample(100.0)
    n = min(int(tr.stats.npts) for tr in picked)
    target_n = int(LENGTH * 100)
    n = min(n, target_n)
    for tr in picked:
        tr.data = np.asarray(tr.data[:n], dtype=np.float32)
        if len(tr.data) < target_n:
            pad = np.zeros(target_n - len(tr.data), dtype=np.float32)
            tr.data = np.concatenate([tr.data, pad])
        tr.stats.npts = target_n
        tr.stats.sampling_rate = 100.0
        tr.stats.starttime = t0
    n = target_n
    picked.write(str(dest), format="MSEED")

    pick_t = pd.to_datetime(row["pick_time"], utc=True)
    tr_start = pd.to_datetime(str(picked[0].stats.starttime), utc=True)
    s_sample = (pick_t - tr_start).total_seconds() * 100.0
    label_ok = 0.0 <= s_sample < n
    reason = "ok" if label_ok else "FAIL_LABEL_OUTSIDE_WINDOW"
    out.update(
        {
            "ok": bool(label_ok),
            "reason": reason,
            "label_outside_window": (not label_ok),
            "include_in_eval": bool(label_ok),
            "mseed_path": str(dest),
            "mseed_sha256": sha_file(dest),
            "channel_prefix": pref,
            "trace_start_time": str(tr_start),
            "sampling_rate_hz": 100.0,
            "n_samples": int(n),
            "s_arrival_sample": float(s_sample),
            "distance_km": float(dist),
            "station_latitude": sc["station_latitude"],
            "station_longitude": sc["station_longitude"],
            "station_elevation_m": float(elev),
            "base_tau_s": float(tau),
            "source_depth_km_used": float(depth),
            "window_start_planned": start_s,
            "window_end_planned": end_s,
        }
    )
    for k in (
        "event_id",
        "origin_time",
        "source_latitude",
        "source_longitude",
        "source_depth_km",
        "source_magnitude",
        "network",
        "station",
        "location",
        "pick_time",
        "pick_public_id",
        "evaluation_mode",
        "phase_hint",
        "trace_key",
    ):
        out[k] = row[k]
    meta_path.write_text(json.dumps(out) + "\n")
    return out


def main() -> int:
    WAVE.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    rec = pd.read_parquet(OUT / "lists" / "selected_records.parquet")
    mlp = load_mlp()
    session = make_session()
    station_cache = StationCache(session)

    results = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(process_one, row.to_dict(), session, mlp, station_cache) for _, row in rec.iterrows()]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            if i % 100 == 0 or i == len(futs):
                n_ok = sum(1 for x in results if x.get("include_in_eval") or (x.get("ok") and not x.get("label_outside_window")))
                # include_in_eval preferred
                n_inc = sum(1 for x in results if x.get("include_in_eval") is True or (x.get("reason") == "ok"))
                print(
                    f"progress {i}/{len(futs)} include={n_inc} elapsed={time.time()-t0:.0f}s",
                    flush=True,
                )
                pd.DataFrame(results).to_parquet(LOG / "download_status_partial.parquet", index=False)

    station_cache.flush()
    df = pd.DataFrame(results)
    df.to_parquet(LOG / "download_status.parquet", index=False)
    df.to_csv(LOG / "download_status.csv", index=False)
    reasons = df["reason"].value_counts(dropna=False).to_dict() if len(df) else {}
    summary = {
        "n_requested": int(len(rec)),
        "n_completed_rows": int(len(df)),
        "n_ok_reason": int((df["reason"] == "ok").sum()) if len(df) else 0,
        "n_label_outside": int((df["reason"] == "FAIL_LABEL_OUTSIDE_WINDOW").sum()) if len(df) else 0,
        "n_failed_other": int((~df["reason"].isin(["ok", "FAIL_LABEL_OUTSIDE_WINDOW"])).sum()) if len(df) else 0,
        "reasons": reasons,
        "elapsed_s": time.time() - t0,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "proxy": "bypassed_process_local",
        "tls_verify": True,
    }
    (LOG / "download_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
