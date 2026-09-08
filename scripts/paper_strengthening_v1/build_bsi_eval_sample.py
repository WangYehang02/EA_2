#!/usr/bin/env python
"""Build auditable BSI independent-period sample list from frozen QuakeML packs."""

from __future__ import annotations

import hashlib
import json
import math
import time
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "artifacts/results/paper_strengthening_v1/bsi_catalog/raw"
OUT = Path("/data/yehang/Earthquake_paper_strengthening_v1/independent_period")
SEED = "paper_strengthening_v1_independent_period_v1"
TARGET_EVENTS = 2000
MAX_RECORDS = 30000
MAX_PER_EVENT = 8


def sha256_hex(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()


def sha_rank(*parts: str) -> float:
    """Uniform [0,1) from sha256."""
    hx = sha256_hex(*parts)
    return int(hx[:16], 16) / float(1 << 64)


def local(tag: str) -> str:
    return tag.split("}", 1)[-1]


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def parse_event_qml(data: bytes, source_zip: str, member: str) -> list[dict]:
    if not data or not data.strip():
        return []
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return []

    # event publicID
    event_el = root
    if local(root.tag) != "event":
        # may be quakeml wrapper
        event_el = None
        for el in root.iter():
            if local(el.tag) == "event":
                event_el = el
                break
        if event_el is None:
            return []

    event_id = event_el.attrib.get("publicID", member)
    # preferred origin
    pref_origin = None
    for c in event_el:
        if local(c.tag) == "preferredOriginID" and c.text:
            pref_origin = c.text.strip()
    origins = {}
    for el in event_el.iter():
        if local(el.tag) != "origin":
            continue
        oid = el.attrib.get("publicID")
        lat = lon = depth = otime = mag = None
        for c in el:
            t = local(c.tag)
            if t == "time":
                for cc in c:
                    if local(cc.tag) == "value" and cc.text:
                        otime = cc.text.strip()
            elif t == "latitude":
                for cc in c:
                    if local(cc.tag) == "value" and cc.text:
                        lat = float(cc.text)
            elif t == "longitude":
                for cc in c:
                    if local(cc.tag) == "value" and cc.text:
                        lon = float(cc.text)
            elif t == "depth":
                for cc in c:
                    if local(cc.tag) == "value" and cc.text:
                        # QuakeML depth in meters
                        depth = float(cc.text) / 1000.0
        origins[oid] = {"origin_time": otime, "source_latitude": lat, "source_longitude": lon, "source_depth_km": depth}

    # magnitude (optional)
    mags = []
    for el in event_el.iter():
        if local(el.tag) == "magnitude":
            for c in el:
                if local(c.tag) == "mag":
                    for cc in c:
                        if local(cc.tag) == "value" and cc.text:
                            try:
                                mags.append(float(cc.text))
                            except ValueError:
                                pass
    mag = float(np.nanmedian(mags)) if mags else float("nan")

    # choose origin
    origin = None
    if pref_origin and pref_origin in origins:
        origin = origins[pref_origin]
    elif origins:
        # first
        origin = next(iter(origins.values()))
    if origin is None or not origin.get("origin_time"):
        return []

    # picks
    picks = {}
    for el in event_el.iter():
        if local(el.tag) != "pick":
            continue
        pid = el.attrib.get("publicID")
        phase = emode = ptime = None
        net = sta = loc = cha = None
        for c in el:
            t = local(c.tag)
            if t == "phaseHint" and c.text:
                phase = c.text.strip()
            elif t == "evaluationMode" and c.text:
                emode = c.text.strip()
            elif t == "time":
                for cc in c:
                    if local(cc.tag) == "value" and cc.text:
                        ptime = cc.text.strip()
            elif t == "waveformID":
                net = c.attrib.get("networkCode")
                sta = c.attrib.get("stationCode")
                loc = c.attrib.get("locationCode") or ""
                cha = c.attrib.get("channelCode")
        picks[pid] = {
            "pick_public_id": pid,
            "phase_hint": phase,
            "evaluation_mode": emode,
            "pick_time": ptime,
            "network": net,
            "station": sta,
            "location": loc,
            "channel": cha,
        }

    rows = []
    for pid, pk in picks.items():
        phase = pk["phase_hint"] or ""
        if not phase.upper().startswith("S"):
            continue
        if pk["evaluation_mode"] != "manual":
            continue
        if not pk["network"] or not pk["station"] or not pk["pick_time"]:
            continue
        rows.append(
            {
                "event_id": str(event_id),
                "source_zip": source_zip,
                "qml_member": member,
                "origin_time": origin["origin_time"],
                "source_latitude": origin["source_latitude"],
                "source_longitude": origin["source_longitude"],
                "source_depth_km": origin["source_depth_km"],
                "source_magnitude": mag,
                "pick_public_id": pk["pick_public_id"],
                "phase_hint": pk["phase_hint"],
                "evaluation_mode": pk["evaluation_mode"],
                "pick_time": pk["pick_time"],
                "network": pk["network"],
                "station": pk["station"],
                "location": pk["location"] or "",
                "channel": pk["channel"] or "",
            }
        )
    return rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "locks").mkdir(exist_ok=True)
    (OUT / "lists").mkdir(exist_ok=True)

    all_rows = []
    exclude = []
    for zpath in sorted(RAW.glob("*QML*.zip")):
        print("parsing", zpath.name, flush=True)
        with zipfile.ZipFile(zpath) as zf:
            names = [n for n in zf.namelist() if n.endswith(".qml") or n.endswith(".xml")]
            for i, name in enumerate(names):
                if i % 1000 == 0:
                    print(f"  {zpath.name} {i}/{len(names)}", flush=True)
                try:
                    data = zf.read(name)
                except Exception as e:
                    exclude.append({"member": name, "zip": zpath.name, "reason": f"read_fail:{e}"})
                    continue
                try:
                    rows = parse_event_qml(data, zpath.name, name)
                except Exception as e:
                    exclude.append({"member": name, "zip": zpath.name, "reason": f"parse_fail:{type(e).__name__}"})
                    continue
                if not rows:
                    exclude.append({"member": name, "zip": zpath.name, "reason": "no_eligible_manual_S_or_origin"})
                all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_parquet(OUT / "lists" / "universe_manual_S.parquet", index=False)
    print("universe records", len(df), "events", df.event_id.nunique())

    # eligible events: finite lat/lon/depth preferred; allow nan depth with fallback later
    ev = (
        df.groupby("event_id", as_index=False)
        .agg(
            origin_time=("origin_time", "first"),
            source_latitude=("source_latitude", "first"),
            source_longitude=("source_longitude", "first"),
            source_depth_km=("source_depth_km", "first"),
            source_magnitude=("source_magnitude", "first"),
            n_manual_s=("pick_public_id", "count"),
        )
    )
    ev["has_coords"] = ev["source_latitude"].notna() & ev["source_longitude"].notna()
    ev = ev[ev["has_coords"]].copy()
    # M>=1.5 when available; if mag nan keep (still has manual S)
    ev["mag_ok"] = (~np.isfinite(ev["source_magnitude"])) | (ev["source_magnitude"] >= 1.5)
    ev = ev[ev["mag_ok"]].copy()
    ev["event_rank"] = [sha_rank(SEED, "event", eid) for eid in ev["event_id"].astype(str)]
    ev = ev.sort_values(["event_rank", "event_id"]).reset_index(drop=True)
    selected_events = ev.head(TARGET_EVENTS).copy()
    selected_events.to_csv(OUT / "lists" / "selected_events.csv", index=False)

    # records for selected events
    rec = df[df["event_id"].isin(set(selected_events["event_id"]))].copy()
    rec["rec_rank"] = [
        sha_rank(SEED, "rec", str(r.event_id), str(r.network), str(r.station), str(r.pick_time), str(r.pick_public_id))
        for r in rec.itertuples()
    ]
    rec = rec.sort_values(["event_id", "rec_rank", "pick_public_id"])
    rec = rec.groupby("event_id", group_keys=False).head(MAX_PER_EVENT).reset_index(drop=True)
    if len(rec) > MAX_RECORDS:
        rec["global_rank"] = [sha_rank(SEED, "glob", str(r.pick_public_id)) for r in rec.itertuples()]
        rec = rec.sort_values(["global_rank", "pick_public_id"]).head(MAX_RECORDS).reset_index(drop=True)

    rec["trace_key"] = rec.apply(
        lambda r: f"{r.event_id}|{r.network}|{r.station}|{r.location}|{r.pick_time}", axis=1
    )
    rec.to_parquet(OUT / "lists" / "selected_records.parquet", index=False)
    rec.to_csv(OUT / "lists" / "selected_records.csv", index=False)

    # sample lock
    body = {
        "seed_material": SEED,
        "hash": "sha256",
        "target_n_events": TARGET_EVENTS,
        "max_records": MAX_RECORDS,
        "max_manual_s_per_event": MAX_PER_EVENT,
        "n_universe_manual_S_records": int(len(df)),
        "n_universe_events_with_manual_S": int(df.event_id.nunique()),
        "n_eligible_events_with_coords": int(len(ev)),
        "n_selected_events": int(len(selected_events)),
        "n_selected_records": int(len(rec)),
        "selected_events_sha256": hashlib.sha256(
            ("\n".join(selected_events["event_id"].astype(str).tolist()) + "\n").encode()
        ).hexdigest(),
        "selected_records_sha256": hashlib.sha256(
            ("\n".join(rec["trace_key"].astype(str).tolist()) + "\n").encode()
        ).hexdigest(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "label_filter": "phaseHint startswith S AND evaluationMode==manual",
        "no_python_builtin_hash": True,
    }
    lock_path = OUT / "locks" / "SAMPLE.LOCK.json"
    text = json.dumps(body, sort_keys=True, indent=2) + "\n"
    digest = hashlib.sha256(text.encode()).hexdigest()
    body["lock_body_sha256"] = digest
    lock_path.write_text(json.dumps(body, indent=2) + "\n")
    (OUT / "locks" / "SAMPLE.LOCK.sha256").write_text(digest + "\n")

    pd.DataFrame(exclude).to_csv(OUT / "lists" / "qml_parse_exclusions.csv", index=False)
    print(json.dumps({k: body[k] for k in body if "sha" in k or k.startswith("n_")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
