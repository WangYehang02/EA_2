#!/usr/bin/env python
"""Independence + information-access audits for independent-period sample (pre-prediction)."""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path("/data/yehang/Earthquake_paper_strengthening_v1/independent_period")
ART = ROOT / "artifacts"


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def extract_numeric_event_id(s: str) -> str | None:
    s = str(s)
    m = re.search(r"eventId=(\d+)", s)
    if m:
        return m.group(1)
    if re.fullmatch(r"\d+", s):
        return s
    return None


def load_historical_events() -> dict[str, pd.DataFrame]:
    out = {}
    paths = {
        "ranker_train_manifest": ART / "results/stage6/phaseC/ranker_train_s_manifest.csv",
        "phaseB_eval_manifest": ART / "results/stage6/phaseB_eval_manifest.csv",
        "pairs_ranker_train": ART / "results/pairwise_pilot/pairs_ranker_train.parquet",
    }
    # confirm lists if present
    for p in (ART / "results/pairwise_confirm").glob("*"):
        if p.suffix in {".csv", ".parquet", ".json"} and "event" in p.name.lower():
            paths[f"confirm_{p.name}"] = p
    for name, p in paths.items():
        if not p.exists():
            continue
        if p.suffix == ".parquet":
            df = pd.read_parquet(p)
        elif p.suffix == ".csv":
            df = pd.read_csv(p)
        else:
            continue
        cols = [c for c in ("event_id", "origin_time", "source_latitude", "source_longitude", "trace_name") if c in df.columns]
        out[name] = df[cols].copy() if cols else df.head(0)
    return out


def time_loc_overlaps(new: pd.DataFrame, old: pd.DataFrame, *, dt_s: float = 5.0, dist_km: float = 25.0) -> int:
    if "origin_time" not in old.columns or "source_latitude" not in old.columns:
        return -1  # unavailable
    a = new.dropna(subset=["origin_time", "source_latitude", "source_longitude"]).copy()
    b = old.dropna(subset=["origin_time", "source_latitude", "source_longitude"]).copy()
    if len(a) == 0 or len(b) == 0:
        return 0
    a["ot"] = pd.to_datetime(a["origin_time"], utc=True, errors="coerce")
    b["ot"] = pd.to_datetime(b["origin_time"], utc=True, errors="coerce")
    a = a.dropna(subset=["ot"])
    b = b.dropna(subset=["ot"])
    # bucket by minute for candidate pairs
    b = b.sort_values("ot")
    n_hit = 0
    # downsample old if huge: use event-level unique
    b_ev = b.drop_duplicates(subset=["event_id"]) if "event_id" in b.columns else b.drop_duplicates(subset=["ot", "source_latitude", "source_longitude"])
    # index by truncated time
    b_ev = b_ev.copy()
    b_ev["bucket"] = b_ev["ot"].dt.floor("min")
    buckets: dict = {}
    for rec in b_ev.to_dict("records"):
        buckets.setdefault(rec["bucket"], []).append(rec)
    R = 6371.0
    for rec in a.to_dict("records"):
        t0 = rec["ot"].floor("min")
        cands = []
        for db in (-1, 0, 1):
            cands.extend(buckets.get(t0 + pd.Timedelta(minutes=db), []))
        if not cands:
            continue
        for c in cands:
            dt = abs((rec["ot"] - c["ot"]).total_seconds())
            if dt > dt_s:
                continue
            p1, p2 = np.radians(rec["source_latitude"]), np.radians(c["source_latitude"])
            dphi = np.radians(c["source_latitude"] - rec["source_latitude"])
            dl = np.radians(c["source_longitude"] - rec["source_longitude"])
            aa = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
            d = 2 * R * np.arcsin(min(1.0, np.sqrt(aa)))
            if d <= dist_km:
                n_hit += 1
                break
    return int(n_hit)


def main() -> int:
    (OUT / "audits").mkdir(parents=True, exist_ok=True)
    (OUT / "locks").mkdir(exist_ok=True)
    sel_ev = pd.read_csv(OUT / "lists" / "selected_events.csv")
    sel_rec = pd.read_parquet(OUT / "lists" / "selected_records.parquet")
    sample_lock = json.loads((OUT / "locks" / "SAMPLE.LOCK.json").read_text())

    new_ids = set()
    for e in sel_ev["event_id"].astype(str):
        n = extract_numeric_event_id(e)
        if n:
            new_ids.add(n)
        new_ids.add(e)

    hist = load_historical_events()
    id_overlaps = {}
    for name, df in hist.items():
        if "event_id" not in df.columns:
            id_overlaps[name] = {"n_old_events": None, "n_overlap_ids": None, "note": "no event_id"}
            continue
        old_ids = set()
        for e in df["event_id"].astype(str).unique():
            n = extract_numeric_event_id(e)
            if n:
                old_ids.add(n)
            old_ids.add(e)
        inter = new_ids & old_ids
        id_overlaps[name] = {
            "n_old_events": int(df["event_id"].nunique()),
            "n_overlap_ids": int(len(inter)),
            "overlap_examples": sorted(list(inter))[:10],
        }

    # time ranges
    time_ranges = {}
    for name, df in hist.items():
        if "origin_time" not in df.columns:
            continue
        ot = pd.to_datetime(df["origin_time"], utc=True, errors="coerce")
        time_ranges[name] = {"min": str(ot.min()), "max": str(ot.max())}
    new_ot = pd.to_datetime(sel_ev["origin_time"], utc=True, errors="coerce")
    time_ranges["independent_period_selected"] = {"min": str(new_ot.min()), "max": str(new_ot.max())}

    # time-location against manifests that have coords
    tl = {}
    for name, df in hist.items():
        if "source_latitude" in df.columns and "origin_time" in df.columns:
            tl[name] = time_loc_overlaps(sel_ev, df)
        else:
            tl[name] = -1

    # picker_train time from residual features / manifest
    hist_man = ART / "results/stage6/history_picker_train_manifest.json"
    picker_note = json.loads(hist_man.read_text()) if hist_man.exists() else {}

    independence = {
        "registered_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sample_lock_sha256": sample_lock.get("lock_body_sha256"),
        "n_selected_events": int(len(sel_ev)),
        "n_selected_records": int(len(sel_rec)),
        "new_period": "2021-01-01 to 2022-12-31",
        "historical_project_origin_range_observed": time_ranges,
        "event_id_overlaps": id_overlaps,
        "time_location_overlap_counts_dt5s_d25km": tl,
        "time_location_note": "negative means coords unavailable in that table",
        "independence_claim": "independent_period_same_region_NOT_cross_region",
        "evidence": [
            "Selected BSI origins fall in 2021–2022; historical INSTANCE-based splits observed through ~2020-01 in project artifacts.",
            "Event ID namespace differs (INGV smi:…eventId=) vs INSTANCE numeric; numeric extract overlap checked.",
            "No retrain/search on this sample; labels not used for tuning.",
        ],
        "gaps": [
            "STEAD pretrained corpus exact event overlap cannot be fully audited from this repo alone.",
            "IDA finetune used INSTANCE picker_train; BSI 2021–22 not in that set by construction of INSTANCE temporal coverage, but STEAD pretrain remains a gap.",
            "Confirm-specific event list may not expose latitudes in all artifacts; ID overlap still checked where present.",
        ],
        "history_store": {
            "path": "artifacts/models/stage6/history_picker_train/temporal_history_store.pkl",
            "protocol": picker_note.get("protocol"),
            "n_picker_train_events": picker_note.get("n_picker_train_events"),
            "will_query_without_test_label_update": True,
        },
        "files_sha256": {
            "selected_events.csv": sha_file(OUT / "lists" / "selected_events.csv"),
            "selected_records.parquet": sha_file(OUT / "lists" / "selected_records.parquet"),
            "SAMPLE.LOCK.json": sha_file(OUT / "locks" / "SAMPLE.LOCK.json"),
        },
    }
    ipath = OUT / "audits" / "independence_audit.json"
    ipath.write_text(json.dumps(independence, indent=2) + "\n")

    access = {
        "registered_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "label_contact": {
            "prior_reads": [
                "BSI QuakeML ZIP download verification",
                "evaluationMode manual vs automatic provenance audit (BSI_LABEL_AUDIT)",
                "Sample construction from QML (this run) — reads pick times/modes to build eval list",
            ],
            "used_for_training": False,
            "used_for_hyperparameter_search": False,
            "used_for_threshold_selection": False,
            "used_for_candidate_algorithm_changes": False,
            "used_for_prediction_post_editing": False,
            "formal_metric_join_policy": "after predictions frozen",
            "window_qc_uses_label_position": "only to mark FAIL_LABEL_OUTSIDE_WINDOW; does not change window",
        },
        "forbidden_post_hoc": [
            "replace primary F1@0.5 with MAE",
            "drop hard/outside-window after seeing method metrics",
            "append samples after seeing eval results",
        ],
        "protocol_supplement": "configs/paper_strengthening_v1/independent_period_protocol_supplement.yaml",
        "independence_audit_path": str(ipath),
        "independence_audit_sha256": sha_file(ipath),
    }
    apath = OUT / "audits" / "information_access_audit.json"
    apath.write_text(json.dumps(access, indent=2) + "\n")
    print(json.dumps({"independence": str(ipath), "access": str(apath), "id_overlaps": id_overlaps, "tl": tl}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
