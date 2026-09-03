"""Frozen 10k train / event-disjoint held-out / cal vs eval splits. Confirm IDs only."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

from earthquake.config import artifacts_dir
from earthquake.stage6.full_splits import load_full_event_ids, load_full_trace_names
from earthquake.stage10.dataset_v2 import annotate_online_catalog

N_TRAIN = 10000
N_HELD = 2000
SEED = 42
NOISE_FRAC = 0.2


def sha256_sorted_ids(ids) -> str:
    h = hashlib.sha256()
    for x in sorted(set(map(str, ids))):
        h.update(x.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def split_heldout_by_event(held: pd.DataFrame, *, seed: int = SEED, cal_frac: float = 0.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    eids = held["event_id"].astype(str).unique()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(eids)
    n_cal = int(round(len(perm) * cal_frac))
    cal_e = set(map(str, perm[:n_cal]))
    eval_e = set(map(str, perm[n_cal:]))
    if cal_e & eval_e:
        raise RuntimeError("calibration and evaluation events overlap")
    cal = held[held["event_id"].astype(str).isin(cal_e)].copy()
    evl = held[held["event_id"].astype(str).isin(eval_e)].copy()
    return cal.reset_index(drop=True), evl.reset_index(drop=True)


def build_diag_sets(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Train 10k event traces + 20% noise; 2000 event-disjoint S-labelled held-out.

    Confirm: event-id list is loaded only to prove disjointness. No confirm waveforms.
    """
    events = pd.read_parquet(artifacts_dir() / "index_full" / "events.parquet")
    noise = pd.read_parquet(artifacts_dir() / "index_full" / "noise.parquet")
    picker = set(load_full_trace_names("stage6_picker_train"))
    dev_tr = set(load_full_trace_names("stage6_dev"))
    confirm = set(load_full_event_ids("stage6_internal_confirm"))
    ev = events[events["trace_name"].astype(str).isin(picker)].copy()
    dv = events[events["trace_name"].astype(str).isin(dev_tr)].copy()
    if set(ev["event_id"].astype(str)) & confirm:
        raise SystemExit("LEAK train∩confirm")
    if set(dv["event_id"].astype(str)) & confirm:
        raise SystemExit("LEAK heldout∩confirm")
    if set(ev["event_id"].astype(str)) & set(dv["event_id"].astype(str)):
        raise SystemExit("LEAK train∩heldout events")
    p = pd.to_numeric(ev["p_arrival_sample"], errors="coerce")
    s = pd.to_numeric(ev["s_arrival_sample"], errors="coerce")
    ev["is_noise"] = False
    dv["is_noise"] = False
    ps = ev[p.notna() & s.notna()]
    po = ev[p.notna() & s.isna()]
    dps = (ps.s_arrival_sample.astype(float) - ps.p_arrival_sample.astype(float)) / 100.0
    short = ps[dps <= 30]
    longp = ps[dps > 30]
    parts = []
    for df, n in [(short, 5000), (longp, min(500, len(longp))), (po, 2500)]:
        if len(df) == 0:
            continue
        parts.append(df.sample(n=min(n, len(df)), random_state=seed))
    tr = pd.concat(parts, ignore_index=True)
    if len(tr) < N_TRAIN:
        extra = ev[~ev.trace_name.isin(tr.trace_name)].sample(
            n=min(N_TRAIN - len(tr), len(ev)), random_state=seed
        )
        tr = pd.concat([tr, extra], ignore_index=True)
    tr = tr.drop_duplicates("trace_name").head(N_TRAIN)
    n_noise = int(round(len(tr) * NOISE_FRAC))
    nsel = noise.sample(n=min(n_noise, len(noise)), random_state=seed).copy()
    nsel["is_noise"] = True
    nsel["event_id"] = "NOISE"
    for c in tr.columns:
        if c not in nsel.columns:
            nsel[c] = np.nan
    tr = pd.concat([tr, nsel[tr.columns]], ignore_index=True)
    sdev = pd.to_numeric(dv["s_arrival_sample"], errors="coerce")
    de = dv[sdev.notna()].sample(n=min(N_HELD, int(sdev.notna().sum())), random_state=seed)
    cal, evl = split_heldout_by_event(de, seed=seed)
    tr_e = set(tr.loc[~tr.is_noise.astype(bool), "event_id"].astype(str))
    if tr_e & confirm:
        raise SystemExit("LEAK train events ∩ confirm")
    if set(cal["event_id"].astype(str)) & confirm:
        raise SystemExit("LEAK cal ∩ confirm")
    if set(evl["event_id"].astype(str)) & confirm:
        raise SystemExit("LEAK eval ∩ confirm")
    if set(cal["event_id"].astype(str)) & set(evl["event_id"].astype(str)):
        raise SystemExit("LEAK cal ∩ eval events")
    meta = {
        "seed": seed,
        "n_train_event_traces": int((~tr.is_noise.astype(bool)).sum()),
        "n_train_noise": int(tr.is_noise.astype(bool).sum()),
        "n_train_rows": int(len(tr)),
        "n_heldout": int(len(de)),
        "n_cal_traces": int(len(cal)),
        "n_eval_traces": int(len(evl)),
        "n_cal_events": int(cal["event_id"].nunique()),
        "n_eval_events": int(evl["event_id"].nunique()),
        "n_heldout_events": int(de["event_id"].nunique()),
        "hash_train_event_ids": sha256_sorted_ids(tr.loc[~tr.is_noise.astype(bool), "event_id"]),
        "hash_train_trace_names": sha256_sorted_ids(tr.loc[~tr.is_noise.astype(bool), "trace_name"]),
        "hash_cal_event_ids": sha256_sorted_ids(cal["event_id"]),
        "hash_eval_event_ids": sha256_sorted_ids(evl["event_id"]),
        "hash_cal_trace_names": sha256_sorted_ids(cal["trace_name"]),
        "hash_eval_trace_names": sha256_sorted_ids(evl["trace_name"]),
        "confirm_id_list_used_for_disjointness_only": True,
        "confirm_waveforms_read": False,
        "confirm_metrics_read": False,
    }
    return annotate_online_catalog(tr), annotate_online_catalog(cal), annotate_online_catalog(evl), meta
