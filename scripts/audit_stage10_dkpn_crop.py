#!/usr/bin/env python
"""Stage10 DKPN v2 — crop / P–S / partial-label audit (metadata only, no confirm waveforms)."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, save_json
from earthquake.stage6.full_splits import load_full_event_ids, load_full_trace_names
from earthquake.stage10.crop_v2 import FSTAB, IN_SAMPLES, WIN_RAW, crop_kind_start, phase_visible, visibility_bucket
from earthquake.utils import ensure_dir


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def main() -> None:
    events = pd.read_parquet(artifacts_dir() / "index_full" / "events.parquet")
    picker = set(load_full_trace_names("stage6_picker_train"))
    ev = events[events["trace_name"].astype(str).isin(picker)].copy()
    # disjoint vs confirm/dev lists only
    confirm = set(load_full_event_ids("stage6_internal_confirm"))
    dev = set(load_full_event_ids("stage6_dev"))
    eids = set(ev["event_id"].astype(str))
    assert not (eids & confirm)
    assert not (eids & dev)

    p = pd.to_numeric(ev["p_arrival_sample"], errors="coerce")
    s = pd.to_numeric(ev["s_arrival_sample"], errors="coerce")
    ps = ev[p.notna() & s.notna()].copy()
    po = ev[p.notna() & s.isna()].copy()
    d = (ps["s_arrival_sample"].astype(float) - ps["p_arrival_sample"].astype(float)).to_numpy()
    d_s = d / 100.0

    def n_over(sec):
        m = d_s > sec
        n_tr = int(m.sum())
        n_ev = int(ps.loc[m, "event_id"].nunique()) if n_tr else 0
        return {"traces": n_tr, "events": n_ev, "frac_traces": float(m.mean())}

    dist = {
        "n_picker_train_traces": int(len(ev)),
        "n_ps_both": int(len(ps)),
        "n_p_only": int(len(po)),
        "n_events": int(ev["event_id"].nunique()),
        "ps_diff_samples": {
            "min": float(d.min()), "median": float(np.median(d)),
            "p75": pct(d, 75), "p90": pct(d, 90), "p95": pct(d, 95), "p99": pct(d, 99), "max": float(d.max()),
        },
        "ps_diff_seconds": {
            "min": float(d_s.min()), "median": float(np.median(d_s)),
            "p75": pct(d_s, 75), "p90": pct(d_s, 90), "p95": pct(d_s, 95), "p99": pct(d_s, 99), "max": float(d_s.max()),
        },
        "ps_gt_20s": n_over(20),
        "ps_gt_25s": n_over(25),
        "ps_gt_30s": n_over(30),
    }

    rng = np.random.default_rng(42)
    n = 12000
    # current v1 strategy: one 'both' midpoint crop per PS trace
    buckets_v1 = {"ps_both": 0, "p_only": 0, "s_only": 0, "neither": 0, "p_vis": 0, "s_vis": 0, "n": 0}
    trunc_v1 = {"p_near_edge": 0, "s_near_edge": 0}
    from earthquake.models.phasenet_finetune import random_crop_window

    for _, row in ps.iterrows():
        pv, sv = float(row.p_arrival_sample), float(row.s_arrival_sample)
        start = random_crop_window(n, WIN_RAW, pv, sv, "both", rng)
        vis_p, vis_s = phase_visible(pv, start), phase_visible(sv, start)
        buckets_v1[visibility_bucket(vis_p, vis_s)] += 1
        buckets_v1["p_vis"] += int(vis_p)
        buckets_v1["s_vis"] += int(vis_s)
        buckets_v1["n"] += 1
        pc = pv - start - FSTAB
        sc = sv - start - FSTAB
        if vis_p and (pc < 20 or pc > IN_SAMPLES - 20):
            trunc_v1["p_near_edge"] += 1
        if vis_s and (sc < 20 or sc > IN_SAMPLES - 20):
            trunc_v1["s_near_edge"] += 1

    # v2: three crops, never drop long PS
    buckets_v2 = {k: {"ps_both": 0, "p_only": 0, "s_only": 0, "neither": 0} for k in ["p_centered", "s_centered", "background"]}
    v2_any_p = v2_any_s = 0
    long_kept = 0
    for _, row in ps.iterrows():
        pv, sv = float(row.p_arrival_sample), float(row.s_arrival_sample)
        long = (sv - pv) / 100.0 > 30
        if long:
            long_kept += 1
        saw_p = saw_s = False
        for kind in ("p_centered", "s_centered", "background"):
            st = crop_kind_start(kind, n, pv, sv, rng)
            vis_p, vis_s = phase_visible(pv, st), phase_visible(sv, st)
            buckets_v2[kind][visibility_bucket(vis_p, vis_s)] += 1
            saw_p |= vis_p
            saw_s |= vis_s
        v2_any_p += int(saw_p)
        v2_any_s += int(saw_s)

    nps = max(len(ps), 1)
    v1_keys = ["ps_both", "p_only", "s_only", "neither"]
    for k in v1_keys:
        buckets_v1[k + "_frac"] = buckets_v1[k] / nps

    out = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "confirm_waveforms_read": False,
        "win_raw_samples": WIN_RAW,
        "label_window_samples": IN_SAMPLES,
        "fstab_samples": FSTAB,
        "distribution": dist,
        "v1_current_strategy": {
            "name": "single_midpoint_both_crop (random_crop_window mode=both)",
            "label_outside_still_in_loss_v1": True,
            "note": "v1 treated missing-in-crop S as noise/N via full-window softmax CE — invalid",
            "visibility_counts": buckets_v1,
            "truncated_near_edge_20samples": trunc_v1,
        },
        "v2_phase_visibility": {
            "crops_per_ps_trace": ["p_centered", "s_centered", "background"],
            "long_ps_traces_kept": long_kept,
            "frac_traces_with_any_P_crop": v2_any_p / nps,
            "frac_traces_with_any_S_crop": v2_any_s / nps,
            "per_kind": buckets_v2,
            "outside_phase_in_loss": False,
            "partial_label": True,
        },
    }
    dest = ensure_dir(ROOT / "artifacts/results/stage10/dkpn")
    save_json(out, dest / "crop_visibility_stats.json")
    md = f"""# DKPN 30 s crop & partial-label audit

**UTC:** {out['created_utc']}  
Confirm waveforms: **not read**. Split lists only (picker_train ∩ confirm = 0).

## Geometry

- Raw crop: **{WIN_RAW}** samples (~34.01 s) including {FSTAB} sample FP-stab
- Label / network window: **{IN_SAMPLES}** samples (~30.01 s) @ 100 Hz
- A 30 s crop **cannot** always hold both P and S; we do **not** require it.

## picker_train P–S (n={dist['n_ps_both']} traces, {dist['n_p_only']} P-only)

Seconds: min={dist['ps_diff_seconds']['min']:.3f} median={dist['ps_diff_seconds']['median']:.3f} P75={dist['ps_diff_seconds']['p75']:.3f} P90={dist['ps_diff_seconds']['p90']:.3f} P95={dist['ps_diff_seconds']['p95']:.3f} P99={dist['ps_diff_seconds']['p99']:.3f} max={dist['ps_diff_seconds']['max']:.3f}

| threshold | traces | events | frac traces |
|--|--:|--:|--:|
| >20 s | {dist['ps_gt_20s']['traces']} | {dist['ps_gt_20s']['events']} | {dist['ps_gt_20s']['frac_traces']:.4f} |
| >25 s | {dist['ps_gt_25s']['traces']} | {dist['ps_gt_25s']['events']} | {dist['ps_gt_25s']['frac_traces']:.4f} |
| >30 s | {dist['ps_gt_30s']['traces']} | {dist['ps_gt_30s']['events']} | {dist['ps_gt_30s']['frac_traces']:.4f} |

## v1 (buggy seed42) crop

Single midpoint `both` crop. Visibility fractions (one crop / PS trace):

- P+S visible: {buckets_v1['ps_both']/nps:.4f}
- P only: {buckets_v1['p_only']/nps:.4f}
- S only: {buckets_v1['s_only']/nps:.4f}
- neither: {buckets_v1['neither']/nps:.4f}

v1 **still applied full-window N/CE** when a phase was outside → that phase was trained as noise. **Invalid.**

## v2 policy

Per PS trace: **P-centered + S-centered + background**. Long P–S **not dropped** ({long_kept} traces >30 s).

- Any-crop covers P: {v2_any_p/nps:.4f}; covers S: {v2_any_s/nps:.4f}
- Outside phase: **not** a negative N class; use `-log[p(S)+p(N)]` or `-log[p(P)+p(N)]`
- Truncated Gaussians renormalized in-window; padding `pad_mask=0`
- Noise traces: full-window N only

See `crop_visibility_stats.json`.
"""
    (ROOT / "reports/stage10/dkpn_crop_and_partial_label_audit.md").write_text(md)
    print(json.dumps({"n_ps": dist["n_ps_both"], "gt30s": dist["ps_gt_30s"], "v1_both_frac": buckets_v1["ps_both"] / nps, "v2_S_cover": v2_any_s / nps}, indent=2))


if __name__ == "__main__":
    main()
