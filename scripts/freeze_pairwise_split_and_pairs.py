#!/usr/bin/env python
"""Freeze pairwise pilot event split + c1/c2 definitions on Stage-6 ranker_train only.

Never reads confirm or phaseB/full-dev manifests for scoring.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, save_json
from earthquake.multistation.soft_ring_rescore import absolute_travel_time_s, fixed_candidate_scores
from earthquake.utils import ensure_dir

OUT = artifacts_dir() / "results" / "pairwise_pilot"
SEED = 42
TOL_S = 0.5
TIE_AE_DIFF_S = 0.1


def _sha_text(lines: list[str]) -> str:
    payload = ("\n".join(lines) + "\n").encode()
    return hashlib.sha256(payload).hexdigest()


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    out = ensure_dir(OUT)
    # --- load ranker_train only ---
    union_path = artifacts_dir() / "cache" / "stage6" / "phaseC" / "ranker_train_union.parquet"
    man_path = artifacts_dir() / "results" / "stage6" / "phaseC" / "ranker_train_s_manifest.csv"
    hist_path = artifacts_dir() / "models" / "stage6" / "history_picker_train" / "residual_history_features.parquet"

    # hard guards
    for bad in ["confirm", "phaseB_eval_manifest", "dev_union"]:
        assert bad not in str(union_path)
        assert bad not in str(man_path)

    union = pd.read_parquet(union_path)
    meta = pd.read_csv(man_path)
    hist = pd.read_parquet(hist_path)
    hist = hist[hist["subset"] == "stage6_ranker_train"].drop_duplicates("trace_name")

    # Prefer precomputed Stage-6 expected_s via residual history + attach in vectorized form.
    # hist already has base_tau_*; map expected using same formulas as attach_expected_s but vectorized.
    from earthquake.config import load_json

    hist_man = load_json(artifacts_dir() / "results" / "stage6" / "history_picker_train_manifest.json")
    global_res = hist_man["global_residual"]
    rs_g = float(global_res["residual_s_median"])

    meta = meta.copy()
    meta["trace_name"] = meta["trace_name"].astype(str)
    hist_cols = [c for c in hist.columns if c not in {"subset", "event_id"}]
    merged = meta.merge(hist[hist_cols], on="trace_name", how="left")

    # Vectorized expected_s_sample (catalog-assisted), matching Stage-6 shrink policy.
    hav = merged["history_available"].to_numpy(bool) if "history_available" in merged.columns else np.zeros(len(merged), dtype=bool)
    hcount = pd.to_numeric(merged.get("history_count"), errors="coerce").to_numpy(float)
    rsm = pd.to_numeric(merged.get("residual_s_median"), errors="coerce").to_numpy(float)
    rmad = pd.to_numeric(merged.get("residual_s_mad"), errors="coerce").to_numpy(float)
    base_tau = pd.to_numeric(merged.get("base_tau_s"), errors="coerce").to_numpy(float)
    base_dsp = pd.to_numeric(merged.get("base_delta_sp"), errors="coerce").to_numpy(float)
    use_hist = hav & np.isfinite(hcount) & (hcount >= 5) & np.isfinite(rmad) & (rmad <= 1.0) & np.isfinite(rsm)
    # shrink residual toward global
    w = np.where(use_hist, hcount / (hcount + 50.0), 0.0)
    rs = np.where(use_hist, w * rsm + (1.0 - w) * rs_g, rs_g)
    pred_tau_s = base_tau + rs
    # sigma_s seconds -> samples
    sig_s = np.where(use_hist & np.isfinite(rmad), np.clip(1.4826 * rmad, 0.05, 1.0), 1.0)
    sr = pd.to_numeric(merged["sampling_rate_hz"], errors="coerce").to_numpy(float)
    origin = pd.to_datetime(merged["origin_time"], utc=True, errors="coerce")
    start = pd.to_datetime(merged["trace_start_time"], utc=True, errors="coerce")
    offset = (start - origin).dt.total_seconds().to_numpy(float)
    expected_s_sample = (pred_tau_s - offset) * sr
    exp_df = pd.DataFrame(
        {
            "trace_name": merged["trace_name"].astype(str),
            "expected_s_sample": expected_s_sample,
            "history_sigma_samples": sig_s * sr,
            "history_available": use_hist.astype(float),
            "base_tau_s": base_tau,
            "base_delta_sp": base_dsp,
        }
    )

    u = union.merge(exp_df, on="trace_name", how="left")
    u = u.merge(
        meta[
            [
                "trace_name",
                "origin_time",
                "trace_start_time",
                "snr_db",
                "network",
                "station",
            ]
        ],
        on="trace_name",
        how="left",
    )
    # pred P
    u["pred_p_sample"] = np.where(
        np.isfinite(u["top1_p_stead"].to_numpy(float)),
        u["top1_p_stead"].to_numpy(float),
        u["top1_p_ida"].to_numpy(float),
    )
    u["cand_prob"] = np.fmax(
        u["stead_probability"].to_numpy(float),
        u["ida_probability"].to_numpy(float),
    )
    # where both nan, use 0
    u["cand_prob"] = np.where(np.isfinite(u["cand_prob"]), u["cand_prob"], 0.0)

    u["fixed_score"] = fixed_candidate_scores(
        cand_sample=u["candidate_sample"].to_numpy(float),
        cand_prob=u["cand_prob"].to_numpy(float),
        expected_s_sample=u["expected_s_sample"].to_numpy(float),
        history_sigma_samples=u["history_sigma_samples"].to_numpy(float),
        history_available=u["history_available"].to_numpy(bool),
        lw=0.5,
        lh=2.0,
        lp=0.0,
    )
    u["tau_s"] = absolute_travel_time_s(
        origin_time=u["origin_time"],
        trace_start_time=u["trace_start_time"],
        sample=u["candidate_sample"].to_numpy(float),
        sampling_rate_hz=u["sampling_rate_hz"].to_numpy(float),
    )
    u["resid_s"] = u["tau_s"] - u["base_tau_s"].to_numpy(float)
    u["delta_sp"] = (
        u["candidate_sample"].to_numpy(float) - u["pred_p_sample"].to_numpy(float)
    ) / u["sampling_rate_hz"].to_numpy(float)
    u["resid_sp"] = u["delta_sp"] - u["base_delta_sp"].to_numpy(float)
    u["source"] = u["candidate_source_mask"].fillna("unknown").astype(str)

    # --- define c1/c2 by frozen fixed_score only ---
    pairs = []
    rng_tie = np.random.default_rng(0)  # unused; ties broken by first occurrence after sort
    for tn, g in u.groupby("trace_name", sort=False):
        g = g.sort_values(["fixed_score", "candidate_index"], ascending=[False, True])
        sr = float(g["sampling_rate_hz"].iloc[0])
        true_s = float(g["true_s_sample"].iloc[0])
        c1 = g.iloc[0]
        n_cand = len(g)
        if n_cand < 2:
            c2 = None
        else:
            c2 = g.iloc[1]
        def ae(row):
            return abs(float(row["candidate_sample"]) - true_s) / sr

        if c2 is None:
            label_class = "single_candidate"
            y = None
            ae1, ae2 = ae(c1), np.nan
            c1_ok, c2_ok = ae1 <= TOL_S, False
        else:
            ae1, ae2 = ae(c1), ae(c2)
            c1_ok, c2_ok = ae1 <= TOL_S, ae2 <= TOL_S
            if abs(ae1 - ae2) < TIE_AE_DIFF_S:
                label_class = "tie_ae_close"
                y = None
            elif c1_ok and not c2_ok:
                label_class = "choose_c1"
                y = 0
            elif (not c1_ok) and c2_ok:
                label_class = "choose_c2"
                y = 1
            elif c1_ok and c2_ok:
                label_class = "both_correct"
                y = None
            else:
                label_class = "both_wrong"
                y = None
        row = {
            "trace_name": str(tn),
            "event_id": str(g["event_id"].iloc[0]),
            "n_candidates": int(n_cand),
            "sampling_rate_hz": sr,
            "true_s_sample": true_s,  # label column — NOT a model feature
            "c1_sample": float(c1["candidate_sample"]),
            "c1_index": int(c1["candidate_index"]),
            "c1_fixed_score": float(c1["fixed_score"]),
            "c1_prob": float(c1["cand_prob"]),
            "c1_source": str(c1["source"]),
            "c1_resid_s": float(c1["resid_s"]),
            "c1_resid_sp": float(c1["resid_sp"]),
            "c1_stead_prob": float(c1["stead_probability"]) if np.isfinite(c1["stead_probability"]) else 0.0,
            "c1_ida_prob": float(c1["ida_probability"]) if np.isfinite(c1["ida_probability"]) else 0.0,
            "c1_delta_sp": float(c1["delta_sp"]),
            "c2_sample": float(c2["candidate_sample"]) if c2 is not None else np.nan,
            "c2_index": int(c2["candidate_index"]) if c2 is not None else -1,
            "c2_fixed_score": float(c2["fixed_score"]) if c2 is not None else np.nan,
            "c2_prob": float(c2["cand_prob"]) if c2 is not None else np.nan,
            "c2_source": str(c2["source"]) if c2 is not None else "",
            "c2_resid_s": float(c2["resid_s"]) if c2 is not None else np.nan,
            "c2_resid_sp": float(c2["resid_sp"]) if c2 is not None else np.nan,
            "c2_stead_prob": float(c2["stead_probability"]) if (c2 is not None and np.isfinite(c2["stead_probability"])) else 0.0,
            "c2_ida_prob": float(c2["ida_probability"]) if (c2 is not None and np.isfinite(c2["ida_probability"])) else 0.0,
            "c2_delta_sp": float(c2["delta_sp"]) if c2 is not None else np.nan,
            "margin_fixed": float(c1["fixed_score"] - (c2["fixed_score"] if c2 is not None else c1["fixed_score"])),
            "ae_c1": float(ae1),
            "ae_c2": float(ae2) if c2 is not None else np.nan,
            "c1_ok": bool(c1_ok),
            "c2_ok": bool(c2_ok) if c2 is not None else False,
            "label_class": label_class,
            "y_choose_c2": y,
            "snr_db": float(g["snr_db"].iloc[0]) if "snr_db" in g.columns else np.nan,
            "distance_km": float(g["distance_km"].iloc[0]),
            "origin_time": str(g["origin_time"].iloc[0]),
            "trace_start_time": str(g["trace_start_time"].iloc[0]),
            "network": str(g["network"].iloc[0]) if "network" in g.columns else "",
            "station": str(g["station"].iloc[0]) if "station" in g.columns else "",
        }
        pairs.append(row)
    pairs_df = pd.DataFrame(pairs)

    # quadrants among >=2
    ge2 = pairs_df[pairs_df.n_candidates >= 2]
    q = {
        "n_traces_total": int(len(pairs_df)),
        "n_events_total": int(pairs_df.event_id.nunique()),
        "n_ge2": int(len(ge2)),
        "Q1_c1_ok_c2_bad": int(((ge2.c1_ok) & (~ge2.c2_ok)).sum()),
        "Q2_c1_bad_c2_ok": int(((~ge2.c1_ok) & (ge2.c2_ok)).sum()),
        "both_correct": int(((ge2.c1_ok) & (ge2.c2_ok)).sum()),
        "both_wrong": int(((~ge2.c1_ok) & (~ge2.c2_ok)).sum()),
        "label_class_counts": pairs_df.label_class.value_counts().to_dict(),
    }
    # top2 oracle + error decomposition
    c1_correct = float(pairs_df.c1_ok.mean())
    top2_oracle = float(((pairs_df.c1_ok) | (pairs_df.c2_ok)).mean())
    n_wrong = int((~pairs_df.c1_ok).sum())
    n_rec = int(((~pairs_df.c1_ok) & (pairs_df.c2_ok)).sum())  # recoverable by switching to c2
    # candidate-missing: c1 wrong and no c2 correct (includes single-cand and both wrong)
    n_miss_pool = int(((~pairs_df.c1_ok) & (~pairs_df.c2_ok)).sum())
    q.update(
        {
            "baseline_c1_acc_at_0.5": c1_correct,
            "top2_candidate_oracle_acc_at_0.5": top2_oracle,
            "n_baseline_wrong": n_wrong,
            "n_ranking_recoverable_via_c2": n_rec,
            "n_candidate_missing_or_both_wrong": n_miss_pool,
            "frac_wrong_that_are_ranking_recoverable": float(n_rec / max(n_wrong, 1)),
            "frac_wrong_candidate_missing": float(n_miss_pool / max(n_wrong, 1)),
            "note_Q2_phaseB_was_1953": "phaseB Q2=1953 is a different split; this freeze is ranker_train only",
        }
    )

    # --- event split 70/15/15 ---
    events = np.array(sorted(pairs_df.event_id.astype(str).unique()))
    rng = np.random.default_rng(SEED)
    rng.shuffle(events)
    n = len(events)
    n_train = int(round(0.70 * n))
    n_cal = int(round(0.15 * n))
    train_e = set(events[:n_train].tolist())
    cal_e = set(events[n_train : n_train + n_cal].tolist())
    eval_e = set(events[n_train + n_cal :].tolist())
    assert train_e.isdisjoint(cal_e) and train_e.isdisjoint(eval_e) and cal_e.isdisjoint(eval_e)

    def assign(eid):
        eid = str(eid)
        if eid in train_e:
            return "train"
        if eid in cal_e:
            return "calibration"
        return "heldout_eval"

    pairs_df["split"] = pairs_df.event_id.astype(str).map(assign)

    # hashes
    split_lock = {
        "seed": SEED,
        "protocol": "ranker_train_only_event_split_70_15_15",
        "n_events": {"train": len(train_e), "calibration": len(cal_e), "heldout_eval": len(eval_e)},
        "n_traces": pairs_df.groupby("split").size().to_dict(),
        "overlap_event_train_cal": 0,
        "overlap_event_train_eval": 0,
        "overlap_event_cal_eval": 0,
        "union_parquet_sha256": _sha_file(union_path),
        "manifest_sha256": _sha_file(man_path),
        "history_features_sha256": hist_man.get("features_sha256"),
        "baseline_mlp_sha256": hist_man.get("baseline_sha256"),
        "c1_c2_definition": "c1=argmax fixed_score(lw=0.5,lh=2,lp=0); c2=second by same score then candidate_index",
        "forbidden": ["confirm", "phaseB_full_dev", "neighbor_labels"],
        "event_sha256": {
            "train": _sha_text(sorted(train_e)),
            "calibration": _sha_text(sorted(cal_e)),
            "heldout_eval": _sha_text(sorted(eval_e)),
        },
        "trace_sha256": {
            s: _sha_text(sorted(pairs_df.loc[pairs_df.split == s, "trace_name"].astype(str).tolist()))
            for s in ["train", "calibration", "heldout_eval"]
        },
    }
    # prove overlap 0
    assert len(train_e & cal_e) == 0
    assert len(train_e & eval_e) == 0
    assert len(cal_e & eval_e) == 0

    pairs_df.to_parquet(out / "pairs_ranker_train.parquet", index=False)
    # also save event lists
    for name, eset in [("train", train_e), ("calibration", cal_e), ("heldout_eval", eval_e)]:
        (out / f"events_{name}.txt").write_text("\n".join(sorted(eset)) + "\n")
        tr = pairs_df.loc[pairs_df.split == name, "trace_name"].astype(str).sort_values()
        (out / f"traces_{name}.txt").write_text("\n".join(tr.tolist()) + "\n")

    save_json(split_lock, out / "SPLIT.LOCK.json")
    save_json({"quadrants_and_pool": q, "split": split_lock["n_traces"]}, out / "dataset_summary.json")

    print(json.dumps(q, indent=2))
    print("split", split_lock["n_events"], split_lock["n_traces"])
    print("Wrote", out)


if __name__ == "__main__":
    main()
