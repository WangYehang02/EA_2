#!/usr/bin/env python
"""Generate STEAD/IDA UNION candidates + c1/c2 pairs (multi-GPU shardable).

Usage:
  python build_independent_candidates_pairs.py              # all GPUs, full run
  python build_independent_candidates_pairs.py --shard 0 8  # one shard
  python build_independent_candidates_pairs.py --merge-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from obspy import read

for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
    os.environ.pop(k, None)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import load_json
from earthquake.gating.cache_io import attach_expected_s
from earthquake.history.residual_prior import path_stats_to_residual_stats
from earthquake.models.seisbench_reference import SeisBenchPhaseNetReference
from earthquake.multistation.soft_ring_rescore import absolute_travel_time_s, fixed_candidate_scores
from earthquake.stage6.bn_policy import set_train_bn_eval
from earthquake.stage6.phaseB import candidates_from_s_proba
from earthquake.stage6.ranker.union_schema import union_candidates_phaseC

OUT = Path("/data/yehang/Earthquake_paper_strengthening_v1/independent_period")
WAVE = OUT / "waveforms"
CAND = OUT / "candidates"
TOL_S = 0.5
TIE_AE_DIFF_S = 0.1
IDA_CKPT = ROOT / "artifacts/models/stage6/phasenet_ida_full_seed42/checkpoints/best.pt"
HIST_PKL = ROOT / "artifacts/models/stage6/history_picker_train/temporal_history_store.pkl"
MLP_PKL = ROOT / "artifacts/models/stage6/history_picker_train/travel_time_baseline_mlp.pkl"
HIST_MAN = ROOT / "artifacts/results/stage6/history_picker_train_manifest.json"


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def key_hash(trace_key: str) -> str:
    return hashlib.sha1(trace_key.encode()).hexdigest()


def load_enz(mseed_path: str) -> np.ndarray:
    st = read(mseed_path)
    by = {tr.stats.channel[-1].upper(): tr for tr in st}
    return np.stack([by["E"].data, by["N"].data, by["Z"].data], axis=0).astype(np.float32)


def load_ida(device: str):
    import seisbench.models as sbm

    blob = torch.load(IDA_CKPT, map_location="cpu", weights_only=False)
    assert int(blob.get("epoch", -1)) == 14, blob.get("epoch")
    model = sbm.PhaseNet.from_pretrained("stead")
    model.load_state_dict(blob["model"], strict=True)
    set_train_bn_eval(model)
    model.to(device)
    model.eval()
    return model


def run_shard(shard_id: int, n_shards: int, device: str) -> Path:
    CAND.mkdir(parents=True, exist_ok=True)
    status = pd.read_parquet(OUT / "download" / "download_status.parquet")
    ok = status[status["reason"] == "ok"].reset_index(drop=True)
    # deterministic shard by sha1 of trace_key
    mask = []
    for tk in ok["trace_key"].astype(str):
        h = int(hashlib.sha1(tk.encode()).hexdigest()[:8], 16)
        mask.append((h % n_shards) == shard_id)
    ok = ok.loc[mask].reset_index(drop=True)
    print(f"shard {shard_id}/{n_shards} device={device} n={len(ok)}", flush=True)

    stead_ref = SeisBenchPhaseNetReference(weight="stead", device=device)
    ida_model = load_ida(device)
    ida_ref = SeisBenchPhaseNetReference(weight="stead", device=device)
    ida_ref.model = ida_model

    with open(HIST_PKL, "rb") as f:
        store = pickle.load(f)
    with open(MLP_PKL, "rb") as f:
        mlp = pickle.load(f)
    hist_man = load_json(HIST_MAN)
    global_res = hist_man["global_residual"]
    shrink_k = float(hist_man.get("shrinkage_k", 50.0))
    min_history = int(hist_man.get("min_history", 5))

    stead_rows, ida_rows, meta_rows, fail_inf = [], [], [], []
    t0 = time.time()
    for i, (_, r) in enumerate(ok.iterrows(), 1):
        tn = str(r["trace_key"])
        hid = key_hash(tn)
        mseed = r.get("mseed_path") or str(WAVE / f"{hid}.mseed")
        try:
            wave = load_enz(mseed)

            def _utc(x):
                return pd.to_datetime(x, utc=True)

            meta = {
                "trace_name": tn,
                "trace_key": tn,
                "event_id": str(r["event_id"]),
                "network": str(r["network"]),
                "station": str(r["station"]),
                "location": str(r.get("location") or ""),
                "channel_prefix": str(r["channel_prefix"]),
                "origin_time": _utc(r["origin_time"]),
                "trace_start_time": _utc(r["trace_start_time"]),
                "sampling_rate_hz": float(r["sampling_rate_hz"]),
                "source_latitude": float(r["source_latitude"]),
                "source_longitude": float(r["source_longitude"]),
                "source_depth_km": float(r["source_depth_km"]) if pd.notna(r["source_depth_km"]) else 10.0,
                "station_latitude": float(r["station_latitude"]),
                "station_longitude": float(r["station_longitude"]),
                "station_elevation_m": float(r["station_elevation_m"]),
                "distance_km": float(r["distance_km"]),
                "s_arrival_sample": float(r["s_arrival_sample"]),
                "true_s_sample": float(r["s_arrival_sample"]),
                "mseed_sha256": r.get("mseed_sha256"),
                "mseed_path": mseed,
                "pick_public_id": r.get("pick_public_id"),
                "evaluation_mode": r.get("evaluation_mode"),
                "phase_hint": r.get("phase_hint"),
            }
            srow = pd.Series(meta)
            base = mlp.predict_row(
                pd.Series(
                    {
                        "distance_km": meta["distance_km"],
                        "source_depth_km": meta["source_depth_km"],
                        "station_elevation_m": meta["station_elevation_m"],
                    }
                )
            )
            meta["base_tau_p"] = float(base["base_tau_p"])
            meta["base_tau_s"] = float(base["base_tau_s"])
            meta["base_delta_sp"] = float(base["base_delta_sp"])
            stats = store.query_row(srow)
            rs = path_stats_to_residual_stats(
                stats,
                {
                    "base_tau_p": meta["base_tau_p"],
                    "base_tau_s": meta["base_tau_s"],
                    "base_delta_sp": meta["base_delta_sp"],
                },
            )
            for k, v in rs.__dict__.items():
                meta[k] = v
            exp = attach_expected_s(
                pd.Series(meta),
                global_res=global_res,
                shrink_k=shrink_k,
                min_history=min_history,
                mad_disable_s=1.0,
                min_sigma_s=0.05,
                max_sigma_s=1.0,
            )
            meta.update(exp)

            with torch.no_grad():
                pred_s = stead_ref.predict_row(wave, pd.Series(meta), remap_to_waveform=True)
                pred_i = ida_ref.predict_row(wave, pd.Series(meta), remap_to_waveform=True)
            s_proba = np.asarray(pred_s["s_proba_on_waveform"], dtype=float)
            i_proba = np.asarray(pred_i["s_proba_on_waveform"], dtype=float)
            p_s = np.asarray(pred_s["p_proba_on_waveform"], dtype=float)
            p_i = np.asarray(pred_i["p_proba_on_waveform"], dtype=float)
            wstart = meta["trace_start_time"]
            sr = 100.0
            sc = candidates_from_s_proba(s_proba, sampling_rate=sr, waveform_starttime=wstart, k=10, source="stead")
            ic = candidates_from_s_proba(i_proba, sampling_rate=sr, waveform_starttime=wstart, k=10, source="ida")
            top1_s = int(np.argmax(s_proba)) if np.isfinite(s_proba).any() else -1
            top1_p_s = int(np.argmax(p_s)) if np.isfinite(p_s).any() else -1
            top1_p_i = int(np.argmax(p_i)) if np.isfinite(p_i).any() else -1
            for c in sc:
                stead_rows.append(
                    {
                        "trace_name": tn,
                        "event_id": meta["event_id"],
                        "sampling_rate_hz": sr,
                        "true_s_sample": meta["true_s_sample"],
                        "top1_s_sample": top1_s,
                        "top1_p_sample": top1_p_s,
                        "top1_p_prob": float(p_s[top1_p_s]) if top1_p_s >= 0 else np.nan,
                        **c,
                    }
                )
            for c in ic:
                ida_rows.append(
                    {
                        "trace_name": tn,
                        "event_id": meta["event_id"],
                        "sampling_rate_hz": sr,
                        "true_s_sample": meta["true_s_sample"],
                        "top1_s_sample": int(np.argmax(i_proba)) if np.isfinite(i_proba).any() else -1,
                        "top1_p_sample": top1_p_i,
                        "top1_p_prob": float(p_i[top1_p_i]) if top1_p_i >= 0 else np.nan,
                        **c,
                    }
                )
            # serialize times as str for parquet
            meta_out = dict(meta)
            meta_out["origin_time"] = str(meta["origin_time"])
            meta_out["trace_start_time"] = str(meta["trace_start_time"])
            meta_rows.append(meta_out)
        except Exception as e:
            fail_inf.append({"trace_key": tn, "reason": f"infer:{type(e).__name__}:{e}"})
        if i % 25 == 0 or i == len(ok):
            print(f"shard{shard_id} {i}/{len(ok)} fail={len(fail_inf)} elapsed={time.time()-t0:.0f}s", flush=True)

    outp = CAND / f"shard_{shard_id:02d}"
    outp.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(meta_rows).to_parquet(outp / "meta.parquet", index=False)
    pd.DataFrame(stead_rows).to_parquet(outp / "stead_candidates.parquet", index=False)
    pd.DataFrame(ida_rows).to_parquet(outp / "ida_candidates.parquet", index=False)
    pd.DataFrame(fail_inf).to_csv(outp / "inference_failures.csv", index=False)
    (outp / "DONE.json").write_text(json.dumps({"n_meta": len(meta_rows), "n_fail": len(fail_inf)}) + "\n")
    return outp


def merge_and_pairs() -> int:
    CAND.mkdir(parents=True, exist_ok=True)
    shards = sorted(CAND.glob("shard_*"))
    if not shards:
        raise SystemExit("no shards")
    metas, steads, idas, fails = [], [], [], []
    for s in shards:
        metas.append(pd.read_parquet(s / "meta.parquet"))
        steads.append(pd.read_parquet(s / "stead_candidates.parquet"))
        idas.append(pd.read_parquet(s / "ida_candidates.parquet"))
        fp = s / "inference_failures.csv"
        if fp.exists() and fp.stat().st_size > 10:
            try:
                fails.append(pd.read_csv(fp))
            except pd.errors.EmptyDataError:
                pass
    meta_df = pd.concat(metas, ignore_index=True)
    stead_df = pd.concat(steads, ignore_index=True)
    ida_df = pd.concat(idas, ignore_index=True)
    fail_df = pd.concat(fails, ignore_index=True) if fails else pd.DataFrame()
    meta_df.to_parquet(CAND / "meta.parquet", index=False)
    stead_df.to_parquet(CAND / "stead_candidates.parquet", index=False)
    ida_df.to_parquet(CAND / "ida_candidates.parquet", index=False)
    fail_df.to_csv(CAND / "inference_failures.csv", index=False)

    status = pd.read_parquet(OUT / "download" / "download_status.parquet")
    outside = status[status["reason"] == "FAIL_LABEL_OUTSIDE_WINDOW"].copy()

    union = union_candidates_phaseC(stead_df, ida_df, stead_k=5, ida_k=5, max_union=10)
    if "snr_db" not in meta_df.columns:
        meta_df["snr_db"] = np.nan
    exp_cols = [
        "trace_name",
        "expected_s_sample",
        "history_sigma_samples",
        "history_available",
        "base_tau_s",
        "base_delta_sp",
        "distance_km",
        "origin_time",
        "trace_start_time",
        "network",
        "station",
        "snr_db",
        "true_s_sample",
        "event_id",
    ]
    u = union.merge(meta_df[[c for c in exp_cols if c in meta_df.columns]], on="trace_name", how="left", suffixes=("", "_m"))
    if "event_id_m" in u.columns:
        u["event_id"] = u["event_id"].fillna(u["event_id_m"])
    if "true_s_sample_m" in u.columns:
        u["true_s_sample"] = u["true_s_sample"].fillna(u["true_s_sample_m"])
    u["cand_prob"] = np.fmax(
        pd.to_numeric(u.get("stead_probability"), errors="coerce").to_numpy(float),
        pd.to_numeric(u.get("ida_probability"), errors="coerce").to_numpy(float),
    )
    u["cand_prob"] = np.where(np.isfinite(u["cand_prob"]), u["cand_prob"], 0.0)
    hav = pd.to_numeric(u["history_available"], errors="coerce").fillna(0).to_numpy(float) > 0.5
    u["fixed_score"] = fixed_candidate_scores(
        cand_sample=u["candidate_sample"].to_numpy(float),
        cand_prob=u["cand_prob"].to_numpy(float),
        expected_s_sample=u["expected_s_sample"].to_numpy(float),
        history_sigma_samples=u["history_sigma_samples"].to_numpy(float),
        history_available=hav,
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
    u["pred_p_sample"] = np.where(
        np.isfinite(pd.to_numeric(u.get("top1_p_stead"), errors="coerce")),
        pd.to_numeric(u.get("top1_p_stead"), errors="coerce"),
        pd.to_numeric(u.get("top1_p_ida"), errors="coerce"),
    )
    u["delta_sp"] = (u["candidate_sample"].to_numpy(float) - u["pred_p_sample"].to_numpy(float)) / u[
        "sampling_rate_hz"
    ].to_numpy(float)
    u["resid_sp"] = u["delta_sp"] - u["base_delta_sp"].to_numpy(float)
    u["source"] = u["candidate_source_mask"].fillna("unknown").astype(str)
    u.to_parquet(CAND / "union_scored.parquet", index=False)

    pairs = []
    for tn, g in u.groupby("trace_name", sort=False):
        g = g.sort_values(["fixed_score", "candidate_index"], ascending=[False, True])
        sr = float(g["sampling_rate_hz"].iloc[0])
        true_s = float(g["true_s_sample"].iloc[0])
        c1 = g.iloc[0]
        n_cand = len(g)
        c2 = g.iloc[1] if n_cand >= 2 else None

        def ae(row):
            return abs(float(row["candidate_sample"]) - true_s) / sr

        if c2 is None:
            label_class, y = "single_candidate", np.nan
            ae1, ae2 = ae(c1), np.nan
            c1_ok, c2_ok = ae1 <= TOL_S, False
        else:
            ae1, ae2 = ae(c1), ae(c2)
            c1_ok, c2_ok = ae1 <= TOL_S, ae2 <= TOL_S
            if abs(ae1 - ae2) < TIE_AE_DIFF_S:
                label_class, y = "tie_ae_close", np.nan
            elif c1_ok and not c2_ok:
                label_class, y = "choose_c1", 0
            elif (not c1_ok) and c2_ok:
                label_class, y = "choose_c2", 1
            elif c1_ok and c2_ok:
                label_class, y = "both_correct", np.nan
            else:
                label_class, y = "both_wrong", np.nan

        def feat(row, prefix):
            if row is None:
                return {
                    f"{prefix}_sample": np.nan,
                    f"{prefix}_index": -1,
                    f"{prefix}_fixed_score": np.nan,
                    f"{prefix}_prob": np.nan,
                    f"{prefix}_source": "",
                    f"{prefix}_resid_s": np.nan,
                    f"{prefix}_resid_sp": np.nan,
                    f"{prefix}_stead_prob": 0.0,
                    f"{prefix}_ida_prob": 0.0,
                    f"{prefix}_delta_sp": np.nan,
                }
            sp = float(row.get("stead_probability", np.nan))
            ip = float(row.get("ida_probability", np.nan))
            return {
                f"{prefix}_sample": float(row["candidate_sample"]),
                f"{prefix}_index": int(row["candidate_index"]),
                f"{prefix}_fixed_score": float(row["fixed_score"]),
                f"{prefix}_prob": float(row["cand_prob"]),
                f"{prefix}_source": str(row["source"]),
                f"{prefix}_resid_s": float(row["resid_s"]),
                f"{prefix}_resid_sp": float(row["resid_sp"]),
                f"{prefix}_stead_prob": sp if np.isfinite(sp) else 0.0,
                f"{prefix}_ida_prob": ip if np.isfinite(ip) else 0.0,
                f"{prefix}_delta_sp": float(row["delta_sp"]),
            }

        pairs.append(
            {
                "trace_name": str(tn),
                "event_id": str(g["event_id"].iloc[0]),
                "n_candidates": int(n_cand),
                "sampling_rate_hz": sr,
                "true_s_sample": true_s,
                **feat(c1, "c1"),
                **feat(c2, "c2"),
                "margin_fixed": float(c1["fixed_score"] - (c2["fixed_score"] if c2 is not None else c1["fixed_score"])),
                "ae_c1": float(ae1),
                "ae_c2": float(ae2) if c2 is not None else np.nan,
                "c1_ok": bool(c1_ok),
                "c2_ok": bool(c2_ok),
                "label_class": label_class,
                "y_choose_c2": y,
                "snr_db": float(g["snr_db"].iloc[0]) if "snr_db" in g.columns else np.nan,
                "distance_km": float(g["distance_km"].iloc[0]) if "distance_km" in g.columns else np.nan,
                "origin_time": str(g["origin_time"].iloc[0]),
                "trace_start_time": str(g["trace_start_time"].iloc[0]),
                "network": str(g["network"].iloc[0]) if "network" in g.columns else "",
                "station": str(g["station"].iloc[0]) if "station" in g.columns else "",
                "split": "independent_period",
            }
        )
    pairs_df = pd.DataFrame(pairs)
    pairs_path = CAND / "pairs_independent_period.parquet"
    pairs_df.to_parquet(pairs_path, index=False)

    out_rows = []
    for _, r in outside.iterrows():
        out_rows.append(
            {
                "trace_name": str(r["trace_key"]),
                "event_id": str(r["event_id"]),
                "n_candidates": 0,
                "sampling_rate_hz": 100.0,
                "true_s_sample": float(r.get("s_arrival_sample", np.nan)),
                "c1_sample": np.nan,
                "c1_index": -1,
                "c1_fixed_score": np.nan,
                "c1_prob": np.nan,
                "c1_source": "",
                "c1_resid_s": np.nan,
                "c1_resid_sp": np.nan,
                "c1_stead_prob": 0.0,
                "c1_ida_prob": 0.0,
                "c1_delta_sp": np.nan,
                "c2_sample": np.nan,
                "c2_index": -1,
                "c2_fixed_score": np.nan,
                "c2_prob": np.nan,
                "c2_source": "",
                "c2_resid_s": np.nan,
                "c2_resid_sp": np.nan,
                "c2_stead_prob": 0.0,
                "c2_ida_prob": 0.0,
                "c2_delta_sp": np.nan,
                "margin_fixed": np.nan,
                "ae_c1": np.nan,
                "ae_c2": np.nan,
                "c1_ok": False,
                "c2_ok": False,
                "label_class": "FAIL_LABEL_OUTSIDE_WINDOW",
                "y_choose_c2": np.nan,
                "snr_db": np.nan,
                "distance_km": float(r["distance_km"]) if pd.notna(r.get("distance_km")) else np.nan,
                "origin_time": str(r.get("origin_time")),
                "trace_start_time": str(r.get("trace_start_time")),
                "network": str(r.get("network")),
                "station": str(r.get("station")),
                "split": "independent_period_outside_window",
            }
        )
    if out_rows:
        pd.concat([pairs_df, pd.DataFrame(out_rows)], ignore_index=True).to_parquet(
            CAND / "pairs_independent_period_with_window_fails.parquet", index=False
        )

    lock = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_pairs": int(len(pairs_df)),
        "n_outside_window": int(len(outside)),
        "n_infer_fail": int(len(fail_df)),
        "ida_ckpt_sha256": sha_file(IDA_CKPT),
        "pairs_sha256": sha_file(pairs_path),
        "meta_sha256": sha_file(CAND / "meta.parquet"),
        "union_sha256": sha_file(CAND / "union_scored.parquet"),
        "history_updated_with_test_labels": False,
        "n_shards": len(shards),
    }
    (OUT / "locks").mkdir(exist_ok=True)
    (OUT / "locks" / "CANDIDATES.LOCK.json").write_text(json.dumps(lock, indent=2) + "\n")
    print(json.dumps(lock, indent=2))
    return 0


def launch_all_gpus() -> int:
    import subprocess

    n = torch.cuda.device_count() if torch.cuda.is_available() else 1
    n = max(n, 1)
    procs = []
    for i in range(n):
        device = f"cuda:{i}" if torch.cuda.is_available() else "cpu"
        cmd = [
            sys.executable,
            "-W",
            "ignore",
            str(Path(__file__).resolve()),
            "--shard",
            str(i),
            str(n),
            "--device",
            device,
        ]
        log = open(OUT / "logs" / f"cand_shard_{i}.log", "w")
        procs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT))
    rc = 0
    for p in procs:
        rc = max(rc, p.wait())
    if rc != 0:
        return rc
    return merge_and_pairs()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", nargs=2, type=int, metavar=("ID", "N"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--merge-only", action="store_true")
    args = ap.parse_args()
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    CAND.mkdir(parents=True, exist_ok=True)
    if args.merge_only:
        return merge_and_pairs()
    if args.shard:
        run_shard(args.shard[0], args.shard[1], args.device)
        return 0
    return launch_all_gpus()


if __name__ == "__main__":
    raise SystemExit(main())
