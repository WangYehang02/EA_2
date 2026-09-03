#!/usr/bin/env python
"""Stage-4 confirmatory evaluation on frozen holdout (no hyperparameter search)."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, load_json, load_yaml_config, resolve_instance_root, save_json
from earthquake.data.hdf5_reader import InstanceHDF5Reader
from earthquake.fusion.candidate_rescorer import pick_argmax_probability, rescore_phase_candidates
from earthquake.fusion.peak_candidates import PeakCandidate, extract_candidates, candidates_to_records
from earthquake.gating.cache_io import attach_expected_s, build_cand_index, residual_stats_from_row
from earthquake.gating.features import load_scaler
from earthquake.gating.inference import predict_gate_pick
from earthquake.gating.scalar_gate import ScalarGate
from earthquake.gating.features import build_trace_features, FEATURE_COLUMNS_CATALOG
from earthquake.history.residual_prior import expected_samples_catalog, predict_with_residual_prior
from earthquake.metrics import audit_pick_errors, match_picks
from earthquake.models.seisbench_reference import SeisBenchPhaseNetReference
from earthquake.picking import pick_from_prob
from earthquake.utils import ensure_dir


def _summarize(pred, true, sr, windows=(0.1, 0.2, 0.5)) -> dict:
    m = match_picks(np.asarray(pred), np.asarray(true), np.asarray(sr), windows_s=windows)
    a = audit_pick_errors(np.asarray(pred), np.asarray(true), np.asarray(sr), window_s=0.5)
    n_lab = max(int(a["n_labeled"]), 1)
    out = {}
    for w in windows:
        out[f"precision@{w}"] = m[f"precision@{w}s"]
        out[f"recall@{w}"] = m[f"recall@{w}s"]
        out[f"f1@{w}"] = m[f"f1@{w}s"]
    out.update(
        {
            "e2e_mae": m["mae"],
            "e2e_median_ae": m["median_ae"],
            "e2e_p90": float(np.nanpercentile(np.abs((np.asarray(pred) - np.asarray(true)) / np.asarray(sr))[np.isfinite(true) & np.isfinite(pred)], 90)) if np.isfinite(true).any() else float("nan"),
            "e2e_p95": m["p95_ae"],
            "matched_timing_mae": a["matched_timing_mae"],
            "matched_timing_median_ae": a["matched_timing_median_ae"],
            "matched_timing_p95": a["matched_timing_p95_ae"],
            "wrong_peak_rate": float(a["n_wrong_peak_beyond_tol"] / n_lab),
            "miss_rate": float(a["n_missed_pick"] / n_lab),
            "no_pick_rate": float((~np.isfinite(pred) & np.isfinite(true)).sum() / n_lab),
            "n_labeled": int(a["n_labeled"]),
            "n_pred": int(a["n_predicted"]),
        }
    )
    return out


def ensure_phasenet_cache(meta: pd.DataFrame, *, weight: str, cfg: dict, device: str, cache_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_dir = ensure_dir(cache_dir)
    cache_path = cache_dir / f"phasenet_{weight}_cache.parquet"
    cands_path = cache_dir / f"phasenet_{weight}_candidates.parquet"
    done = set()
    pick_rows, cand_rows = [], []
    if cache_path.exists():
        prev = pd.read_parquet(cache_path)
        done = set(prev.trace_name.astype(str))
        pick_rows = prev.to_dict(orient="records")
        if cands_path.exists():
            cand_rows = pd.read_parquet(cands_path).to_dict(orient="records")
    need = meta[~meta.trace_name.astype(str).isin(done)]
    thr = float(cfg["pick_threshold"])
    if len(need):
        ref = SeisBenchPhaseNetReference(weight=weight, device=device)
        h5 = resolve_instance_root() / "events" / "Instance_events_counts.hdf5"
        pending_p, pending_c = [], []
        t0 = time.time()
        with InstanceHDF5Reader(h5) as reader:
            for i, (_, row) in enumerate(tqdm(need.iterrows(), total=len(need), desc=f"phasenet:{weight}")):
                wave = reader.read_waveform(str(row.trace_name))
                pred = ref.predict_row(wave, row, remap_to_waveform=True)
                p_pick = pick_from_prob(pred["p"], threshold=thr)
                s_pick = pick_from_prob(pred["s"], threshold=thr)
                p_cands = extract_candidates(
                    pred["p"], phase="p", k=int(cfg["candidate_k"]),
                    min_distance=int(cfg["min_peak_distance"]),
                    min_prominence=float(cfg["min_peak_prominence"]),
                    min_probability=float(cfg["min_peak_probability"]),
                    sampling_rate=float(row.sampling_rate_hz), waveform_starttime=row.trace_start_time,
                )
                s_cands = extract_candidates(
                    pred["s"], phase="s", k=int(cfg["candidate_k"]),
                    min_distance=int(cfg["min_peak_distance"]),
                    min_prominence=float(cfg["min_peak_prominence"]),
                    min_probability=float(cfg["min_peak_probability"]),
                    sampling_rate=float(row.sampling_rate_hz), waveform_starttime=row.trace_start_time,
                )
                pending_p.append(
                    {
                        "trace_name": str(row.trace_name),
                        "event_id": str(row.event_id),
                        "sampling_rate_hz": float(row.sampling_rate_hz),
                        "true_p_sample": float(row.p_arrival_sample) if pd.notna(row.p_arrival_sample) else np.nan,
                        "true_s_sample": float(row.s_arrival_sample) if pd.notna(row.s_arrival_sample) else np.nan,
                        "pred_p_sample": p_pick["peak_sample"],
                        "pred_s_sample": s_pick["peak_sample"],
                        "p_peak_probability": p_pick["peak_probability"],
                        "s_peak_probability": s_pick["peak_probability"],
                        "p_n_cands": len(p_cands),
                        "s_n_cands": len(s_cands),
                    }
                )
                for c in p_cands + s_cands:
                    pending_c.append(
                        {
                            "trace_name": str(row.trace_name),
                            "event_id": str(row.event_id),
                            "phase": c.phase,
                            "sample_index": c.sample_index,
                            "peak_probability": c.peak_probability,
                            "prominence": c.prominence,
                            "peak_width": c.peak_width,
                            "local_entropy": c.local_entropy,
                            "rank": c.rank,
                            "fallback_peak": c.fallback_peak,
                        }
                    )
                if (i + 1) % int(cfg.get("flush_every", 50)) == 0 or (i + 1) == len(need):
                    pick_rows.extend(pending_p)
                    cand_rows.extend(pending_c)
                    pd.DataFrame(pick_rows).drop_duplicates("trace_name", keep="last").to_parquet(cache_path, index=False)
                    pd.DataFrame(cand_rows).drop_duplicates(["trace_name", "phase", "rank"], keep="last").to_parquet(cands_path, index=False)
                    pending_p, pending_c = [], []
        (cache_dir / f"phasenet_{weight}_timing.json").write_text(
            json.dumps({"n": len(need), "seconds": time.time() - t0, "sec_per_trace": (time.time() - t0) / max(len(need), 1)}, indent=2)
        )
    picks = pd.read_parquet(cache_path)
    cands = pd.read_parquet(cands_path)
    tgt = set(meta.trace_name.astype(str))
    picks = picks[picks.trace_name.astype(str).isin(tgt)].drop_duplicates("trace_name", keep="last")
    cands = cands[cands.trace_name.astype(str).isin(tgt)].drop_duplicates(["trace_name", "phase", "rank"], keep="last")
    return picks, cands


def _tau_to_sample(row, tau_s):
    if not np.isfinite(tau_s):
        return float("nan")
    origin = pd.Timestamp(row["origin_time"])
    start = pd.Timestamp(row["trace_start_time"])
    sr = float(row["sampling_rate_hz"])
    return float(((origin + pd.to_timedelta(tau_s, unit="s")) - start).total_seconds() * sr)


def predict_modes(meta: pd.DataFrame, cand_index, cfg, global_res, lambdas) -> pd.DataFrame:
    lw_p, lh_p, lp_p = lambdas["p"]
    lw_s, lh_s, lp_s = lambdas["s"]
    shrink = float(cfg["shrinkage_k"])
    rows = []
    t_rescore = []
    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="rescore-modes"):
        sr = float(row.sampling_rate_hz)
        p_c = cand_index.get((str(row.trace_name), "p"), [])
        s_c = cand_index.get((str(row.trace_name), "s"), [])
        base = {"base_tau_p": float(row.get("base_tau_p", np.nan)), "base_tau_s": float(row.get("base_tau_s", np.nan)), "base_delta_sp": float(row.get("base_delta_sp", np.nan))}
        path = residual_stats_from_row(row)
        pn_p = float(row.pred_p_sample)
        pn_s = float(pick_argmax_probability(s_c).sample_index) if s_c else float(row.pred_s_sample)

        # distance only
        dist_p = _tau_to_sample(row, base["base_tau_p"])
        dist_s = _tau_to_sample(row, base["base_tau_s"])
        # raw path median
        raw_p = _tau_to_sample(row, float(row.get("tau_p_median", np.nan)))
        raw_s = _tau_to_sample(row, float(row.get("tau_s_median", np.nan)))

        pred0 = predict_with_residual_prior(base, path, global_res, shrinkage_k=0.0, min_history=int(cfg["min_history"]), mad_disable_s=float(cfg["mad_disable_s"]), min_sigma_s=float(cfg["min_sigma_s"]), max_sigma_s=float(cfg["max_sigma_s"]))
        predk = predict_with_residual_prior(base, path, global_res, shrinkage_k=shrink, min_history=int(cfg["min_history"]), mad_disable_s=float(cfg["mad_disable_s"]), min_sigma_s=float(cfg["min_sigma_s"]), max_sigma_s=float(cfg["max_sigma_s"]))
        un_p, un_s = expected_samples_catalog(row, pred0)
        sh_p, sh_s = expected_samples_catalog(row, predk)

        hist_ok = bool(predk.history_available or predk.used_path_residual) and np.isfinite(sh_s)

        def rescore_s(expected, sigma, hist, lw, lh, lp):
            t1 = time.perf_counter()
            best, _ = rescore_phase_candidates(s_c, expected_sample=expected, sigma_samples=sigma, lambda_wave=lw, lambda_history=lh, lambda_prominence=lp, history_available=hist)
            t_rescore.append(time.perf_counter() - t1)
            return float(best.sample_index) if best else pn_s

        def rescore_p(expected, sigma, hist, lw, lh, lp):
            best, _ = rescore_phase_candidates(p_c, expected_sample=expected, sigma_samples=sigma, lambda_wave=lw, lambda_history=lh, lambda_prominence=lp, history_available=hist)
            return float(best.sample_index) if best else pn_p

        catalog_p = rescore_p(sh_p, predk.sigma_p_s * sr, hist_ok, lw_p, lh_p, lp_p)
        catalog_s = rescore_s(sh_s, predk.sigma_s_s * sr, hist_ok, lw_s, lh_s, lp_s)
        # identity / no history
        id_p = rescore_p(np.nan, 1.0, False, lw_p, lh_p, lp_p)
        id_s = rescore_s(np.nan, 1.0, False, lw_s, lh_s, lp_s)

        rows.append(
            {
                "trace_name": str(row.trace_name),
                "event_id": str(row.event_id),
                "sampling_rate_hz": sr,
                "true_p_sample": float(row.true_p_sample) if pd.notna(row.get("true_p_sample", np.nan)) else (float(row.p_arrival_sample) if pd.notna(row.p_arrival_sample) else np.nan),
                "true_s_sample": float(row.true_s_sample) if pd.notna(row.get("true_s_sample", np.nan)) else (float(row.s_arrival_sample) if pd.notna(row.s_arrival_sample) else np.nan),
                "pred_p_phasenet": pn_p,
                "pred_s_phasenet": pn_s,
                "pred_p_distance_only": dist_p,
                "pred_s_distance_only": dist_s,
                "pred_p_raw_path": raw_p,
                "pred_s_raw_path": raw_s,
                "pred_p_unshrunk": un_p,
                "pred_s_unshrunk": un_s,
                "pred_p_shrunk": sh_p,
                "pred_s_shrunk": sh_s,
                "pred_p_catalog_rescore": catalog_p,
                "pred_s_catalog_rescore": catalog_s,
                "pred_p_history_unavailable": id_p,
                "pred_s_history_unavailable": id_s,
                "expected_s_sample": float(sh_s) if np.isfinite(sh_s) else np.nan,
                "history_sigma_samples": float(predk.sigma_s_s * sr),
                "history_available": bool(hist_ok),
                "history_count": int(row.get("history_count", 0) or 0),
                "residual_s_mad": float(row.get("residual_s_mad", np.nan)),
                "fallback_level": int(row.get("fallback_level", -1)) if pd.notna(row.get("fallback_level", np.nan)) else -1,
                "s_peak_probability": float(row.get("s_peak_probability", np.nan)),
                "s_n_cands": int(row.get("s_n_cands", len(s_c))),
                "distance_km": float(row.get("distance_km", np.nan)),
                "source_depth_km": float(row.get("source_depth_km", np.nan)),
                "snr_db": float(row.get("snr_db", np.nan)) if pd.notna(row.get("snr_db", np.nan)) else np.nan,
                "station_id": str(row.get("station_id", "")),
            }
        )
    out = pd.DataFrame(rows)
    out.attrs["mean_rescore_seconds"] = float(np.mean(t_rescore)) if t_rescore else float("nan")
    return out


def add_negative_controls(df: pd.DataFrame, cand_index, cfg, lambdas) -> pd.DataFrame:
    rng = np.random.default_rng(int(cfg["seed"]))
    lw_s, lh_s, lp_s = lambdas["s"]
    ok = df[df.history_available].copy()
    shuffle_map = {}
    if len(ok):
        perm = rng.permutation(len(ok))
        src = ok.iloc[perm]
        for i, (_, row) in enumerate(ok.iterrows()):
            shuffle_map[str(row.trace_name)] = src.iloc[i]
    shuf_s, bias_s = [], []
    for _, row in df.iterrows():
        s_c = cand_index.get((str(row.trace_name), "s"), [])
        sr = float(row.sampling_rate_hz)
        if str(row.trace_name) in shuffle_map and row.history_available:
            other = shuffle_map[str(row.trace_name)]
            exp = float(other.expected_s_sample)
            sig = float(other.history_sigma_samples)
            best, _ = rescore_phase_candidates(s_c, expected_sample=exp, sigma_samples=sig, lambda_wave=lw_s, lambda_history=lh_s, lambda_prominence=lp_s, history_available=True)
            shuf_s.append(float(best.sample_index) if best else row.pred_s_phasenet)
        else:
            shuf_s.append(float(row.pred_s_catalog_rescore))
        # biased +5s on expected
        if row.history_available and np.isfinite(row.expected_s_sample):
            exp = float(row.expected_s_sample) + float(cfg["bias_seconds"]) * sr
            best, _ = rescore_phase_candidates(s_c, expected_sample=exp, sigma_samples=float(row.history_sigma_samples), lambda_wave=lw_s, lambda_history=lh_s, lambda_prominence=lp_s, history_available=True)
            bias_s.append(float(best.sample_index) if best else row.pred_s_phasenet)
        else:
            bias_s.append(float(row.pred_s_catalog_rescore))
    df = df.copy()
    df["pred_s_shuffled_history"] = shuf_s
    df["pred_s_biased_history"] = bias_s
    df["pred_p_shuffled_history"] = df["pred_p_catalog_rescore"]
    df["pred_p_biased_history"] = df["pred_p_catalog_rescore"]
    return df


def add_oracles(df: pd.DataFrame, cand_index) -> pd.DataFrame:
    ora_c, ora_sel = [], []
    for _, row in df.iterrows():
        s_c = cand_index.get((str(row.trace_name), "s"), [])
        true = float(row.true_s_sample)
        sr = float(row.sampling_rate_hz)
        if s_c and np.isfinite(true):
            best = min(s_c, key=lambda c: abs(c.sample_index - true))
            ora_c.append(float(best.sample_index))
        else:
            ora_c.append(float("nan"))
        e_pn = abs(row.pred_s_phasenet - true) / sr if np.isfinite(true) else np.inf
        e_fr = abs(row.pred_s_catalog_rescore - true) / sr if np.isfinite(true) else np.inf
        ora_sel.append(float(row.pred_s_phasenet if e_pn <= e_fr else row.pred_s_catalog_rescore))
    df = df.copy()
    df["pred_s_oracle_candidate"] = ora_c
    df["pred_s_oracle_selector"] = ora_sel
    df["pred_p_oracle_candidate"] = df["pred_p_phasenet"]
    df["pred_p_oracle_selector"] = df["pred_p_phasenet"]
    return df


@torch.no_grad()
def add_learned_gate(df: pd.DataFrame, cand_index, meta: pd.DataFrame, cfg, device) -> pd.DataFrame:
    scaler_path = artifacts_dir() / "models" / "learned_gate" / "feature_scaler.pkl"
    if not scaler_path.exists():
        return df
    scaler = load_scaler(scaler_path)
    seeds = list(cfg.get("gate_seeds", [42, 123, 2026]))
    meta_idx = meta.set_index("trace_name")
    for seed in seeds:
        ckpt_path = artifacts_dir() / "models" / "learned_gate" / f"scalar_gate_seed{seed}" / "best.pt"
        if not ckpt_path.exists():
            continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = ScalarGate(int(ckpt["n_features"]), hidden_dim=int(ckpt.get("cfg", {}).get("hidden_dim", 64)), dropout=0.0, init_gate=0.8).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        picks, gates = [], []
        for _, row in df.iterrows():
            s_c = cand_index.get((str(row.trace_name), "s"), [])
            if not row.history_available or not s_c:
                picks.append(float(row.pred_s_phasenet))
                gates.append(1.0)
                continue
            mrow = meta_idx.loc[str(row.trace_name)] if str(row.trace_name) in meta_idx.index else row
            s_dicts = [
                {"sample_index": c.sample_index, "peak_probability": c.peak_probability, "prominence": c.prominence, "peak_width": c.peak_width, "local_entropy": c.local_entropy, "fallback_peak": c.fallback_peak}
                for c in s_c
            ]
            feats = build_trace_features(
                mrow, s_dicts, expected_s_sample=float(row.expected_s_sample), history_sigma_samples=float(row.history_sigma_samples), sampling_rate=float(row.sampling_rate_hz), mode="catalog"
            )
            X, M = scaler.transform(pd.DataFrame([feats]))
            k = min(len(s_c), int(cfg["candidate_k"]))
            prob = torch.zeros(1, k, device=device)
            prom = torch.zeros(1, k, device=device)
            samp = torch.zeros(1, k, device=device)
            mask = torch.zeros(1, k, dtype=torch.bool, device=device)
            for j in range(k):
                prob[0, j] = s_c[j].peak_probability
                prom[0, j] = s_c[j].prominence if np.isfinite(s_c[j].prominence) else s_c[j].peak_probability
                samp[0, j] = s_c[j].sample_index
                mask[0, j] = True
            out = predict_gate_pick(
                model,
                torch.tensor(X, device=device),
                torch.tensor(M, device=device),
                prob, prom, samp, mask,
                torch.tensor([[float(row.expected_s_sample)]], device=device),
                torch.tensor([[float(row.history_sigma_samples)]], device=device),
                torch.tensor([[True]], device=device),
            )
            picks.append(float(out["pick_sample"][0].item()))
            gates.append(float(out["gate"][0].item()))
        df[f"pred_s_learned_gate_seed{seed}"] = picks
        df[f"gate_seed{seed}"] = gates
        df[f"pred_p_learned_gate_seed{seed}"] = df["pred_p_phasenet"]
    # mean gate pick across seeds (for main ablation table use seed42 + report all)
    if any(f"pred_s_learned_gate_seed{s}" in df.columns for s in seeds):
        cols = [f"pred_s_learned_gate_seed{s}" for s in seeds if f"pred_s_learned_gate_seed{s}" in df.columns]
        # use seed 42 as primary ablation column
        if "pred_s_learned_gate_seed42" in df.columns:
            df["pred_s_learned_gate"] = df["pred_s_learned_gate_seed42"]
            df["pred_p_learned_gate"] = df["pred_p_phasenet"]
            df["gate"] = df["gate_seed42"]
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage4/confirmatory.yaml")
    parser.add_argument("--weight", default=None, help="override single PhaseNet weight")
    args = parser.parse_args()
    cfg = load_yaml_config(ROOT / args.config)
    out = ensure_dir(artifacts_dir() / "results" / "stage4")
    tables = ensure_dir(out / "tables")
    logs = ensure_dir(out / "logs")
    lock = out / "method_lock.json"
    if not lock.exists():
        raise SystemExit("method_lock.json missing — run scripts/lock_stage4_method.py first")
    hold = pd.read_parquet(out / "holdout" / "holdout_events.parquet")
    if len(hold) == 0:
        raise SystemExit("Empty holdout")

    resid = pd.read_parquet(artifacts_dir() / "history" / "residual_history_features_frozen.parquet")
    meta = hold.merge(resid, on=["trace_name", "event_id"], how="left", suffixes=("", "_r"))
    weight = args.weight or cfg["phasenet_weight"]
    device = cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    cache_dir = ensure_dir(out / "cache")
    picks, cands = ensure_phasenet_cache(meta, weight=weight, cfg=cfg, device=device, cache_dir=cache_dir)
    meta = meta.merge(picks, on=["trace_name", "event_id"], how="inner", suffixes=("", "_pk"))
    for c in ["true_p_sample", "true_s_sample"]:
        if c in meta.columns and f"{c}" in meta.columns:
            pass
    if "p_arrival_sample" in meta.columns:
        meta["true_p_sample"] = meta["p_arrival_sample"]
    if "s_arrival_sample" in meta.columns:
        meta["true_s_sample"] = meta["s_arrival_sample"]

    cand_index = build_cand_index(cands)
    rh = load_json(artifacts_dir() / "results" / "stage2" / "residual_history_meta.json")
    global_res = rh["global_residual"]
    lambdas = {
        "p": (float(cfg["lambda_p"]["lw"]), float(cfg["lambda_p"]["lh"]), float(cfg["lambda_p"]["lp"])),
        "s": (float(cfg["lambda_s"]["lw"]), float(cfg["lambda_s"]["lh"]), float(cfg["lambda_s"]["lp"])),
    }

    df = predict_modes(meta, cand_index, cfg, global_res, lambdas)
    df = add_negative_controls(df, cand_index, cfg, lambdas)
    df = add_oracles(df, cand_index)
    if weight == "stead":
        df = add_learned_gate(df, cand_index, meta, cfg, device)

    # metrics
    methods = {
        "phasenet": ("pred_p_phasenet", "pred_s_phasenet"),
        "distance_only": ("pred_p_distance_only", "pred_s_distance_only"),
        "raw_path_history": ("pred_p_raw_path", "pred_s_raw_path"),
        "mlp_residual_unshrunk": ("pred_p_unshrunk", "pred_s_unshrunk"),
        "mlp_residual_shrink50": ("pred_p_shrunk", "pred_s_shrunk"),
        "fixed_catalog_rescore": ("pred_p_catalog_rescore", "pred_s_catalog_rescore"),
        "shuffled_history": ("pred_p_shuffled_history", "pred_s_shuffled_history"),
        "biased_history": ("pred_p_biased_history", "pred_s_biased_history"),
        "history_unavailable": ("pred_p_history_unavailable", "pred_s_history_unavailable"),
        "oracle_selector": ("pred_p_oracle_selector", "pred_s_oracle_selector"),
        "oracle_candidate": ("pred_p_oracle_candidate", "pred_s_oracle_candidate"),
    }
    if "pred_s_learned_gate" in df.columns:
        methods["learned_gate"] = ("pred_p_learned_gate", "pred_s_learned_gate")
        for seed in cfg.get("gate_seeds", []):
            if f"pred_s_learned_gate_seed{seed}" in df.columns:
                methods[f"learned_gate_seed{seed}"] = (f"pred_p_learned_gate_seed{seed}", f"pred_s_learned_gate_seed{seed}")

    results = []
    for name, (pc, sc) in methods.items():
        sp = _summarize(df[pc], df["true_p_sample"], df["sampling_rate_hz"])
        ss = _summarize(df[sc], df["true_s_sample"], df["sampling_rate_hz"])
        results.append({"method": name, "weight": weight, "phase": "P", **{f"P_{k}": v for k, v in sp.items()}})
        results.append({"method": name, "weight": weight, "phase": "S", **{f"S_{k}": v for k, v in ss.items()}})
        print(name, {k: ss[k] for k in ["f1@0.1", "f1@0.5", "e2e_p95", "wrong_peak_rate"]}, flush=True)

    # flatten for main table
    flat = []
    for name, (pc, sc) in methods.items():
        sp = _summarize(df[pc], df["true_p_sample"], df["sampling_rate_hz"])
        ss = _summarize(df[sc], df["true_s_sample"], df["sampling_rate_hz"])
        flat.append({"method": name, "weight": weight, **{f"P_{k}": sp[k] for k in sp}, **{f"S_{k}": ss[k] for k in ss}})

    tag = f"_{weight}" if args.weight else ""
    df.to_parquet(out / f"holdout_picks{tag}.parquet", index=False)
    pd.DataFrame(flat).to_csv(tables / f"main_results{tag}.csv", index=False)
    if not args.weight:
        pd.DataFrame(flat).to_csv(tables / "main_results.csv", index=False)
        # ablation subset
        abl = [r for r in flat if r["method"] in {
            "phasenet", "distance_only", "raw_path_history", "mlp_residual_unshrunk", "mlp_residual_shrink50",
            "fixed_catalog_rescore", "learned_gate", "shuffled_history", "biased_history", "history_unavailable",
            "oracle_selector", "oracle_candidate",
        }]
        pd.DataFrame(abl).to_csv(tables / "ablation_results.csv", index=False)
    save_json({"weight": weight, "n": len(df), "methods": list(methods)}, out / f"eval_meta{tag}.json")
    print({"saved_picks": str(out / f"holdout_picks{tag}.parquet"), "n": len(df)}, flush=True)


if __name__ == "__main__":
    main()
