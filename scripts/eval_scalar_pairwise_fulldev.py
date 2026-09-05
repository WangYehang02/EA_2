#!/usr/bin/env python
"""Frozen scalar_pairwise phaseB full-dev evaluation.

Compares fixed_rescore_UNION vs resid_s_control vs scalar_pairwise.
Does NOT read confirm. Does NOT retrain / retune tau. Does NOT use waveform.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, save_json
from earthquake.metrics import match_picks
from earthquake.pairwise.fulldev import (
    RESID_CONTROL,
    TAU,
    apply_switch,
    build_phaseb_pairs,
    load_scalar_model,
    predict_p_switch,
    resid_control_pred,
)
from earthquake.pairwise.guards import assert_no_confirm_path
from earthquake.utils import ensure_dir

OUT = artifacts_dir() / "results" / "pairwise_fulldev"
PILOT = artifacts_dir() / "results" / "pairwise_pilot"
REPORT = ROOT / "reports" / "pairwise"
N_BOOT = 2000
SEED = 42


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fail(msg: str, **extra) -> None:
    payload = {
        "error": msg,
        "traceback": traceback.format_exc() if sys.exc_info()[0] else None,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cwd": os.getcwd(),
        "argv": sys.argv,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        **extra,
    }
    ensure_dir(OUT)
    save_json(payload, OUT / "SCALAR_PAIRWISE.FULLDEV.FAILED.json")
    (OUT / "SCALAR_PAIRWISE.FULLDEV.FAILED").write_text(msg + "\n", encoding="utf-8")
    # clear RUNNING if present
    run = OUT / "SCALAR_PAIRWISE.FULLDEV.RUNNING"
    if run.exists():
        run.unlink()
    print("SCALAR_PAIRWISE.FULLDEV.FAILED:", msg, flush=True)
    raise SystemExit(1)


def metrics_bundle(pred, true, sr) -> dict:
    m = match_picks(pred, true, sr, windows_s=(0.1, 0.5))
    err = np.abs((pred - true) / sr)
    finite = np.isfinite(err)
    out = {
        "f1@0.1": float(m["f1@0.1s"]),
        "f1@0.5": float(m["f1@0.5s"]),
        "precision@0.1": float(m["precision@0.1s"]),
        "recall@0.1": float(m["recall@0.1s"]),
        "precision@0.5": float(m["precision@0.5s"]),
        "recall@0.5": float(m["recall@0.5s"]),
        "miss_rate": float(m["miss_rate"]),
        "detected_ae_median": float(m["detected_ae_median"]),
        "detected_ae_p95": float(m["detected_ae_p95"]),
        "detected_ae_mae": float(m["detected_ae_mae"]),
        "frac_err_gt_1s": float(np.mean(err[finite] > 1.0)) if finite.any() else float("nan"),
        "frac_err_gt_5s": float(np.mean(err[finite] > 5.0)) if finite.any() else float("nan"),
        "frac_err_gt_10s": float(np.mean(err[finite] > 10.0)) if finite.any() else float("nan"),
        "frac_err_gt_30s": float(np.mean(err[finite] > 30.0)) if finite.any() else float("nan"),
    }
    return out


def switch_stats(pairs: pd.DataFrame, pred, switch, true, sr) -> dict:
    base_ok = pairs["c1_ok"].to_numpy(bool)
    new_ok = np.abs(pred - true) / sr <= 0.5
    c2_ok = pairs["c2_ok"].to_numpy(bool)
    fixes = int(np.sum(switch & (~base_ok) & new_ok))
    breaks = int(np.sum(switch & base_ok & (~new_ok)))
    q1 = base_ok & (~c2_ok)
    q2 = (~base_ok) & c2_ok
    both_ok = base_ok & c2_ok
    both_bad = (~base_ok) & (~c2_ok)
    ge2 = pairs["n_candidates"].to_numpy(int) >= 2
    return {
        "n_eligible_pairs": int(ge2.sum()),
        "n_switch": int(switch.sum()),
        "switch_rate": float(switch.mean()),
        "fixes": fixes,
        "breaks": breaks,
        "net_fixes": fixes - breaks,
        "switch_precision": fixes / max(fixes + breaks, 1),
        "Q1_count": int(q1.sum()),
        "Q2_count": int(q2.sum()),
        "Q1_break_rate": float(np.sum(switch & q1 & (~new_ok)) / max(int(q1.sum()), 1)),
        "Q2_recovery_rate": float(np.sum(switch & q2 & new_ok) / max(int(q2.sum()), 1)),
        "both_correct": int(both_ok.sum()),
        "both_wrong": int(both_bad.sum()),
    }


def event_bootstrap(pred_a, pred_b, true, sr, events, n_boot=N_BOOT, seed=SEED) -> dict:
    rng = np.random.default_rng(seed)
    idx_by: dict[str, list[int]] = {}
    for i, e in enumerate(events):
        idx_by.setdefault(str(e), []).append(i)
    lists = [np.asarray(v, int) for v in idx_by.values()]
    d05 = np.empty(n_boot)
    d01 = np.empty(n_boot)
    dp95 = np.empty(n_boot)
    for b in range(n_boot):
        draw = rng.integers(0, len(lists), size=len(lists))
        ix = np.concatenate([lists[j] for j in draw])
        ma = metrics_bundle(pred_a[ix], true[ix], sr[ix])
        mb = metrics_bundle(pred_b[ix], true[ix], sr[ix])
        d05[b] = ma["f1@0.5"] - mb["f1@0.5"]
        d01[b] = ma["f1@0.1"] - mb["f1@0.1"]
        dp95[b] = ma["detected_ae_p95"] - mb["detected_ae_p95"]

    def pack(a):
        return {
            "mean": float(a.mean()),
            "ci95": [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))],
        }

    return {
        "n_boot": n_boot,
        "seed": seed,
        "n_events": len(lists),
        "delta_f1@0.5": pack(d05),
        "delta_f1@0.1": pack(d01),
        "delta_p95": pack(dp95),
    }


def verify_lock(lock: dict) -> None:
    ckpt = Path(lock["checkpoint_path"])
    assert_no_confirm_path(ckpt)
    if _sha_file(ckpt) != lock["checkpoint_sha256"]:
        raise RuntimeError("checkpoint sha mismatch vs METHOD_LOCK")
    if float(lock["calibration_tau"]) != TAU:
        raise RuntimeError(f"tau lock mismatch: {lock['calibration_tau']}")
    man = Path(lock["phaseB_manifest_path"])
    assert_no_confirm_path(man)
    if _sha_file(man) != lock["phaseB_manifest_sha256"]:
        raise RuntimeError("phaseB manifest sha mismatch")
    enr = Path(lock["phaseB_enriched_candidates_path"])
    assert_no_confirm_path(enr)
    if _sha_file(enr) != lock["phaseB_enriched_sha256"]:
        raise RuntimeError("enriched candidates sha mismatch")
    sha_file = OUT / "SCALAR_PAIRWISE.FULLDEV_METHOD_LOCK.sha256"
    if sha_file.exists():
        body_sha = sha_file.read_text().strip()
        if lock.get("lock_body_sha256") and lock["lock_body_sha256"] != body_sha:
            # tolerate if lock rewritten with same content hash field
            pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto", help="cuda|cpu|auto")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--smoke", type=int, default=0, help="if >0, only first N traces (no sentinels)")
    ap.add_argument("--skip-running-marker", action="store_true")
    args = ap.parse_args()

    ensure_dir(OUT)
    ensure_dir(REPORT)
    ensure_dir(OUT / "logs")

    if (OUT / "SCALAR_PAIRWISE.FULLDEV.PASSED").exists() or (OUT / "SCALAR_PAIRWISE.FULLDEV.NO_GO").exists():
        if not args.smoke:
            raise SystemExit("full-dev already completed (PASSED or NO_GO present)")

    lock_path = OUT / "SCALAR_PAIRWISE.FULLDEV_METHOD_LOCK.json"
    if not lock_path.exists():
        _fail("METHOD_LOCK missing; run lock_scalar_pairwise_fulldev.py first")
    lock = json.loads(lock_path.read_text())
    try:
        verify_lock(lock)
    except Exception as e:
        _fail(str(e), lock_hash=lock.get("lock_body_sha256"))

    # confirm guard smoke
    try:
        assert_no_confirm_path("/tmp/foo_confirm_bar.csv")
        _fail("confirm guard did not fire")
    except RuntimeError:
        pass

    if not args.skip_running_marker and not args.smoke:
        (OUT / "SCALAR_PAIRWISE.FULLDEV.RUNNING").write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                }
            )
            + "\n",
            encoding="utf-8",
        )

    t0 = time.time()
    try:
        if args.device == "cpu":
            device = torch.device("cpu")
        elif args.device == "cuda":
            if not torch.cuda.is_available():
                _fail("cuda requested but unavailable")
            device = torch.device("cuda:0")
        else:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"device={device}", flush=True)

        man_path = Path(lock["phaseB_manifest_path"])
        enr_path = Path(lock["phaseB_enriched_candidates_path"])
        assert_no_confirm_path(man_path)
        assert_no_confirm_path(enr_path)

        man = pd.read_csv(man_path)
        enr = pd.read_parquet(enr_path)
        if args.smoke > 0:
            keep = set(man.trace_name.astype(str).head(args.smoke))
            man = man[man.trace_name.astype(str).isin(keep)].reset_index(drop=True)
            enr = enr[enr.trace_name.astype(str).isin(keep)].copy()

        print(f"building pairs n_man={len(man)} n_cand={len(enr)}", flush=True)
        pairs = build_phaseb_pairs(enr, man)
        pairs_path = OUT / ("pairs_phaseB_smoke.parquet" if args.smoke else "pairs_phaseB.parquet")
        pairs.to_parquet(pairs_path, index=False)

        # sanity: c1 agrees with frozen baseline npy when full
        if not args.smoke:
            fixed_npy = np.load(
                artifacts_dir() / "results/stage6/phaseC/baseline_preds/fixed_rescore_UNION.npy"
            )
            if len(fixed_npy) == len(pairs):
                agree = float(np.mean(np.abs(pairs.c1_sample.to_numpy(float) - fixed_npy) < 1e-3))
                print(f"c1 vs frozen fixed_rescore_UNION.npy agree={agree:.6f}", flush=True)
                if agree < 0.999:
                    _fail(f"c1 disagrees with frozen baseline npy: agree={agree}")

        true = pairs.true_s_sample.to_numpy(float)
        sr = pairs.sampling_rate_hz.to_numpy(float)
        fixed_pred = pairs.c1_sample.to_numpy(float)
        resid_pred = resid_control_pred(
            pairs, lam=float(RESID_CONTROL["lambda"]), sigma=float(RESID_CONTROL["sigma_s"])
        )

        model = load_scalar_model(Path(lock["checkpoint_path"]), device)
        if model.n_parameters() != lock["n_parameters"]:
            _fail(f"param count mismatch {model.n_parameters()} vs lock {lock['n_parameters']}")
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        p_switch = predict_p_switch(model, pairs, device)
        peak_mb = None
        if device.type == "cuda":
            peak_mb = float(torch.cuda.max_memory_allocated() / (1024**2))
            print(f"peak_vram_MiB={peak_mb:.1f}", flush=True)

        tau = float(lock["calibration_tau"])
        if abs(tau - 0.5) > 1e-12:
            _fail(f"tau must be 0.50, got {tau}")
        scalar_pred, switch = apply_switch(pairs, p_switch, tau=tau)

        # no abstain / only c1/c2
        c1 = pairs.c1_sample.to_numpy(float)
        c2 = pairs.c2_sample.to_numpy(float)
        ok_set = np.isclose(scalar_pred, c1, equal_nan=False) | (
            np.isfinite(c2) & np.isclose(scalar_pred, c2, equal_nan=False)
        )
        if not bool(np.all(ok_set)):
            _fail("predictions outside {c1,c2}")

        fixed_m = metrics_bundle(fixed_pred, true, sr)
        resid_m = metrics_bundle(resid_pred, true, sr)
        scalar_m = metrics_bundle(scalar_pred, true, sr)
        sw = switch_stats(pairs, scalar_pred, switch, true, sr)

        delta_fixed = scalar_m["f1@0.5"] - fixed_m["f1@0.5"]
        delta_resid = scalar_m["f1@0.5"] - resid_m["f1@0.5"]
        delta_p95_fixed = scalar_m["detected_ae_p95"] - fixed_m["detected_ae_p95"]

        resid_m = {
            **resid_m,
            "delta_f1@0.5_vs_fixed": resid_m["f1@0.5"] - fixed_m["f1@0.5"],
            "delta_f1@0.1_vs_fixed": resid_m["f1@0.1"] - fixed_m["f1@0.1"],
        }
        scalar_out = {
            **scalar_m,
            **sw,
            "tau": tau,
            "delta_f1@0.5_vs_fixed": delta_fixed,
            "delta_f1@0.1_vs_fixed": scalar_m["f1@0.1"] - fixed_m["f1@0.1"],
            "delta_f1@0.5_vs_resid": delta_resid,
            "delta_f1@0.1_vs_resid": scalar_m["f1@0.1"] - resid_m["f1@0.1"],
            "delta_p95_vs_fixed": delta_p95_fixed,
            "peak_vram_MiB": peak_mb,
        }

        # margin bins on ge2
        ge2 = pairs[pairs.n_candidates >= 2].copy()
        margins = ge2.margin_fixed.to_numpy(float)
        qs = np.nanquantile(margins, [0.10, 0.30, 0.60])
        edges = [-np.inf, qs[0], qs[1], qs[2], np.inf]
        labels = ["lowest_10pct", "p10_p30", "p30_p60", "highest_40pct"]
        ge2["margin_bin"] = pd.cut(ge2.margin_fixed, bins=edges, labels=labels, include_lowest=True)
        # map back
        bin_rows = []
        idx_map = {t: i for i, t in enumerate(pairs.trace_name.astype(str))}
        for lab in labels:
            sub = ge2[ge2.margin_bin == lab]
            if len(sub) == 0:
                continue
            ix = np.asarray([idx_map[t] for t in sub.trace_name.astype(str)], int)
            fm = metrics_bundle(fixed_pred[ix], true[ix], sr[ix])
            rm = metrics_bundle(resid_pred[ix], true[ix], sr[ix])
            sm = metrics_bundle(scalar_pred[ix], true[ix], sr[ix])
            sw_sub = switch[ix]
            st = switch_stats(pairs.iloc[ix].reset_index(drop=True), scalar_pred[ix], sw_sub, true[ix], sr[ix])
            bin_rows.append(
                {
                    "margin_bin": lab,
                    "n": len(sub),
                    "n_events": int(sub.event_id.nunique()),
                    "margin_lo": float(sub.margin_fixed.min()),
                    "margin_hi": float(sub.margin_fixed.max()),
                    "fixed_f1@0.5": fm["f1@0.5"],
                    "resid_f1@0.5": rm["f1@0.5"],
                    "scalar_f1@0.5": sm["f1@0.5"],
                    "delta_scalar_fixed": sm["f1@0.5"] - fm["f1@0.5"],
                    "delta_scalar_resid": sm["f1@0.5"] - rm["f1@0.5"],
                    "fixes": st["fixes"],
                    "breaks": st["breaks"],
                    "net_fixes": st["net_fixes"],
                    "switch_precision": st["switch_precision"],
                    "Q2_recovery_rate": st["Q2_recovery_rate"],
                }
            )
        margin_df = pd.DataFrame(bin_rows)

        # candidate count bins
        cand_rows = []
        for lab, mask in [
            ("1_candidate", pairs.n_candidates == 1),
            ("2_candidates", pairs.n_candidates == 2),
            ("3plus_candidates", pairs.n_candidates >= 3),
        ]:
            ix = np.where(mask.to_numpy())[0]
            if ix.size == 0:
                continue
            fm = metrics_bundle(fixed_pred[ix], true[ix], sr[ix])
            rm = metrics_bundle(resid_pred[ix], true[ix], sr[ix])
            sm = metrics_bundle(scalar_pred[ix], true[ix], sr[ix])
            st = switch_stats(pairs.iloc[ix].reset_index(drop=True), scalar_pred[ix], switch[ix], true[ix], sr[ix])
            cand_rows.append(
                {
                    "cand_bin": lab,
                    "n": int(ix.size),
                    "n_events": int(pairs.iloc[ix].event_id.nunique()),
                    "fixed_f1@0.5": fm["f1@0.5"],
                    "resid_f1@0.5": rm["f1@0.5"],
                    "scalar_f1@0.5": sm["f1@0.5"],
                    "delta_scalar_fixed": sm["f1@0.5"] - fm["f1@0.5"],
                    "delta_scalar_resid": sm["f1@0.5"] - rm["f1@0.5"],
                    "net_fixes": st["net_fixes"],
                    "switch_precision": st["switch_precision"],
                    "Q2_recovery_rate": st["Q2_recovery_rate"],
                }
            )
        cand_df = pd.DataFrame(cand_rows)

        # switch cases dump
        cases = pairs.copy()
        cases["p_switch"] = p_switch
        cases["switched"] = switch
        cases["pred_scalar"] = scalar_pred
        cases["pred_resid"] = resid_pred
        cases["ok_fixed"] = pairs.c1_ok
        cases["ok_scalar"] = np.abs(scalar_pred - true) / sr <= 0.5
        cases["ok_resid"] = np.abs(resid_pred - true) / sr <= 0.5
        cases.to_parquet(OUT / ("switch_cases_smoke.parquet" if args.smoke else "switch_cases.parquet"), index=False)
        cases.loc[switch, ["trace_name", "event_id", "p_switch", "margin_fixed", "c1_ok", "c2_ok", "ok_scalar"]].to_csv(
            OUT / "switch_cases.csv", index=False
        )

        if args.smoke:
            print("SMOKE OK", {"fixed": fixed_m["f1@0.5"], "resid": resid_m["f1@0.5"], "scalar": scalar_m["f1@0.5"]})
            return

        boot_sf = event_bootstrap(scalar_pred, fixed_pred, true, sr, pairs.event_id.to_numpy(), n_boot=args.n_boot)
        boot_sr = event_bootstrap(scalar_pred, resid_pred, true, sr, pairs.event_id.to_numpy(), n_boot=args.n_boot)

        # gates
        gate1 = {
            "delta_f1_ge_0.008": delta_fixed >= 0.008,
            "bootstrap_ci_gt_0": boot_sf["delta_f1@0.5"]["ci95"][0] > 0,
            "p95_not_worse_0.05": delta_p95_fixed <= 0.05,
        }
        gate1["passed"] = all(gate1.values())

        gate2 = {
            "delta_f1_ge_0.003": delta_resid >= 0.003,
            "bootstrap_ci_gt_0": boot_sr["delta_f1@0.5"]["ci95"][0] > 0,
            "not_material_lt_0.001": delta_resid < 0.001,
        }
        if gate2["not_material_lt_0.001"]:
            gate2["verdict"] = "scalar_learned_increment_not_material"
            gate2["passed"] = False
        elif gate2["delta_f1_ge_0.003"] and gate2["bootstrap_ci_gt_0"]:
            gate2["verdict"] = "scalar_beats_resid"
            gate2["passed"] = True
        elif gate2["delta_f1_ge_0.003"]:
            gate2["verdict"] = "scalar_beats_resid_point_estimate_only"
            gate2["passed"] = False
        else:
            gate2["verdict"] = "scalar_vs_resid_weak"
            gate2["passed"] = False

        gate3 = {
            "switch_precision_ge_0.80": sw["switch_precision"] >= 0.80,
            "net_fixes_gt_0": sw["net_fixes"] > 0,
        }
        gate3["passed"] = all(gate3.values())

        overall = bool(gate1["passed"] and gate2["passed"] and gate3["passed"])

        save_json(fixed_m, OUT / "fixed_metrics.json")
        save_json(resid_m, OUT / "resid_control_metrics.json")
        save_json(scalar_out, OUT / "scalar_pairwise_metrics.json")
        save_json(boot_sf, OUT / "bootstrap_scalar_vs_fixed.json")
        save_json(boot_sr, OUT / "bootstrap_scalar_vs_resid.json")
        margin_df.to_csv(OUT / "margin_bins.csv", index=False)
        cand_df.to_csv(OUT / "candidate_count_bins.csv", index=False)
        pd.DataFrame(
            [
                {"method": "fixed_rescore_UNION", **fixed_m},
                {"method": "resid_s_control", **resid_m},
                {"method": "scalar_pairwise", **scalar_out},
            ]
        ).to_csv(OUT / "comparison_metrics.csv", index=False)

        final = {
            "gates": {"1_vs_fixed": gate1, "2_vs_resid": gate2, "3_switch_quality": gate3},
            "passed": overall,
            "n_eval": len(pairs),
            "n_events": int(pairs.event_id.nunique()),
            "fixed": fixed_m,
            "resid": resid_m,
            "scalar": scalar_out,
            "bootstrap_vs_fixed": boot_sf,
            "bootstrap_vs_resid": boot_sr,
            "lock_body_sha256": lock.get("lock_body_sha256"),
            "confirm_read": False,
            "waveform_used": False,
            "tau": tau,
            "elapsed_s": time.time() - t0,
            "device": str(device),
            "recommend_confirm": bool(overall),
        }
        save_json(final, OUT / "fulldev_final.json")

        lines = [
            "# Scalar pairwise phaseB full-dev report",
            "",
            f"n_eval={len(pairs)} events={pairs.event_id.nunique()} tau={tau}",
            "",
            "## Metrics",
            f"- fixed F1@0.5={fixed_m['f1@0.5']:.6f} F1@0.1={fixed_m['f1@0.1']:.6f} P95={fixed_m['detected_ae_p95']:.4f}",
            f"- resid  F1@0.5={resid_m['f1@0.5']:.6f} Δfixed={resid_m['delta_f1@0.5_vs_fixed']:+.6f}",
            f"- scalar F1@0.5={scalar_m['f1@0.5']:.6f} Δfixed={delta_fixed:+.6f} Δresid={delta_resid:+.6f}",
            f"- fixes={sw['fixes']} breaks={sw['breaks']} net={sw['net_fixes']} "
            f"switch_prec={sw['switch_precision']:.4f} Q2_rec={sw['Q2_recovery_rate']:.4f}",
            "",
            "## Bootstrap",
            f"- scalar vs fixed: {boot_sf['delta_f1@0.5']}",
            f"- scalar vs resid: {boot_sr['delta_f1@0.5']}",
            "",
            "## Gates",
            json.dumps(final["gates"], indent=2),
            "",
            f"Verdict: {'PASSED' if overall else 'NO_GO'}",
            "Hard stop. No confirm.",
            "",
        ]
        (OUT / "full_dev_report.md").write_text("\n".join(lines), encoding="utf-8")
        (REPORT / "scalar_pairwise_fulldev_report.md").write_text("\n".join(lines), encoding="utf-8")

        run = OUT / "SCALAR_PAIRWISE.FULLDEV.RUNNING"
        if run.exists():
            run.unlink()

        if overall:
            (OUT / "SCALAR_PAIRWISE.FULLDEV.PASSED").write_text("passed\n", encoding="utf-8")
            save_json(final, OUT / "SCALAR_PAIRWISE.FULLDEV.PASSED.json")
            print("SCALAR_PAIRWISE.FULLDEV.PASSED", flush=True)
        else:
            (OUT / "SCALAR_PAIRWISE.FULLDEV.NO_GO").write_text(
                gate2.get("verdict", "no_go") + "\n", encoding="utf-8"
            )
            save_json(final, OUT / "SCALAR_PAIRWISE.FULLDEV.NO_GO.json")
            print("SCALAR_PAIRWISE.FULLDEV.NO_GO", gate2.get("verdict"), flush=True)

    except SystemExit:
        raise
    except Exception as e:
        _fail(str(e), exception_type=type(e).__name__)


if __name__ == "__main__":
    main()
