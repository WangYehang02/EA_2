#!/usr/bin/env python
"""One-shot confirm evaluation for frozen scalar_pairwise.

Evaluation-only. No retrain / retune. Reads frozen Stage-6 confirm artifacts.
Does not modify Stage-6 CONFIRM.CONSUMED / method lock.
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
from earthquake.pairwise.confirm_data import build_confirm_enriched, build_confirm_pairs
from earthquake.pairwise.fulldev import (
    RESID_CONTROL,
    TAU,
    apply_switch,
    load_scalar_model,
    predict_p_switch,
    resid_control_pred,
)
from earthquake.utils import ensure_dir

EXPECTED_METHOD_LOCK_HASH = "1fb09af7ebd96bb0a71f4b4af91313d8ed10410b96f4aeb0616a813d3725a75b"
OUT = artifacts_dir() / "results" / "pairwise_confirm"
FULLDEV = artifacts_dir() / "results" / "pairwise_fulldev"
PILOT = artifacts_dir() / "results" / "pairwise_pilot"
REPORT = ROOT / "reports" / "pairwise"
N_BOOT = 5000
SEED = 20260817


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fail(msg: str, **extra) -> None:
    ensure_dir(OUT)
    payload = {
        "error": msg,
        "traceback": traceback.format_exc() if sys.exc_info()[0] else None,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "argv": sys.argv,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        **extra,
    }
    save_json(payload, OUT / "SCALAR_PAIRWISE.CONFIRM.FAILED.json")
    (OUT / "SCALAR_PAIRWISE.CONFIRM.FAILED").write_text(msg + "\n", encoding="utf-8")
    run = OUT / "SCALAR_PAIRWISE.CONFIRM.RUNNING"
    if run.exists():
        run.unlink()
    print("SCALAR_PAIRWISE.CONFIRM.FAILED:", msg, flush=True)
    raise SystemExit(1)


def metrics_bundle(pred, true, sr) -> dict:
    m = match_picks(pred, true, sr, windows_s=(0.1, 0.5))
    err = np.abs((pred - true) / sr)
    finite = np.isfinite(err)
    return {
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


def switch_stats(pairs: pd.DataFrame, pred, switch, true, sr) -> dict:
    base_ok = pairs["c1_ok"].to_numpy(bool)
    new_ok = np.abs(pred - true) / sr <= 0.5
    c2_ok = pairs["c2_ok"].to_numpy(bool)
    fixes = int(np.sum(switch & (~base_ok) & new_ok))
    breaks = int(np.sum(switch & base_ok & (~new_ok)))
    q1 = base_ok & (~c2_ok)
    q2 = (~base_ok) & c2_ok
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
        "both_correct": int((base_ok & c2_ok).sum()),
        "both_wrong": int(((~base_ok) & (~c2_ok)).sum()),
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
        return {"mean": float(a.mean()), "ci95": [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]}

    return {
        "n_boot": n_boot,
        "seed": seed,
        "n_events": len(lists),
        "delta_f1@0.5": pack(d05),
        "delta_f1@0.1": pack(d01),
        "delta_p95": pack(dp95),
    }


def structural_sanity(ex_lock: dict) -> None:
    if ex_lock["method_lock_hash"] != EXPECTED_METHOD_LOCK_HASH:
        raise RuntimeError("method_lock_hash mismatch")
    if abs(float(ex_lock["tau"]) - 0.50) > 1e-12:
        raise RuntimeError("tau != 0.50")
    ckpt = Path(ex_lock["checkpoint_path"])
    if _sha_file(ckpt) != ex_lock["checkpoint_sha256"]:
        raise RuntimeError("checkpoint sha mismatch")
    man = Path(ex_lock["confirm_manifest_path"])
    if _sha_file(man) != ex_lock["confirm_manifest_sha256"]:
        raise RuntimeError("confirm manifest sha mismatch")
    if not (FULLDEV / "SCALAR_PAIRWISE.FULLDEV.PASSED").exists():
        raise RuntimeError("FULLDEV.PASSED missing")
    sha = (OUT / "SCALAR_PAIRWISE.CONFIRM_EXECUTION_LOCK.sha256").read_text().strip()
    if ex_lock.get("execution_lock_body_sha256") != sha:
        raise RuntimeError("execution lock sha file mismatch")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--structural-only", action="store_true")
    args = ap.parse_args()

    ensure_dir(OUT)
    ensure_dir(OUT / "logs")
    ensure_dir(REPORT)

    for done in [
        "SCALAR_PAIRWISE.CONFIRM.CONFIRMED",
        "SCALAR_PAIRWISE.CONFIRM.PARTIALLY_CONFIRMED",
        "SCALAR_PAIRWISE.CONFIRM.NOT_CONFIRMED",
    ]:
        if (OUT / done).exists() and not args.smoke and not args.structural_only:
            raise SystemExit(f"confirm already finished: {done}")

    ex_path = OUT / "SCALAR_PAIRWISE.CONFIRM_EXECUTION_LOCK.json"
    if not ex_path.exists():
        _fail("CONFIRM_EXECUTION_LOCK missing")
    ex_lock = json.loads(ex_path.read_text())
    try:
        structural_sanity(ex_lock)
    except Exception as e:
        _fail(str(e))

    if args.structural_only:
        print("STRUCTURAL_SANITY_OK", flush=True)
        return

    if not args.smoke:
        (OUT / "SCALAR_PAIRWISE.CONFIRM.RUNNING").write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "method_lock_hash": EXPECTED_METHOD_LOCK_HASH,
                    "execution_lock_hash": ex_lock.get("execution_lock_body_sha256"),
                }
            )
            + "\n",
            encoding="utf-8",
        )

    t0 = time.time()
    try:
        if args.device == "cpu" or (args.device == "auto" and not torch.cuda.is_available()):
            device = torch.device("cpu")
        else:
            if not torch.cuda.is_available():
                _fail("cuda requested but unavailable")
            device = torch.device("cuda:0")
        print(f"device={device}", flush=True)

        cache_enr = OUT / ("confirm_enriched_smoke.parquet" if args.smoke else "confirm_enriched.parquet")
        cache_pairs = OUT / ("confirm_pairs_smoke.parquet" if args.smoke else "confirm_pairs.parquet")
        if cache_enr.exists() and cache_pairs.exists() and not args.smoke:
            print("loading cached enriched/pairs", flush=True)
            enr = pd.read_parquet(cache_enr)
            man = pd.read_csv(ex_lock["confirm_manifest_path"])
            pairs = pd.read_parquet(cache_pairs)
            enrich_stats = json.loads((OUT / "confirm_enrich_stats.json").read_text())
        else:
            print("building confirm enriched via Stage-6 rescoring...", flush=True)
            enr, man, enrich_stats = build_confirm_enriched(
                n_workers=args.workers, max_traces=(args.smoke or None)
            )
            frozen_preds = pd.read_parquet(ex_lock["confirm_frozen_predictions_path"])
            frozen_map = {
                str(r.trace_name): float(r.fixed_rescore_UNION)
                for r in frozen_preds.itertuples(index=False)
            }
            # if smoke, only check subset
            if args.smoke:
                frozen_map = {k: v for k, v in frozen_map.items() if k in set(man.trace_name.astype(str))}
            pairs, agree_info = build_confirm_pairs(enr, man, frozen_fixed_by_trace=frozen_map)
            enrich_stats.update(agree_info)
            enr.to_parquet(cache_enr, index=False)
            pairs.to_parquet(cache_pairs, index=False)
            save_json(enrich_stats, OUT / "confirm_enrich_stats.json")
            print("enrich_stats", enrich_stats, flush=True)

        coverage = {
            "n_traces": int(len(pairs)),
            "n_events": int(pairs.event_id.nunique()),
            "n_S_labels": int(np.isfinite(pairs.true_s_sample.to_numpy(float)).sum()),
            "n_with_candidates": int((pairs.n_candidates >= 1).sum()),
            "n_with_2plus_candidates": int((pairs.n_candidates >= 2).sum()),
        }
        save_json(coverage, OUT / "confirm_coverage.json")
        print("coverage", coverage, flush=True)

        true = pairs.true_s_sample.to_numpy(float)
        sr = pairs.sampling_rate_hz.to_numpy(float)
        # fixed = frozen Stage-6 predictions (aligned)
        frozen_preds = pd.read_parquet(ex_lock["confirm_frozen_predictions_path"])
        fmap = {str(r.trace_name): float(r.fixed_rescore_UNION) for r in frozen_preds.itertuples(index=False)}
        fixed_pred = np.asarray([fmap[str(t)] for t in pairs.trace_name], float)
        # sanity: equals c1
        if float(np.mean(np.abs(fixed_pred - pairs.c1_sample.to_numpy(float)) < 1e-3)) < 0.999:
            _fail("fixed frozen preds diverge from scored c1")

        resid_pred = resid_control_pred(
            pairs, lam=float(RESID_CONTROL["lambda"]), sigma=float(RESID_CONTROL["sigma_s"])
        )
        model = load_scalar_model(Path(ex_lock["checkpoint_path"]), device)
        p_switch = predict_p_switch(model, pairs, device)
        if np.any(~np.isfinite(p_switch[pairs.n_candidates.to_numpy(int) >= 2])):
            _fail("NaN model outputs on eligible pairs")
        scalar_pred, switch = apply_switch(pairs, p_switch, tau=float(ex_lock["tau"]))
        c1 = pairs.c1_sample.to_numpy(float)
        c2 = pairs.c2_sample.to_numpy(float)
        ok_set = np.isclose(scalar_pred, c1) | (np.isfinite(c2) & np.isclose(scalar_pred, c2))
        if not bool(np.all(ok_set)):
            _fail("prediction outside {c1,c2}")

        fixed_m = metrics_bundle(fixed_pred, true, sr)
        resid_m = metrics_bundle(resid_pred, true, sr)
        scalar_m = metrics_bundle(scalar_pred, true, sr)
        sw = switch_stats(pairs, scalar_pred, switch, true, sr)
        d_fixed = scalar_m["f1@0.5"] - fixed_m["f1@0.5"]
        d_resid = scalar_m["f1@0.5"] - resid_m["f1@0.5"]

        resid_m = {**resid_m, "delta_f1@0.5_vs_fixed": resid_m["f1@0.5"] - fixed_m["f1@0.5"]}
        scalar_out = {
            **scalar_m,
            **sw,
            "tau": float(ex_lock["tau"]),
            "delta_f1@0.5_vs_fixed": d_fixed,
            "delta_f1@0.1_vs_fixed": scalar_m["f1@0.1"] - fixed_m["f1@0.1"],
            "delta_f1@0.5_vs_resid": d_resid,
            "delta_f1@0.1_vs_resid": scalar_m["f1@0.1"] - resid_m["f1@0.1"],
            "delta_p95_vs_fixed": scalar_m["detected_ae_p95"] - fixed_m["detected_ae_p95"],
        }

        if args.smoke:
            print("SMOKE", {"fixed": fixed_m["f1@0.5"], "resid": resid_m["f1@0.5"], "scalar": scalar_m["f1@0.5"]})
            return

        # margin bins post-hoc
        ge2 = pairs[pairs.n_candidates >= 2].copy()
        qs = np.nanquantile(ge2.margin_fixed.to_numpy(float), [0.10, 0.30, 0.60])
        edges = [-np.inf, qs[0], qs[1], qs[2], np.inf]
        labels = ["lowest_10pct", "p10_p30", "p30_p60", "highest_40pct"]
        ge2["margin_bin"] = pd.cut(ge2.margin_fixed, bins=edges, labels=labels, include_lowest=True)
        idx_map = {t: i for i, t in enumerate(pairs.trace_name.astype(str))}
        bin_rows = []
        for lab in labels:
            sub = ge2[ge2.margin_bin == lab]
            if len(sub) == 0:
                continue
            ix = np.asarray([idx_map[t] for t in sub.trace_name.astype(str)], int)
            fm = metrics_bundle(fixed_pred[ix], true[ix], sr[ix])
            rm = metrics_bundle(resid_pred[ix], true[ix], sr[ix])
            sm = metrics_bundle(scalar_pred[ix], true[ix], sr[ix])
            st = switch_stats(pairs.iloc[ix].reset_index(drop=True), scalar_pred[ix], switch[ix], true[ix], sr[ix])
            bin_rows.append(
                {
                    "margin_bin": lab,
                    "n": len(sub),
                    "n_events": int(sub.event_id.nunique()),
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
        margin_df.to_csv(OUT / "confirm_margin_bins.csv", index=False)

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
        pd.DataFrame(cand_rows).to_csv(OUT / "confirm_candidate_count_bins.csv", index=False)

        cases = pairs.copy()
        cases["p_switch"] = p_switch
        cases["switched"] = switch
        cases["pred_scalar"] = scalar_pred
        cases["pred_fixed"] = fixed_pred
        cases["pred_resid"] = resid_pred
        cases.to_csv(OUT / "confirm_switch_cases.csv", index=False)

        print("bootstrap...", flush=True)
        boot_sf = event_bootstrap(scalar_pred, fixed_pred, true, sr, pairs.event_id.to_numpy(), n_boot=args.n_boot)
        boot_sr = event_bootstrap(scalar_pred, resid_pred, true, sr, pairs.event_id.to_numpy(), n_boot=args.n_boot)
        save_json(boot_sf, OUT / "confirm_bootstrap_scalar_vs_fixed.json")
        save_json(boot_sr, OUT / "confirm_bootstrap_scalar_vs_resid.json")

        # replication table
        held_df = 0.0207
        held_dr = 0.0207 - 0.0155  # scalar-resid on heldout ≈ 0.0052
        # from pilot report: scalar 0.9403, resid 0.9350 → 0.0052
        held_dr = 0.005255850852836197
        fd_df = 0.0137
        fd_dr = 0.0073
        repl = pd.DataFrame(
            [
                {
                    "split": "pilot-heldout",
                    "fixed": 0.9196,
                    "resid": 0.9350,
                    "scalar": 0.9403,
                    "scalar-fixed": held_df,
                    "scalar-resid": held_dr,
                },
                {
                    "split": "phaseB-full-dev",
                    "fixed": 0.8668,
                    "resid": 0.8732,
                    "scalar": 0.8805,
                    "scalar-fixed": fd_df,
                    "scalar-resid": fd_dr,
                },
                {
                    "split": "confirm",
                    "fixed": fixed_m["f1@0.5"],
                    "resid": resid_m["f1@0.5"],
                    "scalar": scalar_m["f1@0.5"],
                    "scalar-fixed": d_fixed,
                    "scalar-resid": d_resid,
                },
            ]
        )
        repl["ratio_scalar_fixed_vs_fulldev"] = repl["scalar-fixed"] / fd_df
        repl["ratio_scalar_fixed_vs_heldout"] = repl["scalar-fixed"] / held_df
        repl["ratio_scalar_resid_vs_fulldev"] = repl["scalar-resid"] / fd_dr
        repl["ratio_scalar_resid_vs_heldout"] = repl["scalar-resid"] / held_dr
        repl.to_csv(OUT / "effect_replication_summary.csv", index=False)

        ci_f = boot_sf["delta_f1@0.5"]["ci95"]
        ci_r = boot_sr["delta_f1@0.5"]["ci95"]
        vs_fixed_ok = ci_f[0] > 0
        vs_resid_ok = ci_r[0] > 0
        if vs_fixed_ok and vs_resid_ok:
            status = "CONFIRMED"
            interpretation = (
                "PAIRWISE CONFIRMED: scalar pairwise improved over both the fixed rescoring baseline "
                "and the residual-only control on the held-out confirm set "
                "(catalog-assisted S-phase candidate re-picking / reranking)."
            )
        elif vs_fixed_ok:
            status = "PARTIALLY_CONFIRMED"
            interpretation = (
                "PAIRWISE PARTIALLY CONFIRMED: pairwise retained improvement over the fixed baseline, "
                "while its incremental gain over the residual-only control was not confirmed."
            )
        else:
            status = "NOT_CONFIRMED"
            interpretation = (
                "PAIRWISE NOT CONFIRMED: the development-set improvement did not replicate on the confirm set."
            )

        save_json(fixed_m, OUT / "confirm_fixed_metrics.json")
        save_json(resid_m, OUT / "confirm_resid_metrics.json")
        save_json(scalar_out, OUT / "confirm_scalar_metrics.json")
        pd.DataFrame(
            [
                {"method": "fixed_rescore_UNION", **fixed_m},
                {"method": "resid_s_control", **resid_m},
                {"method": "scalar_pairwise", **scalar_out},
            ]
        ).to_csv(OUT / "confirm_comparison.csv", index=False)

        final = {
            "status": status,
            "interpretation": interpretation,
            "coverage": coverage,
            "fixed": fixed_m,
            "resid": resid_m,
            "scalar": scalar_out,
            "bootstrap_vs_fixed": boot_sf,
            "bootstrap_vs_resid": boot_sr,
            "effect_replication": repl.to_dict(orient="records"),
            "method_lock_hash": EXPECTED_METHOD_LOCK_HASH,
            "execution_lock_hash": ex_lock.get("execution_lock_body_sha256"),
            "confirm_is_evaluation_only": True,
            "elapsed_s": time.time() - t0,
            "device": str(device),
            "n_boot": args.n_boot,
            "margin_mechanism_note": (
                "post-hoc only; low-margin bins typically carry larger Δ vs fixed; "
                "must not redefine confirm method"
            ),
            "support_scalar_as_final_main_method": status == "CONFIRMED",
            "auto_next_stage": False,
        }
        save_json(final, OUT / "confirm_final.json")

        lines = [
            "# Scalar pairwise confirm report",
            "",
            f"**Status: {status}**",
            "",
            interpretation,
            "",
            f"n_traces={coverage['n_traces']} n_events={coverage['n_events']} "
            f"n_ge2={coverage['n_with_2plus_candidates']} tau=0.50",
            "",
            "## Metrics",
            f"- fixed  F1@0.5={fixed_m['f1@0.5']:.6f} F1@0.1={fixed_m['f1@0.1']:.6f} P95={fixed_m['detected_ae_p95']:.4f}",
            f"- resid  F1@0.5={resid_m['f1@0.5']:.6f} Δfixed={resid_m['delta_f1@0.5_vs_fixed']:+.6f}",
            f"- scalar F1@0.5={scalar_m['f1@0.5']:.6f} Δfixed={d_fixed:+.6f} Δresid={d_resid:+.6f}",
            f"- fixes={sw['fixes']} breaks={sw['breaks']} net={sw['net_fixes']} "
            f"switch_prec={sw['switch_precision']:.4f} Q2_rec={sw['Q2_recovery_rate']:.4f}",
            "",
            "## Bootstrap (event, 5000)",
            f"- vs fixed: {boot_sf['delta_f1@0.5']}",
            f"- vs resid: {boot_sr['delta_f1@0.5']}",
            "",
            "## Effect replication",
            repl.to_string(index=False),
            "",
            "## Margin bins (post-hoc)",
            margin_df.to_string(index=False),
            "",
            "Hard stop. No retrain. No confirm retune. No next stage.",
            "",
        ]
        (OUT / "confirm_report.md").write_text("\n".join(lines), encoding="utf-8")
        (REPORT / "scalar_pairwise_confirm_report.md").write_text("\n".join(lines), encoding="utf-8")

        run = OUT / "SCALAR_PAIRWISE.CONFIRM.RUNNING"
        if run.exists():
            run.unlink()

        marker = OUT / f"SCALAR_PAIRWISE.CONFIRM.{status}"
        marker.write_text(status + "\n", encoding="utf-8")
        save_json(final, OUT / f"SCALAR_PAIRWISE.CONFIRM.{status}.json")
        print(f"SCALAR_PAIRWISE.CONFIRM.{status}", flush=True)

    except SystemExit:
        raise
    except Exception as e:
        _fail(str(e), exception_type=type(e).__name__)


if __name__ == "__main__":
    main()
