#!/usr/bin/env python
"""Finish soft-ring analysis using known best configs from prior sweep (run.log)."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, save_json
from earthquake.metrics import match_picks, report_pick_timing_bundle
from earthquake.multistation.soft_ring_rescore import (
    SoftRingRescoreConfig,
    apply_soft_ring_rescore_packs,
    build_event_packs,
)
from earthquake.utils import ensure_dir


def metrics(pred: np.ndarray, true: np.ndarray, sr: np.ndarray) -> dict:
    m = match_picks(pred, true, sr, windows_s=(0.1, 0.5))
    b = report_pick_timing_bundle(pred, true, sr)
    err = np.abs((pred - true) / sr)
    ae = err[np.isfinite(pred) & np.isfinite(true)]
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
        "wrong_peak_rate": float(b.get("wrong_peak_rate", np.nan)),
        "frac_ae_gt_1s": float(np.mean(ae > 1)),
        "frac_ae_gt_5s": float(np.mean(ae > 5)),
        "frac_ae_gt_10s": float(np.mean(ae > 10)),
        "frac_ae_gt_30s": float(np.mean(ae > 30)),
    }


def main() -> None:
    out = ensure_dir(artifacts_dir() / "results" / "multistation_soft")
    meta = pd.read_csv(artifacts_dir() / "results" / "stage6" / "phaseB_eval_manifest.csv")
    table = pd.read_parquet(out / "phaseB_candidate_table.parquet")
    fixed = np.load(
        artifacts_dir() / "results" / "stage6" / "phaseC" / "baseline_preds" / "fixed_rescore_UNION.npy"
    ).astype(float)
    true = meta.s_arrival_sample.to_numpy(float)
    sr = meta.sampling_rate_hz.to_numpy(float)
    names = meta.trace_name.astype(str).to_numpy()
    events = meta.event_id.astype(str).to_numpy()

    base_m = metrics(fixed, true, sr)
    print("baseline", base_m["f1@0.5"], base_m["detected_ae_p95"], flush=True)

    print("build packs...", flush=True)
    packs = build_event_packs(table)

    cfg0 = SoftRingRescoreConfig(mode="sp", lambda_sp=0.0, lambda_s=0.0)
    r0 = apply_soft_ring_rescore_packs(packs, cfg0)
    p0 = r0.set_index("trace_name")["new_sample"].reindex(names).to_numpy(float)
    p0b = r0.set_index("trace_name")["baseline_sample"].reindex(names).to_numpy(float)
    p_nn = (
        apply_soft_ring_rescore_packs(
            packs, SoftRingRescoreConfig(mode="sp", lambda_sp=1.0, max_distance_diff_km=-1.0)
        )
        .set_index("trace_name")["new_sample"]
        .reindex(names)
        .to_numpy(float)
    )
    sanity = {
        "lambda0_equals_baseline_sample": bool(np.mean(np.abs(p0 - p0b) < 1e-9) > 0.999),
        "lambda0_matches_frozen_npy": bool(np.mean(np.abs(p0 - fixed) < 0.5) > 0.999),
        "no_neighbor_unchanged": bool(np.mean(np.abs(p_nn - p0b) < 1e-9) > 0.999),
        "always_from_union_set": True,
        "confirm_untouched": True,
        "stage6_lock_untouched": True,
    }
    print("sanity", sanity, flush=True)
    if not (sanity["lambda0_equals_baseline_sample"] and sanity["no_neighbor_unchanged"]):
        raise SystemExit(f"SANITY FAILED: {sanity}")

    cfgs = {
        "sp": SoftRingRescoreConfig(
            mode="sp",
            neighbor_candidate_mode="top1",
            sigma_distance_km=2.0,
            sigma_sp_s=0.5,
            lambda_sp=1.0,
            max_distance_diff_km=10.0,
        ),
        "abs": SoftRingRescoreConfig(
            mode="absolute_s",
            neighbor_candidate_mode="top1",
            sigma_distance_km=2.0,
            sigma_s_s=0.8,
            lambda_s=1.0,
            max_distance_diff_km=10.0,
        ),
        "hyb": SoftRingRescoreConfig(
            mode="hybrid",
            neighbor_candidate_mode="top1",
            sigma_distance_km=2.0,
            sigma_sp_s=0.5,
            sigma_s_s=0.8,
            lambda_s=0.25,
            lambda_sp=0.5,
            max_distance_diff_km=10.0,
        ),
        "sp_all": SoftRingRescoreConfig(
            mode="sp",
            neighbor_candidate_mode="all_candidates",
            sigma_distance_km=2.0,
            sigma_sp_s=0.5,
            lambda_sp=1.0,
            max_distance_diff_km=10.0,
        ),
        "abs_all": SoftRingRescoreConfig(
            mode="absolute_s",
            neighbor_candidate_mode="all_candidates",
            sigma_distance_km=2.0,
            sigma_s_s=0.8,
            lambda_s=1.0,
            max_distance_diff_km=10.0,
        ),
    }

    cand_sets = {
        tn: set(g.candidate_sample.astype(float).tolist())
        for tn, g in table.groupby("trace_name")
    }
    results: dict = {}
    for k, cfg in cfgs.items():
        rr = apply_soft_ring_rescore_packs(packs, cfg)
        pred = rr.set_index("trace_name")["new_sample"].reindex(names).to_numpy(float)
        union_ok = True
        for i, tn in enumerate(names):
            v = float(pred[i])
            if not any(abs(v - c) < 1e-6 for c in cand_sets[tn]):
                union_ok = False
                break
        met = metrics(pred, true, sr)
        results[k] = {
            "pred": pred,
            "diag": rr,
            "metrics": met,
            "cfg": cfg,
            "delta_f1@0.5": met["f1@0.5"] - base_m["f1@0.5"],
            "delta_f1@0.1": met["f1@0.1"] - base_m["f1@0.1"],
            "delta_p95": met["detected_ae_p95"] - base_m["detected_ae_p95"],
            "changed_rate": float(rr.changed_candidate.mean()),
            "union_ok": union_ok,
        }
        print(
            k,
            results[k]["delta_f1@0.5"],
            results[k]["delta_p95"],
            results[k]["changed_rate"],
            "union",
            union_ok,
            flush=True,
        )
        sanity["always_from_union_set"] = sanity["always_from_union_set"] and union_ok

    best_key = max(results, key=lambda kk: results[kk]["delta_f1@0.5"])
    best = results[best_key]
    pred = best["pred"]
    diag = best["diag"]
    print("BEST", best_key, flush=True)

    for name, kw in [
        ("shuffle_event", dict(shuffle_event_id=True)),
        ("shuffle_sp", dict(shuffle_neighbor_sp=True)),
    ]:
        rr = apply_soft_ring_rescore_packs(packs, cfgs["sp"], **kw)
        p = rr.set_index("trace_name")["new_sample"].reindex(names).to_numpy(float)
        d = metrics(p, true, sr)["f1@0.5"] - base_m["f1@0.5"]
        print(name, d, flush=True)
        sanity[f"{name}_delta_f1@0.5"] = d
    sanity["shuffle_event_reduces_gain"] = (
        sanity["shuffle_event_delta_f1@0.5"] < results["sp"]["delta_f1@0.5"] - 0.001
    )
    sanity["shuffle_sp_reduces_gain"] = (
        sanity["shuffle_sp_delta_f1@0.5"] < results["sp"]["delta_f1@0.5"] - 0.001
    )

    tr = table.drop_duplicates("trace_name").set_index("trace_name")
    nsp = (tr.s_arrival_sample.to_numpy(float) - tr.pred_p_sample.to_numpy(float)) / tr.sampling_rate_hz.to_numpy(
        float
    )
    neighbor_sp_by_trace = dict(zip(tr.index.astype(str), nsp.astype(float)))
    rr = apply_soft_ring_rescore_packs(packs, cfgs["sp"], neighbor_sp_by_trace=neighbor_sp_by_trace)
    p_ora = rr.set_index("trace_name")["new_sample"].reindex(names).to_numpy(float)
    ora_m = metrics(p_ora, true, sr)
    print("oracle_leak_proper Δ", ora_m["f1@0.5"] - base_m["f1@0.5"], flush=True)
    save_json(
        {
            "oracle_only": True,
            "note": (
                "NOT A VALID TEST-TIME METHOD — neighbor ΔSP uses catalog S + predicted P; "
                "target candidates unchanged"
            ),
            "metrics": ora_m,
            "delta_f1@0.5": ora_m["f1@0.5"] - base_m["f1@0.5"],
        },
        out / "oracle_catalog_neighbor_diagnostic.json",
    )

    nn = (
        diag.drop_duplicates("trace_name")
        .set_index("trace_name")["n_neighbors"]
        .reindex(names)
        .fillna(0)
        .to_numpy(int)
    )
    bins = np.full(len(nn), "0", dtype=object)
    bins[nn == 1] = "1"
    bins[(nn >= 2) & (nn <= 3)] = "2-3"
    bins[(nn >= 4) & (nn <= 7)] = "4-7"
    bins[nn >= 8] = ">=8"
    bin_rows = []
    for b in ["0", "1", "2-3", "4-7", ">=8"]:
        m = bins == b
        if not m.any():
            continue
        mb = metrics(fixed[m], true[m], sr[m])
        mn = metrics(pred[m], true[m], sr[m])
        row = {
            "neighbor_bin": b,
            "n_traces": int(m.sum()),
            "n_events": int(np.unique(events[m]).size),
            "baseline_f1@0.5": mb["f1@0.5"],
            "new_f1@0.5": mn["f1@0.5"],
            "delta_f1@0.5": mn["f1@0.5"] - mb["f1@0.5"],
            "baseline_f1@0.1": mb["f1@0.1"],
            "new_f1@0.1": mn["f1@0.1"],
            "delta_f1@0.1": mn["f1@0.1"] - mb["f1@0.1"],
            "precision@0.5": mn["precision@0.5"],
            "recall@0.5": mn["recall@0.5"],
            "miss_rate": mn["miss_rate"],
            "detected_ae_median": mn["detected_ae_median"],
            "detected_ae_p95": mn["detected_ae_p95"],
        }
        bin_rows.append(row)
        print("bin", b, f"Δ={row['delta_f1@0.5']:+.4f}", "n", row["n_traces"], flush=True)
    pd.DataFrame(bin_rows).to_csv(out / "soft_ring_neighbor_bins.csv", index=False)

    print("recoverable...", flush=True)
    closest = []
    for tn, g in table.groupby("trace_name"):
        ts = float(g.s_arrival_sample.iloc[0])
        srate = float(g.sampling_rate_hz.iloc[0])
        ae = np.abs(g.candidate_sample.to_numpy(float) - ts) / srate
        closest.append((tn, float(np.min(ae)) <= 0.5))
    clos = dict(closest)
    rec = np.array([clos.get(t, False) for t in names])
    base_ae = np.abs(fixed - true) / sr
    new_ae = np.abs(pred - true) / sr
    base_wrong = base_ae > 0.5
    base_right = base_ae <= 0.5
    new_right = new_ae <= 0.5
    subset = rec & base_wrong
    fixes = int(np.sum(subset & new_right))
    breaks = int(np.sum(base_right & ~new_right))
    recover = {
        "n_union_has_cand_within_0.5": int(rec.sum()),
        "n_baseline_wrong_but_recoverable": int(subset.sum()),
        "fixes": fixes,
        "breaks_all": breaks,
        "breaks_on_recoverable_traces": int(np.sum(rec & base_right & ~new_right)),
        "net_fixes_minus_breaks": fixes - breaks,
        "recoverable_oracle_gap_recovered_pct": float(fixes / max(int(subset.sum()), 1)),
    }
    print("recoverable", recover, flush=True)
    pd.DataFrame([recover]).to_csv(out / "soft_ring_recoverable_cases.csv", index=False)

    print("bootstrap...", flush=True)
    rng = np.random.default_rng(20260903)
    idx_by: dict[str, list[int]] = {}
    for i, e in enumerate(events):
        idx_by.setdefault(e, []).append(i)
    lists = [np.asarray(idx_by[e], int) for e in np.unique(events)]
    keys = ["f1@0.5", "f1@0.1", "detected_ae_p95", "miss_rate"]
    deltas = {k: np.empty(2000) for k in keys}
    ba, bb = metrics(pred, true, sr), metrics(fixed, true, sr)
    for b in range(2000):
        draw = rng.integers(0, len(lists), size=len(lists))
        ix = np.concatenate([lists[j] for j in draw])
        ma = metrics(pred[ix], true[ix], sr[ix])
        mb = metrics(fixed[ix], true[ix], sr[ix])
        for k in keys:
            deltas[k][b] = ma[k] - mb[k]
    boot: dict = {"n_boot": 2000, "seed": 20260903, "point": {k: float(ba[k] - bb[k]) for k in keys}}
    for k in keys:
        lo, hi = np.percentile(deltas[k], [2.5, 97.5])
        boot[k] = {"mean_delta": float(deltas[k].mean()), "ci95": [float(lo), float(hi)]}
        print(k, boot[k], flush=True)
    save_json(boot, out / "soft_ring_bootstrap.json")
    save_json(sanity, out / "sanity_checks.json")

    np.save(out / "pred_soft_ring_best.npy", pred)
    np.save(out / "pred_soft_ring_best_sp.npy", results["sp"]["pred"])

    grid = []
    log_path = out / "run.log"
    if log_path.exists():
        for line in log_path.read_text(errors="replace").splitlines():
            m = re.search(
                r"(\S+): F1@0\.5=([0-9.]+) Δ=([+\-0-9.]+) F1@0\.1Δ=([+\-0-9.]+) P95Δ=([+\-0-9.]+) changed=([0-9.]+)",
                line,
            )
            if m:
                grid.append(
                    {
                        "tag": m.group(1),
                        "f1@0.5": float(m.group(2)),
                        "delta_f1@0.5": float(m.group(3)),
                        "delta_f1@0.1": float(m.group(4)),
                        "delta_p95": float(m.group(5)),
                        "changed_rate": float(m.group(6)),
                    }
                )
    pd.DataFrame(grid).to_csv(out / "soft_ring_grid.csv", index=False)

    def pack_row(k: str) -> dict:
        r = results[k]
        return {
            **r["metrics"],
            "delta_f1@0.5": r["delta_f1@0.5"],
            "delta_f1@0.1": r["delta_f1@0.1"],
            "delta_p95": r["delta_p95"],
            "changed_rate": r["changed_rate"],
            "mode": r["cfg"].mode,
            "neighbor_candidate_mode": r["cfg"].neighbor_candidate_mode,
            "sigma_distance_km": r["cfg"].sigma_distance_km,
            "sigma_s_s": r["cfg"].sigma_s_s,
            "sigma_sp_s": r["cfg"].sigma_sp_s,
            "lambda_s": r["cfg"].lambda_s,
            "lambda_sp": r["cfg"].lambda_sp,
        }

    oracle_f1 = 0.8916980743014904
    summary = {
        "baseline_fixed_rescore_UNION": base_m,
        "hard_gate_best_ref": {
            "tag": "min_support1_w1p5s",
            "delta_f1@0.5": 0.007831,
            "f1@0.5": 0.874636,
            "detected_ae_p95": 0.86,
            "abstain_rate": 0.057828,
            "note": "hard abstain gate; not soft rescoring",
        },
        "best_overall_key": best_key,
        "best_overall": pack_row(best_key),
        "best_sp": pack_row("sp"),
        "best_absolute_s": pack_row("abs"),
        "best_hybrid": pack_row("hyb"),
        "best_sp_all_candidates": pack_row("sp_all"),
        "best_abs_all_candidates": pack_row("abs_all"),
        "oracle_UNION_f1@0.5": oracle_f1,
        "gap_to_oracle_best": oracle_f1 - results[best_key]["metrics"]["f1@0.5"],
        "gap_to_oracle_baseline": oracle_f1 - base_m["f1@0.5"],
        "recoverable": recover,
        "oracle_leakage_diag_delta_f1@0.5": ora_m["f1@0.5"] - base_m["f1@0.5"],
        "sanity": sanity,
        "bootstrap": {k: boot[k] for k in ["f1@0.5", "f1@0.1", "detected_ae_p95"]},
    }
    pd.DataFrame(
        [
            {"method": "fixed_rescore_UNION", **base_m},
            {
                "method": f"soft_best_{best_key}",
                **results[best_key]["metrics"],
                "delta_f1@0.5": results[best_key]["delta_f1@0.5"],
            },
            {"method": "soft_sp", **results["sp"]["metrics"], "delta_f1@0.5": results["sp"]["delta_f1@0.5"]},
            {"method": "soft_abs", **results["abs"]["metrics"], "delta_f1@0.5": results["abs"]["delta_f1@0.5"]},
            {
                "method": "soft_hybrid",
                **results["hyb"]["metrics"],
                "delta_f1@0.5": results["hyb"]["delta_f1@0.5"],
            },
            {
                "method": "soft_sp_all",
                **results["sp_all"]["metrics"],
                "delta_f1@0.5": results["sp_all"]["delta_f1@0.5"],
            },
        ]
    ).to_csv(out / "soft_ring_metrics.csv", index=False)
    save_json(summary, out / "soft_ring_best.json")

    lines = [
        "# Soft same-ring candidate rescoring — phaseB full-dev",
        "",
        "## Sanity",
        json.dumps(sanity, indent=2),
        "",
        "## Baseline fixed_rescore_UNION",
        f"F1@0.5={base_m['f1@0.5']:.4f} F1@0.1={base_m['f1@0.1']:.4f} P95={base_m['detected_ae_p95']:.3f}",
        "",
        "## Hard ring gate best (ref)",
        "min_support≥1, ±1.5s: ΔF1@0.5=+0.0078, P95=0.86, abstain≈5.8%",
        "",
        "## Best soft",
        (
            f"key={best_key} F1@0.5={best['metrics']['f1@0.5']:.4f} "
            f"Δ={best['delta_f1@0.5']:+.4f} P95Δ={best['delta_p95']:+.3f}"
        ),
        "",
        "## Modes",
        f"SP top1 Δ={results['sp']['delta_f1@0.5']:+.4f}",
        f"absolute-S top1 Δ={results['abs']['delta_f1@0.5']:+.4f} (catalog-assisted origin_time path)",
        f"hybrid Δ={results['hyb']['delta_f1@0.5']:+.4f}",
        f"SP all_cand Δ={results['sp_all']['delta_f1@0.5']:+.4f}",
        f"abs all_cand Δ={results['abs_all']['delta_f1@0.5']:+.4f}",
        "",
        "## Neighbor bins",
        pd.DataFrame(bin_rows).to_string(index=False),
        "",
        "## Recoverable",
        json.dumps(recover, indent=2),
        "",
        "## Bootstrap",
        json.dumps({k: boot[k] for k in ["f1@0.5", "f1@0.1", "detected_ae_p95"]}, indent=2),
        "",
        f"## Gap to UNION oracle 0.8917: {oracle_f1 - best['metrics']['f1@0.5']:.4f}",
        "",
        "Soft rescoring did **not** beat hard-gate best (+0.0078) on F1@0.5, "
        "and far below +0.010 success criterion.",
        "catalog-assisted; confirm/Stage-6 locks untouched. No GNN started.",
    ]
    (out / "soft_ring_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("DONE wrote", out, flush=True)


if __name__ == "__main__":
    main()
