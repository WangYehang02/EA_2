#!/usr/bin/env python
"""Run paper_strengthening_v1 strong controls on frozen pairs (no new experiments on confirm)."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, save_json
from earthquake.paper_strengthening import (
    C_CONFIGS,
    D_C_VALUES,
    D_TAU_VALUES,
    LinearPairwiseModel,
    fit_linear_pairwise,
    metrics_from_pred,
    paired_event_bootstrap_delta,
    pred_base_tau_c1c2,
    pred_fixed,
    pred_resid_control,
    select_best_by_f1,
    switch_mechanism,
)
from earthquake.pairwise.fulldev import TAU, apply_switch, load_scalar_model, predict_p_switch
from earthquake.utils import ensure_dir

OUT = artifacts_dir() / "results" / "paper_strengthening_v1"
REPORT = ROOT / "reports" / "paper_strengthening_v1"
MODELS = artifacts_dir() / "models" / "paper_strengthening_v1"


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def eval_split(pairs: pd.DataFrame, pred: np.ndarray) -> dict:
    true = pairs["true_s_sample"].to_numpy(float)
    sr = pairs["sampling_rate_hz"].to_numpy(float)
    met = metrics_from_pred(pred, true, sr)
    mech = switch_mechanism(pairs, pred)
    return {**met, **{f"mech_{k}": v for k, v in mech.items()}}


def main() -> int:
    t0 = time.time()
    ensure_dir(OUT / "dev")
    ensure_dir(OUT / "predictions")
    ensure_dir(OUT / "locks")
    ensure_dir(OUT / "audit")
    ensure_dir(MODELS)
    ensure_dir(REPORT)

    # --- audit notes from frozen artifacts ---
    audit = {
        "base_tau_as_expected_sig0p5": {
            "source_script": "scripts/audit_residual_control_equivalence.py",
            "formula": "fixed_candidate_scores(expected=base_tau_as_sample, sigma=0.5*sr, hist=all, lw=0.5, lh=2)",
            "historical_selection_scope": "FULL UNION argmax (audit)",
            "historical_phaseB_f1@0.5": 0.8791,
            "note": "Main table in this package restricts the SAME formula to shared c1/c2.",
        },
        "resid_s_control": {
            "formula": "fixed_score + 1.0*exp(-0.5*(resid_s/0.5)^2); switch if s2>s1",
            "scope_pairwise": "c1/c2 only",
            "fulldev_f1@0.5": 0.8732,
        },
        "scalar": {
            "ckpt": "artifacts/results/pairwise_pilot/ckpt_scalar_pairwise_seed42.pt",
            "ckpt_sha256": sha_file(ROOT / "artifacts/results/pairwise_pilot/ckpt_scalar_pairwise_seed42.pt"),
            "tau": 0.50,
            "features": 10,
            "standardization": "NONE in original scalar training",
            "pos_weight": "n_neg/n_pos on decisive train (~11.01); swap augmentation flips y",
            "pos_weight_issue": "Sample weight depends on label after optional swap; DOES affect original training. This package does NOT silently replace scalar; linear control forbids order-dependent weights.",
            "fulldev_f1@0.5": 0.8805,
        },
        "splits": {
            "protocol": "ranker_train_only_event_split_70_15_15 seed=42",
            "split_lock_sha256": sha_file(ROOT / "artifacts/results/pairwise_pilot/SPLIT.LOCK.json"),
            "heldout_and_fulldev_are_historical": True,
            "confirm_already_used_not_new_test": True,
        },
    }
    save_json(audit, OUT / "audit" / "implementation_audit.json")

    # --- load pairs ---
    pilot = pd.read_parquet(artifacts_dir() / "results/pairwise_pilot/pairs_ranker_train.parquet")
    phaseb = pd.read_parquet(artifacts_dir() / "results/pairwise_fulldev/pairs_phaseB.parquet")
    train = pilot[pilot["split"] == "train"].reset_index(drop=True)
    cal = pilot[pilot["split"] == "calibration"].reset_index(drop=True)
    held = pilot[pilot["split"] == "heldout_eval"].reset_index(drop=True)

    # sanity: calibration natural population includes singles
    cal_pop = {
        "n": int(len(cal)),
        "n_ge2": int((cal["n_candidates"] >= 2).sum()),
        "n_single": int((cal["n_candidates"] < 2).sum()),
        "n_decisive": int(cal["y_choose_c2"].isin([0, 1]).sum()),
        "label_class_counts": cal["label_class"].value_counts(dropna=False).to_dict(),
    }
    save_json(cal_pop, OUT / "dev" / "calibration_population.json")

    # ========== C grid on calibration ==========
    c_rows = []
    for i, (sig, lam) in enumerate(C_CONFIGS):
        pred = pred_base_tau_c1c2(cal, sigma_s=sig, lambda_history=lam)
        met = eval_split(cal, pred)
        row = {
            "method": "C_base_tau_c1c2",
            "sigma_s": sig,
            "lambda_history": lam,
            "order_index": i,
            "config_id": f"sig{sig}_lam{lam}",
            **{k: met[k] for k in ("f1@0.5", "f1@0.1", "detected_ae_p95", "miss_rate", "n")},
        }
        c_rows.append(row)
    c_df = pd.DataFrame(c_rows)
    c_df.to_csv(OUT / "dev" / "C_grid_calibration.csv", index=False)
    c_best = select_best_by_f1(c_rows, order_keys=["order_index"])
    save_json(c_best, OUT / "dev" / "C_best_calibration.json")

    # ========== D linear search ==========
    d_model_rows = []
    d_tau_rows = []
    best_d = None
    for j, Cval in enumerate(D_C_VALUES):
        model = fit_linear_pairwise(train, C=Cval)
        # save model
        blob = {
            "coef": model.coef.tolist(),
            "mean": model.mean.tolist(),
            "scale": model.scale.tolist(),
            "C": model.C,
            "feature_names": [
                "prob",
                "stead_prob",
                "ida_prob",
                "fixed_score",
                "resid_s",
                "resid_sp",
                "delta_sp",
                "src_stead",
                "src_ida",
                "src_both",
            ],
            "form": "sigmoid(w^T (x2-x1)); no intercept; train-only standardize; no sample weights",
        }
        mp = MODELS / f"linear_pairwise_C{Cval}.json"
        mp.write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")
        p_cal = model.p_switch(cal)
        for k, tau in enumerate(D_TAU_VALUES):
            pred = cal["c1_sample"].to_numpy(float).copy()
            ge2 = cal["n_candidates"].to_numpy(int) >= 2
            sw = ge2 & np.isfinite(p_cal) & (p_cal > tau)
            pred[sw] = cal["c2_sample"].to_numpy(float)[sw]
            met = eval_split(cal, pred)
            order_index = j * len(D_TAU_VALUES) + k
            row = {
                "method": "D_linear_pairwise",
                "C": Cval,
                "tau": tau,
                "order_index": order_index,
                "config_id": f"C{Cval}_tau{tau}",
                **{k2: met[k2] for k2 in ("f1@0.5", "f1@0.1", "detected_ae_p95", "miss_rate", "n")},
            }
            d_tau_rows.append(row)
            if best_d is None or select_best_by_f1([best_d, row], order_keys=["order_index"]) is row:
                best_d = row
                best_d_model = model
                best_d_tau = tau
        d_model_rows.append({"C": Cval, "model_path": str(mp), "model_sha256": sha_file(mp)})
    pd.DataFrame(d_tau_rows).to_csv(OUT / "dev" / "D_grid_calibration.csv", index=False)
    save_json(best_d, OUT / "dev" / "D_best_calibration.json")
    save_json({"models": d_model_rows}, OUT / "dev" / "D_models_index.json")
    # persist selected linear
    sel_lin = {
        "coef": best_d_model.coef.tolist(),
        "mean": best_d_model.mean.tolist(),
        "scale": best_d_model.scale.tolist(),
        "C": best_d_model.C,
        "tau": best_d_tau,
        "calibration_metrics": best_d,
    }
    (MODELS / "linear_pairwise_SELECTED.json").write_text(json.dumps(sel_lin, indent=2) + "\n", encoding="utf-8")

    # ========== B on calibration ==========
    pred_b_cal = pred_resid_control(cal)
    met_b = eval_split(cal, pred_b_cal)
    b_row = {"method": "B_resid_s_control", "order_index": 0, "config_id": "resid_lam1_sig0.5", **met_b}
    save_json(b_row, OUT / "dev" / "B_calibration.json")

    # ========== choose main strong control among B, C_best, D_best ==========
    candidates = [
        {
            "id": "B_resid_s_control",
            "family": "B",
            "f1@0.5": met_b["f1@0.5"],
            "f1@0.1": met_b["f1@0.1"],
            "order_index": 0,
            "detail": {"lam": 1.0, "sigma": 0.5},
        },
        {
            "id": "C_base_tau_c1c2",
            "family": "C",
            "f1@0.5": c_best["f1@0.5"],
            "f1@0.1": c_best["f1@0.1"],
            "order_index": 1 + int(c_best["order_index"]),
            "detail": {"sigma_s": c_best["sigma_s"], "lambda_history": c_best["lambda_history"], "config_id": c_best["config_id"]},
        },
        {
            "id": "D_linear_pairwise",
            "family": "D",
            "f1@0.5": best_d["f1@0.5"],
            "f1@0.1": best_d["f1@0.1"],
            "order_index": 100 + int(best_d["order_index"]),
            "detail": {"C": best_d["C"], "tau": best_d["tau"], "config_id": best_d["config_id"]},
        },
    ]
    main_ctrl = select_best_by_f1(candidates, order_keys=["order_index"])
    save_json(
        {
            "main_strong_control": main_ctrl,
            "candidates": candidates,
            "selection_population": "calibration natural",
            "criterion": "max F1@0.5 then F1@0.1 then pre-registered order",
            "frozen_before_new_external_test": True,
        },
        OUT / "locks" / "MAIN_STRONG_CONTROL.LOCK.json",
    )

    # ========== evaluate A–E on cal / heldout / fulldev ==========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scalar = load_scalar_model(ROOT / "artifacts/results/pairwise_pilot/ckpt_scalar_pairwise_seed42.pt", device)

    def run_all(pairs: pd.DataFrame, split_name: str) -> dict:
        preds = {}
        preds["A_fixed"] = pred_fixed(pairs)
        preds["B_resid"] = pred_resid_control(pairs)
        preds["C_best"] = pred_base_tau_c1c2(
            pairs, sigma_s=float(c_best["sigma_s"]), lambda_history=float(c_best["lambda_history"])
        )
        preds["D_best"], _ = best_d_model.predict(pairs, float(best_d_tau))
        p_sw = predict_p_switch(scalar, pairs, device)
        preds["E_scalar"], _ = apply_switch(pairs, p_sw, tau=TAU)

        # also exact registered base_tau sig0.5 lh2 for reference
        preds["C_registered_sig0p5_lam2"] = pred_base_tau_c1c2(pairs, sigma_s=0.5, lambda_history=2.0)

        rows = []
        for name, pred in preds.items():
            met = eval_split(pairs, pred)
            rows.append({"split": split_name, "method": name, **met})
            # save predictions hash
            arr = pred.astype(np.float64)
            np.save(OUT / "predictions" / f"{split_name}__{name}.npy", arr)
            (OUT / "predictions" / f"{split_name}__{name}.sha256").write_text(
                hashlib.sha256(arr.tobytes()).hexdigest() + "\n", encoding="utf-8"
            )
        return {"rows": rows, "preds": preds}

    results_tables = []
    bundle = {}
    for split_name, df in [
        ("calibration", cal),
        ("heldout_eval", held),
        ("fulldev_phaseB", phaseb),
    ]:
        out = run_all(df, split_name)
        bundle[split_name] = {k: v for k, v in out.items() if k != "preds"}
        results_tables.extend(out["rows"])
        # primary comparison on this split: E vs main strong control
        ctrl_name = {"B": "B_resid", "C": "C_best", "D": "D_best"}[main_ctrl["family"]]
        pred_e = out["preds"]["E_scalar"]
        pred_c = out["preds"][ctrl_name]
        boot = paired_event_bootstrap_delta(
            df["event_id"].to_numpy(),
            pred_e,
            pred_c,
            df["true_s_sample"].to_numpy(float),
            df["sampling_rate_hz"].to_numpy(float),
            n_boot=5000,
            seed=42,
        )
        bundle[split_name]["primary_comparison"] = {
            "definition": "E_scalar - main_strong_control",
            "main_strong_control": main_ctrl["id"],
            "control_pred_key": ctrl_name,
            "point_delta_f1@0.5": float(
                metrics_from_pred(pred_e, df["true_s_sample"].to_numpy(float), df["sampling_rate_hz"].to_numpy(float))[
                    "f1@0.5"
                ]
                - metrics_from_pred(pred_c, df["true_s_sample"].to_numpy(float), df["sampling_rate_hz"].to_numpy(float))[
                    "f1@0.5"
                ]
            ),
            "bootstrap": boot,
            "status_note": (
                "historical_research_split_not_independent"
                if split_name != "calibration"
                else "calibration_used_for_control_selection_not_independent_test"
            ),
        }
        # free preds from memory in bundle dump
        del out["preds"]

    tab = pd.DataFrame(results_tables)
    tab.to_csv(OUT / "dev" / "methods_A_to_E_dev_splits.csv", index=False)

    # smoke: mutating labels must not change E prediction
    smoke = held.head(200).copy()
    p0 = predict_p_switch(scalar, smoke, device)
    pred0, _ = apply_switch(smoke, p0, tau=TAU)
    smoke2 = smoke.copy()
    smoke2["true_s_sample"] = smoke2["true_s_sample"] + 5000
    p1 = predict_p_switch(scalar, smoke2, device)
    pred1, _ = apply_switch(smoke2, p1, tau=TAU)
    label_isolation_ok = bool(np.allclose(pred0, pred1, equal_nan=True))
    save_json(
        {"label_isolation_ok": label_isolation_ok, "n": int(len(smoke))},
        OUT / "audit" / "label_isolation_smoke.json",
    )

    # supplement: full-UNION base_tau on phaseB if enriched available
    union_note = {"attempted": False}
    enr_path = artifacts_dir() / "results/multistation_moveout/phaseB_enriched_candidate_table.parquet"
    man_path = artifacts_dir() / "results/stage6/phaseB_eval_manifest.csv"
    if enr_path.exists() and man_path.exists():
        from earthquake.multistation.soft_ring_rescore import fixed_candidate_scores

        union_note["attempted"] = True
        meta = pd.read_csv(man_path)
        table = pd.read_parquet(enr_path)
        names = meta.trace_name.astype(str).to_numpy()
        true = meta.s_arrival_sample.to_numpy(float)
        sr = meta.sampling_rate_hz.to_numpy(float)
        # rebuild registered score
        sr_arr = table["sampling_rate_hz"].to_numpy(float)
        origin = pd.to_datetime(table["origin_time"], utc=True, errors="coerce")
        start = pd.to_datetime(table["trace_start_time"], utc=True, errors="coerce")
        offset = (start - origin).dt.total_seconds().to_numpy(float)
        base_samp = (table["base_tau_s"].to_numpy(float) - offset) * sr_arr
        sc = fixed_candidate_scores(
            cand_sample=table["candidate_sample"].to_numpy(float),
            cand_prob=table["cand_prob"].to_numpy(float),
            expected_s_sample=base_samp,
            history_sigma_samples=0.5 * sr_arr,
            history_available=np.ones(len(table), dtype=bool),
            lw=0.5,
            lh=2.0,
            lp=0.0,
        )
        table = table.copy()
        table["score_base_tau_union"] = sc
        idx = table.groupby("trace_name")["score_base_tau_union"].idxmax()
        pred_u = table.loc[idx].set_index("trace_name")["candidate_sample"].reindex(names).to_numpy(float)
        met_u = metrics_from_pred(pred_u, true, sr)
        # c1/c2 restricted registered
        pred_c12 = pred_base_tau_c1c2(phaseb, sigma_s=0.5, lambda_history=2.0)
        met_c12 = metrics_from_pred(pred_c12, phaseb["true_s_sample"].to_numpy(float), phaseb["sampling_rate_hz"].to_numpy(float))
        union_note.update(
            {
                "full_UNION_base_tau_sig0p5_lh2": met_u,
                "c1c2_base_tau_sig0p5_lh2": met_c12,
                "difference_note": "UNION may pick rank>=3; MAIN comparison uses c1/c2 only",
            }
        )
    save_json(union_note, OUT / "dev" / "C_union_vs_c1c2_supplement.json")

    summary = {
        "elapsed_s": time.time() - t0,
        "main_strong_control": main_ctrl,
        "C_best": c_best,
        "D_best": best_d,
        "B_calibration_f1@0.5": met_b["f1@0.5"],
        "dev_results_csv": str(OUT / "dev" / "methods_A_to_E_dev_splits.csv"),
        "primary_comparisons": {k: bundle[k]["primary_comparison"] for k in bundle},
        "label_isolation_ok": label_isolation_ok,
        "independent_validation": "NOT_RUN_IN_THIS_SCRIPT",
        "device": str(device),
    }
    save_json(summary, OUT / "dev" / "strong_controls_summary.json")
    save_json(bundle, OUT / "dev" / "dev_eval_bundle.json")

    # markdown report fragment
    def r4(x):
        return f"{float(x):.4f}"

    lines = [
        "# Paper strengthening v1 — strong controls (dev)",
        "",
        f"- Main strong control (calibration-selected): **{main_ctrl['id']}**",
        f"- Detail: `{json.dumps(main_ctrl.get('detail', {}), sort_keys=True)}`",
        f"- Label isolation smoke: **{label_isolation_ok}**",
        "",
        "## Calibration selection",
        "",
        f"| Family | F1@0.5 | F1@0.1 | Config |",
        f"| --- | ---: | ---: | --- |",
        f"| B resid | {r4(met_b['f1@0.5'])} | {r4(met_b['f1@0.1'])} | lam=1,σ=0.5 |",
        f"| C best | {r4(c_best['f1@0.5'])} | {r4(c_best['f1@0.1'])} | {c_best['config_id']} |",
        f"| D best | {r4(best_d['f1@0.5'])} | {r4(best_d['f1@0.1'])} | {best_d['config_id']} |",
        "",
        "## A–E on historical splits (NOT independent validation)",
        "",
    ]
    pivot = tab.pivot_table(index="method", columns="split", values="f1@0.5")
    lines.append(pivot.to_csv())
    lines.append("")
    lines.append("## Primary comparison: E_scalar − main_strong_control")
    lines.append("")
    for split_name, pc in summary["primary_comparisons"].items():
        b = pc["bootstrap"]
        lines.append(
            f"- **{split_name}**: Δ={r4(pc['point_delta_f1@0.5'])}, "
            f"boot mean={r4(b['mean'])}, CI=[{r4(b['ci95'][0])}, {r4(b['ci95'][1])}] "
            f"({pc['status_note']})"
        )
    lines.append("")
    lines.append("## Independent validation")
    lines.append("")
    lines.append("**INDEPENDENT_VALIDATION_NOT_COMPLETED** in this run (see INGV probe / data requirements).")
    lines.append("")
    (REPORT / "strong_controls_dev_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2)[:2000])
    print("WROTE", REPORT / "strong_controls_dev_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
