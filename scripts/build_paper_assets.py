#!/usr/bin/env python
"""Build paper tables/figures from frozen Stage 2–4 artifacts (no re-training)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, save_json
from earthquake.utils import ensure_dir

plt.rcParams.update({"font.size": 9, "figure.dpi": 300, "savefig.dpi": 300, "pdf.fonttype": 42})


def loadj(p: Path):
    return json.loads(Path(p).read_text())


def save_fig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"))
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    ad = artifacts_dir()
    paper = ensure_dir(ROOT / "paper")
    tables = ensure_dir(paper / "tables")
    figs = ensure_dir(paper / "figures")
    reports = ensure_dir(ROOT / "reports")
    ledger = []

    def add(name, value, source, key, stage, role, used_for_tuning=False, has_ci=False, allow_abstract=False):
        ledger.append(
            {
                "name": name,
                "value": value,
                "source": source,
                "json_key": key,
                "stage": stage,
                "role": role,
                "used_for_tuning": used_for_tuning,
                "has_event_ci": has_ci,
                "allow_in_abstract": allow_abstract,
            }
        )

    # --- Stage 3 main metrics ---
    s3_eval = loadj(ad / "results/stage3/learned_gate_eval_test.json")
    agg = s3_eval["aggregate"]
    boot_fx = loadj(ad / "results/stage3/bootstrap_fixed_vs_phasenet.json")
    boot_gate = loadj(ad / "results/stage3/bootstrap_gate.json")
    ora = loadj(ad / "results/stage3/oracle_selector_ceiling.json")
    split3 = loadj(ad / "results/stage3/split_audit.json")
    best = loadj(ad / "results/stage2/best_lambdas.json")
    s4v = loadj(ad / "results/stage4/stage4_final_verdict.json")
    s4a = loadj(ad / "results/stage4/holdout_split_audit.json")
    s4boot = loadj(ad / "results/stage4/bootstrap_confirmatory.json")

    def g(key):
        return float(agg[key]["mean"])

    main_rows = [
        {"method": "PhaseNet (STEAD)", "S_F1_0.1": g("phasenet__f1@0.1"), "S_F1_0.5": g("phasenet__f1@0.5"), "S_e2e_P95_s": g("phasenet__e2e_p95"), "S_e2e_MAE_s": g("phasenet__e2e_mae"), "role": "baseline"},
        {"method": "fixed catalog_rescore", "S_F1_0.1": g("fixed_catalog_rescore__f1@0.1"), "S_F1_0.5": g("fixed_catalog_rescore__f1@0.5"), "S_e2e_P95_s": g("fixed_catalog_rescore__e2e_p95"), "S_e2e_MAE_s": g("fixed_catalog_rescore__e2e_mae"), "role": "main"},
        {"method": "learned gate (ablation)", "S_F1_0.1": g("learned_gate__f1@0.1"), "S_F1_0.5": g("learned_gate__f1@0.5"), "S_e2e_P95_s": g("learned_gate__e2e_p95"), "S_e2e_MAE_s": g("learned_gate__e2e_mae"), "role": "ablation"},
        {"method": "oracle PN/fixed selector", "S_F1_0.1": float(ora["oracle_source_selector"]["f1@0.1s"]), "S_F1_0.5": float(ora["oracle_source_selector"]["f1@0.5s"]), "S_e2e_P95_s": float(ora["oracle_source_selector"]["p95_ae"]), "S_e2e_MAE_s": float(ora["oracle_source_selector"]["mae"]), "role": "diagnostic_ceiling"},
        {"method": "oracle candidate K=5", "S_F1_0.1": float(ora["oracle_candidate_ceiling"][2]["oracle_f1@0.1"]), "S_F1_0.5": float(ora["oracle_candidate_ceiling"][2]["oracle_f1@0.5"]), "S_e2e_P95_s": float(ora["oracle_candidate_ceiling"][2]["oracle_e2e_p95"]), "S_e2e_MAE_s": float(ora["oracle_candidate_ceiling"][2]["oracle_e2e_mae"]), "role": "diagnostic_ceiling"},
    ]
    pd.DataFrame(main_rows).to_csv(tables / "table_main_stage3.csv", index=False)
    # also write LaTeX-friendly
    with open(tables / "table_main_stage3.tex", "w") as f:
        f.write("\\begin{tabular}{lrrrr}\n\\toprule\nMethod & F1@0.1 & F1@0.5 & e2e P95 (s) & e2e MAE (s) \\\\\n\\midrule\n")
        for r in main_rows:
            name = str(r["method"]).replace("_", "\\_")
            f.write(f"{name} & {r['S_F1_0.1']:.3f} & {r['S_F1_0.5']:.3f} & {r['S_e2e_P95_s']:.2f} & {r['S_e2e_MAE_s']:.2f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    add("stage3_pn_f1_05", main_rows[0]["S_F1_0.5"], "stage3/learned_gate_eval_test.json", "aggregate.phasenet__f1@0.5.mean", 3, "test", False, False, True)
    add("stage3_fx_f1_05", main_rows[1]["S_F1_0.5"], "stage3/learned_gate_eval_test.json", "aggregate.fixed_catalog_rescore__f1@0.5.mean", 3, "test", False, True, True)
    add("stage3_pn_p95", main_rows[0]["S_e2e_P95_s"], "stage3/learned_gate_eval_test.json", "aggregate.phasenet__e2e_p95.mean", 3, "test", False, False, True)
    add("stage3_fx_p95", main_rows[1]["S_e2e_P95_s"], "stage3/learned_gate_eval_test.json", "aggregate.fixed_catalog_rescore__e2e_p95.mean", 3, "test", False, True, True)
    add("stage3_gate_f1_05", main_rows[2]["S_F1_0.5"], "stage3/learned_gate_eval_test.json", "aggregate.learned_gate__f1@0.5.mean", 3, "test", False, True, False)
    add("stage3_oracle_cand_f1_05", main_rows[4]["S_F1_0.5"], "stage3/oracle_selector_ceiling.json", "oracle_candidate_ceiling[K=5].oracle_f1@0.5", 3, "test", False, False, False)

    # bootstrap table
    boot_rows = []
    for k, d in boot_fx["deltas"].items():
        boot_rows.append({"comparison": "fixed_vs_phasenet", "metric": k, **d, "stage": 3, "role": "test"})
    for comp, block in boot_gate.get("comparisons", {}).items():
        for k, d in block.items():
            if isinstance(d, dict) and "mean" in d:
                boot_rows.append({"comparison": comp, "metric": k, **d, "stage": 3, "role": "test"})
    pd.DataFrame(boot_rows).to_csv(tables / "table_bootstrap_stage3.csv", index=False)
    d05 = boot_fx["deltas"]["delta_f1_0.5"]
    add("stage3_delta_f1_05_mean", d05["mean"], "stage3/bootstrap_fixed_vs_phasenet.json", "deltas.delta_f1_0.5.mean", 3, "test", False, True, True)
    add("stage3_delta_f1_05_ci", f"[{d05['ci95_low']:.3f},{d05['ci95_high']:.3f}]", "stage3/bootstrap_fixed_vs_phasenet.json", "deltas.delta_f1_0.5.ci95", 3, "test", False, True, True)

    # hyperparameters
    hyp = {
        "candidate_K": 5,
        "shrinkage_k": best["shrinkage_k"],
        "lambda_s_wave": best["best_s"]["lw"],
        "lambda_s_history": best["best_s"]["lh"],
        "lambda_s_prominence": best["best_s"]["lp"],
        "lambda_selected_on": "Stage-2 chronological validation (not Stage-3/4 test)",
        "phasenet_weight": "stead",
    }
    save_json(hyp, tables / "hyperparameters_frozen.json")
    add("shrinkage_k", 50, "stage2/best_lambdas.json", "shrinkage_k", 2, "development", True, False, False)
    add("lambda_s", "(0.5,2.0,0.0)", "stage2/best_lambdas.json", "best_s", 2, "development", True, False, False)

    # Stage 4 auxiliary
    s4main = pd.read_csv(ad / "results/stage4/tables/main_results.csv")
    s4main.to_csv(tables / "table_stage4_auxiliary.csv", index=False)
    add("stage4_n_events", s4a["n_events"], "stage4/holdout_split_audit.json", "n_events", 4, "auxiliary", False, False, True)
    add("stage4_underpowered", True, "stage4/stage4_final_verdict.json", "holdout_underpowered", 4, "auxiliary", False, False, True)

    # transfer
    tr = ad / "results/stage4/tables/transfer_weights_results.csv"
    if tr.exists():
        pd.read_csv(tr).to_csv(tables / "table_transfer_weights.csv", index=False)

    # ablation from stage3 metrics if present
    abl = []
    for method, prefix in [
        ("PhaseNet", "phasenet"),
        ("fixed catalog_rescore", "fixed_catalog_rescore"),
        ("learned gate", "learned_gate"),
        ("shuffled history", "learned_gate_shuffled_history"),
        ("biased history", "learned_gate_biased_history"),
        ("oracle selector", "oracle_selector"),
        ("oracle candidate", "oracle_candidate"),
    ]:
        k = f"{prefix}__f1@0.5"
        if k in agg:
            abl.append({"method": method, "S_F1_0.5": g(k), "S_e2e_P95": g(f"{prefix}__e2e_p95") if f"{prefix}__e2e_p95" in agg else np.nan})
    pd.DataFrame(abl).to_csv(tables / "table_ablation_stage3.csv", index=False)

    # split audit table
    pd.DataFrame(
        [
            {"set": "Stage-3 test-only", "n_traces": split3["test_only"]["n_traces"], "n_events": split3["test_only"]["n_events"], "role": "main_test", "pure_chronological_test": True},
            {"set": "Original fixed eval (Stage1.5/2)", "n_traces": split3["original_fixed_eval"]["n_traces"], "n_events": split3["original_fixed_eval"]["n_events"], "role": "development_mixed", "pure_chronological_test": False},
            {"set": "Stage-4 unused holdout", "n_traces": s4a["n_traces"], "n_events": s4a["n_events"], "role": "auxiliary_underpowered", "pure_chronological_test": True},
        ]
    ).to_csv(tables / "table_data_splits.csv", index=False)

    # compute cost if present
    cc = ad / "results/stage4/tables/compute_cost.csv"
    if cc.exists():
        pd.read_csv(cc).to_csv(tables / "table_compute_cost.csv", index=False)

    # Stage-4 main numbers for ledger (auxiliary)
    s4_stead = pd.read_csv(ad / "results/stage4/tables/main_results_stead.csv")
    for meth, key in [("phasenet", "stage4_pn_f1_05"), ("fixed_catalog_rescore", "stage4_fx_f1_05")]:
        row = s4_stead[s4_stead["method"] == meth]
        if len(row):
            add(key, float(row.iloc[0]["S_f1@0.5"]), "stage4/tables/main_results_stead.csv", f"{meth}.S_f1@0.5", 4, "auxiliary", False, meth != "phasenet", False)
            add(key.replace("f1_05", "p95"), float(row.iloc[0]["S_e2e_p95"]), "stage4/tables/main_results_stead.csv", f"{meth}.S_e2e_p95", 4, "auxiliary", False, meth != "phasenet", False)

    dp95 = boot_fx["deltas"]["delta_e2e_p95"]
    add("stage3_delta_p95_ci", f"[{dp95['ci95_low']:.3f},{dp95['ci95_high']:.3f}]", "stage3/bootstrap_fixed_vs_phasenet.json", "deltas.delta_e2e_p95.ci95", 3, "test", False, True, True)
    gate_block = boot_gate.get("comparisons", {}).get("learned_gate_vs_fixed_catalog_rescore", {})
    gate_d = gate_block.get("delta_f1_0.5") or gate_block.get("delta_f1@0.5")
    if isinstance(gate_d, dict) and "ci95_low" in gate_d:
        add("stage3_gate_vs_fixed_f1_05_ci", f"[{gate_d['ci95_low']:.4f},{gate_d['ci95_high']:.4f}]", "stage3/bootstrap_gate.json", "comparisons.learned_gate_vs_fixed_catalog_rescore", 3, "test", False, True, False)

    # --- Figures from Stage3 picks ---
    import shutil

    picks = pd.read_parquet(ad / "results/stage3/learned_gate_picks_test.parquet")
    picks = picks[picks.seed == 42].copy()
    sr = picks.sampling_rate_hz.to_numpy()
    true = picks.true_s_sample.to_numpy()
    e_pn = np.abs(picks.pred_s_phasenet - true) / sr
    e_fx = np.abs(picks.pred_s_fixed_rescore - true) / sr
    m = np.isfinite(e_pn) & np.isfinite(e_fx)

    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    for err, lab, c in [(e_pn[m], "PhaseNet", "C0"), (e_fx[m], "fixed re-ranking", "C1")]:
        x = np.sort(err)
        y = np.linspace(0, 1, len(x), endpoint=False)
        ax.plot(x, y, label=lab, color=c, lw=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("Absolute S error (s)")
    ax.set_ylabel("Empirical CDF")
    ax.legend(frameon=False)
    ax.set_title("Stage 3 test-only: S-phase error CDF")
    save_fig(fig, figs / "fig_error_cdf_stage3")

    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    methods = [
        ("PhaseNet", g("phasenet__f1@0.1"), g("phasenet__f1@0.5")),
        ("fixed", g("fixed_catalog_rescore__f1@0.1"), g("fixed_catalog_rescore__f1@0.5")),
        ("learned gate", g("learned_gate__f1@0.1"), g("learned_gate__f1@0.5")),
    ]
    for name, f01, f05 in methods:
        ax.plot([0.1, 0.5], [f01, f05], "o-", label=name)
    ax.set_xlabel("Tolerance (s)")
    ax.set_ylabel("S F1")
    ax.legend(frameon=False)
    ax.set_title("Stage 3: S F1 vs tolerance")
    save_fig(fig, figs / "fig_f1_tolerance_stage3")

    # method diagram
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.axis("off")
    labels = ["3C waveform", "PhaseNet\nSTEAD", "Top-$K{=}5$\nS peaks", "Distance MLP\n+ path residual", "Fixed score\nre-ranking", "S pick"]
    xs = np.linspace(0.02, 0.86, len(labels))
    for x, t in zip(xs, labels):
        ax.add_patch(plt.Rectangle((x, 0.3), 0.12, 0.4, fill=False, lw=1.2))
        ax.text(x + 0.06, 0.5, t, ha="center", va="center", fontsize=8)
    for i in range(len(xs) - 1):
        ax.annotate("", xy=(xs[i + 1], 0.5), xytext=(xs[i] + 0.12, 0.5), arrowprops=dict(arrowstyle="->", lw=1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Catalog-assisted S-phase candidate re-ranking", fontsize=11)
    save_fig(fig, figs / "fig_method_flowchart")

    # Development hard-subset ΔF1 (Stage 2 mixed fixed-eval; NOT main test)
    hs_json = ad / "results/stage2/hard_cases_summary.json"
    if hs_json.exists():
        hs = loadj(hs_json)
        want = ["s_multi_peak", "s_prob_<0.3", "mad_<0.2", "hist_>=20"]
        rows = []
        for block in hs.get("helps_most", []) + hs.get("hurts_most", []):
            if block["subset"] in want:
                rows.append(block)
        # de-duplicate by subset preserving order of want
        by = {r["subset"]: r for r in rows}
        rows = [by[s] for s in want if s in by]
        if rows:
            fig, ax = plt.subplots(figsize=(5.0, 3.2))
            names = [r["subset"] for r in rows]
            deltas = [float(r["delta_s_f1@0.5"]) for r in rows]
            ns = [int(r["n"]) for r in rows]
            ax.bar(range(len(names)), deltas, color="#4C72B0")
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels([f"{n}\n(n={nn})" for n, nn in zip(names, ns)], fontsize=8)
            ax.set_ylabel(r"$\Delta$ S F1@0.5 (fixed $-$ PhaseNet)")
            ax.set_title("Stage 2 development hard subsets (not main test)")
            ax.axhline(0, color="k", lw=0.8)
            save_fig(fig, figs / "fig_subset_gains_stage2_dev")
            pd.DataFrame(rows).to_csv(tables / "table_subset_stage2_dev.csv", index=False)
            add("stage2_subset_multi_peak_delta_f1", float(by["s_multi_peak"]["delta_s_f1@0.5"]) if "s_multi_peak" in by else None, "stage2/hard_cases_summary.json", "helps_most.s_multi_peak.delta_s_f1@0.5", 2, "development", False, False, False)

    # Stage-4 hard subsets (auxiliary / underpowered)
    hs4 = ad / "results/stage4/tables/hard_subset_results.csv"
    if hs4.exists():
        tab4 = pd.read_csv(hs4)
        keep = tab4[tab4["feature"].isin(["s_n_cands", "s_peak_probability", "residual_s_mad", "history_count"])].copy() if "feature" in tab4.columns else tab4
        if len(keep):
            fig, ax = plt.subplots(figsize=(5.2, 3.2))
            labels = [f"{r.feature}:{r.bin}" for _, r in keep.iterrows()]
            ax.barh(range(len(keep)), keep["delta_f1_0.5"].to_numpy(), color="#55A868")
            ax.set_yticks(range(len(keep)))
            ax.set_yticklabels(labels, fontsize=7)
            ax.set_xlabel(r"$\Delta$ F1@0.5")
            ax.set_title("Stage 4 hard subsets (9 events; underpowered)")
            ax.axvline(0, color="k", lw=0.8)
            save_fig(fig, figs / "fig_subset_gains_stage4_aux")
            keep.to_csv(tables / "table_subset_stage4_aux.csv", index=False)

    # Qualitative cases from frozen Stage-3 gate_cases (no re-inference)
    cases = ad / "results/stage3/gate_cases"
    case_map = {
        "fig_case_success_wrongpeak_fix": "pn_wrong_gate_fix.png",
        "fig_case_success_fixed_ok": "gate_wrong_fixed_ok.png",
        "fig_case_failure_fixed_wrong": "fixed_wrong_gate_ok.png",
        "fig_case_failure_no_candidate": "no_correct_candidate.png",
    }
    for out_stem, src_name in case_map.items():
        src = cases / src_name
        if not src.exists():
            continue
        shutil.copy(src, figs / f"{out_stem}.png")
        # wrap PNG into a PDF page for LaTeX graphicspath consistency
        img = plt.imread(src)
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ax.imshow(img)
        ax.axis("off")
        fig.savefig(figs / f"{out_stem}.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # copy stage4 figures if useful
    for name in ["error_cdf_s", "f1_vs_tolerance", "bootstrap_deltas", "gain_by_multipeak", "gain_by_history_mad", "gain_by_phasenet_prob"]:
        src = ad / "results/stage4/figures" / f"{name}.pdf"
        if src.exists():
            shutil.copy(src, figs / f"fig_stage4_{name}.pdf")
            png = src.with_suffix(".png")
            if png.exists():
                shutil.copy(png, figs / f"fig_stage4_{name}.png")

    # bootstrap LaTeX snippet
    with open(tables / "table_bootstrap_stage3.tex", "w") as f:
        f.write("\\begin{tabular}{lrrr}\n\\toprule\nMetric & mean $\\Delta$ & 95\\% CI low & 95\\% CI high \\\\\n\\midrule\n")
        for metric, label in [
            ("delta_f1_0.5", "F1@0.5"),
            ("delta_f1_0.1", "F1@0.1"),
            ("delta_e2e_p95", "e2e P95 (s)"),
            ("delta_e2e_mae", "e2e MAE (s)"),
            ("delta_wrong_peak_rate", "wrong-peak rate"),
        ]:
            d = boot_fx["deltas"][metric]
            f.write(f"{label} & {d['mean']:.4f} & {d['ci95_low']:.4f} & {d['ci95_high']:.4f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    # evidence ledger
    led = pd.DataFrame(ledger)
    led.to_csv(tables / "evidence_ledger.csv", index=False)
    lines = [
        "# Paper evidence ledger\n",
        "Auto-generated by `scripts/build_paper_assets.py` from frozen artifacts (no re-training).\n\n",
        "| name | value | source | key | stage | role | tuning? | CI? | abstract? |\n",
        "|---|---|---|---|---|---|---|---|---|\n",
    ]
    for r in ledger:
        lines.append(
            f"| {r['name']} | `{r['value']}` | `{r['source']}` | `{r['json_key']}` | {r['stage']} | {r['role']} | {r['used_for_tuning']} | {r['has_event_ci']} | {r['allow_in_abstract']} |\n"
        )
    lines.append("\n## Evidence policy\n")
    lines.append("- **Main test:** Stage 3 test-only 10k (event-disjoint).\n")
    lines.append("- **Development:** Stage 2 mixed fixed eval (λ / shrinkage / hard subsets).\n")
    lines.append("- **Auxiliary:** Stage 4 unused 9-event holdout (direction only; underpowered).\n")
    lines.append("- Numbers in paper tables/figures must be regenerated via this script; do not hand-edit CSV constants.\n")
    (reports / "paper_evidence_ledger.md").write_text("".join(lines))

    # software environment snapshot
    import platform
    import subprocess

    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    try:
        env["matplotlib"] = __import__("matplotlib").__version__
    except Exception:
        pass
    try:
        env["conda_env"] = subprocess.check_output(["conda", "info", "--envs"], text=True)
    except Exception:
        env["conda_env"] = "unavailable"
    save_json(env, paper / "environment_snapshot.json")

    meta = {
        "title": "Historical Path-Residual Guided Candidate Re-ranking for Catalog-Assisted S-Phase Repicking",
        "alt_titles": [
            "Catalog-Assisted Re-ranking of PhaseNet S-Wave Candidates Using Historical Path Residuals",
            "Reducing S-Phase Wrong-Peak Errors with Shrinkage-Regularized Path Residuals",
        ],
        "n_ledger_entries": len(ledger),
        "stage3_fixed_vs_pn_significant_f1_05": bool(boot_fx["deltas"]["delta_f1_0.5"]["significant"]),
        "stage4_underpowered": bool(s4v["holdout_underpowered"]),
        "main_method": "fixed_catalog_rescore",
        "learned_gate_is_main": False,
        "continue_gnn": False,
        "sota_claim_allowed": False,
    }
    save_json(meta, paper / "paper_meta.json")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
