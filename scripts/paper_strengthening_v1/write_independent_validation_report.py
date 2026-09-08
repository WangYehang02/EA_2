#!/usr/bin/env python
"""Write independent-period validation report + update paper judgment from frozen results."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path("/data/yehang/Earthquake_paper_strengthening_v1/independent_period")
REPORT = ROOT / "reports" / "paper_strengthening_v1"


def main() -> int:
    summary = json.loads((OUT / "results" / "EVAL_SUMMARY.json").read_text())
    comps = json.loads((OUT / "results" / "comparisons.json").read_text())
    metrics = pd.read_csv(OUT / "results" / "metrics_A_to_E.csv")
    dl = json.loads((OUT / "download" / "download_summary.json").read_text())
    sample = json.loads((OUT / "locks" / "SAMPLE.LOCK.json").read_text())
    ready = json.loads((OUT / "READY.json").read_text())
    indep = json.loads((OUT / "audits" / "independence_audit.json").read_text())

    prim = comps["primary_scalar_minus_C"]
    sec = comps["secondary_C_minus_fixed"]
    d_f1 = prim["delta"]
    ci = prim["bootstrap"]["ci95"]
    d_mae = prim["tail_secondary"]["detected_ae_mae"]["delta"]
    d_gt5 = prim["tail_secondary"]["frac_err_gt_5s"]["delta"]
    d_cf = sec["delta"]
    ci_cf = sec["bootstrap"]["ci95"]

    # Interpret without auto-equivalence / auto-practicality
    f1_retained = d_f1 > 0 and ci[0] > 0
    f1_positive_point = d_f1 > 0
    mae_better = d_mae < 0  # lower MAE better
    gt5_better = d_gt5 < 0
    c_beats_fixed = d_cf > 0 and ci_cf[0] > 0

    practical_note = (
        "Practical F1 discussion scale remains |ΔF1@0.5|≈0.003; "
        "do not claim material F1 gain from a barely-positive CI alone."
    )

    lines = []
    lines.append("# Independent-period validation report (BSI 2021–2022)")
    lines.append("")
    lines.append(f"- Generated (UTC): `{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}`")
    lines.append(f"- Run status: **{summary.get('run_status')}** (execution complete ≠ positive result)")
    lines.append(f"- Validation label: independent period, same region — **not** cross-region generalization")
    lines.append("")
    lines.append("## Sample & download")
    lines.append(f"- Selected events/records (locked): {sample['n_selected_events']} / {sample['n_selected_records']}")
    lines.append(f"- Download requested: {dl.get('n_requested')}; ok: {dl.get('n_ok_reason')}; "
                 f"label-outside-window: {dl.get('n_label_outside')}; other-fail: {dl.get('n_failed_other')}")
    lines.append(f"- Formal pairs (READY): n={ready['n_pairs']}, events={ready['n_events']}")
    lines.append(f"- Independence: event-id overlaps vs ranker_train/phaseB/pairs = "
                 f"{indep['event_id_overlaps']}")
    lines.append("")
    lines.append("## A–E absolute metrics (shared QC-passed pairs)")
    lines.append("")
    # avoid tabulate dependency
    show_cols = [
        "method",
        "f1@0.5",
        "f1@0.1",
        "detected_ae_mae",
        "detected_ae_median",
        "detected_ae_p95",
        "frac_err_gt_5s",
        "mech_fixes",
        "mech_breaks",
        "mech_net_fixes",
        "n",
    ]
    cols = [c for c in show_cols if c in metrics.columns]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, r in metrics.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append(f"{v:.6g}" if abs(v) < 1e3 else f"{v:.4g}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Pre-registered comparisons")
    lines.append("")
    lines.append("### Primary: scalar − C (F1@0.5)")
    lines.append(f"- ΔF1@0.5 = **{d_f1:.6f}** (E={prim['E_f1@0.5']:.6f}, C={prim['C_f1@0.5']:.6f})")
    lines.append(f"- Paired event bootstrap 5000: mean={prim['bootstrap']['mean']:.6f}, "
                 f"95% CI=[{ci[0]:.6f}, {ci[1]:.6f}]")
    lines.append(f"- Tail secondary ΔMAE (E−C) = {d_mae:.6f}; Δfrac_err_gt_5s (E−C) = {d_gt5:.6f}")
    lines.append("- Tail metrics **must not** replace the primary F1 endpoint.")
    lines.append("")
    lines.append("### Secondary (preregistered): C − fixed (F1@0.5)")
    lines.append(f"- ΔF1@0.5 = **{d_cf:.6f}** (C={sec['C_f1@0.5']:.6f}, A={sec['A_f1@0.5']:.6f})")
    lines.append(f"- Bootstrap 95% CI=[{ci_cf[0]:.6f}, {ci_cf[1]:.6f}]")
    lines.append("")
    lines.append("## Answers to the three questions")
    lines.append(f"1. Does scalar F1 gain vs C retain? point={f1_positive_point}, "
                 f"CI_excludes_zero_positive={f1_retained}. {practical_note}")
    lines.append(f"2. Do scalar MAE / >5s tail benefits retain? mae_better={mae_better}, gt5_better={gt5_better} "
                 f"(lower is better for both).")
    lines.append(f"3. Does C retain gain vs fixed? {c_beats_fixed} (Δ={d_cf:.6f}, CI={ci_cf}).")
    lines.append("")
    lines.append("## Artifact paths")
    lines.append(f"- Data root: `{OUT}`")
    lines.append(f"- READY: `{OUT/'READY.json'}`")
    lines.append(f"- Predictions lock: `{OUT/'locks'/'PREDICTIONS.LOCK.json'}`")
    lines.append(f"- Metrics: `{OUT/'results'/'metrics_A_to_E.csv'}`")
    lines.append(f"- Comparisons: `{OUT/'results'/'comparisons.json'}`")
    lines.append(f"- Logs: `{OUT/'logs'}`")
    text = "\n".join(lines) + "\n"
    (REPORT / "INDEPENDENT_PERIOD_VALIDATION_REPORT.md").write_text(text)
    (OUT / "results" / "INDEPENDENT_PERIOD_VALIDATION_REPORT.md").write_text(text)

    # Updated paper judgment
    judg = []
    judg.append("# Updated paper judgment (after independent-period validation)")
    judg.append("")
    judg.append("## Execution")
    judg.append(f"- Independent-period formal eval: **{summary.get('run_status')}**")
    judg.append(f"- Primary ΔF1@0.5 (scalar−C) = {d_f1:.6f}, 95% CI [{ci[0]:.6f}, {ci[1]:.6f}]")
    judg.append(f"- Secondary ΔF1@0.5 (C−fixed) = {d_cf:.6f}, 95% CI [{ci_cf[0]:.6f}, {ci_cf[1]:.6f}]")
    judg.append(f"- Tail ΔMAE (scalar−C) = {d_mae:.6f}; Δfrac>5s = {d_gt5:.6f}")
    judg.append("")
    judg.append("## Paper-facing conclusions")
    if c_beats_fixed or d_cf > 0:
        judg.append("- **Simple prior (C / base_τ ranking) remains a primary effective method** relative to fixed on this new period "
                    f"(ΔF1={d_cf:.4f}). Treat simple catalog-assisted prior + candidate ranking as the main story backbone.")
    else:
        judg.append("- On this independent period, C did **not** clearly beat fixed on F1@0.5; do not oversell prior strength here. "
                    "Historical full-dev still showed C≈scalar; report both.")
    if f1_retained and abs(d_f1) >= 0.003:
        judg.append("- Scalar shows a **repeatable F1@0.5 gain** vs C on the new period at practical scale.")
    elif f1_positive_point and abs(d_f1) < 0.003:
        judg.append("- Scalar vs C F1@0.5 increment is **tiny / not practically material** on the new period "
                    f"(Δ≈{d_f1:.4f}); do not center the paper on F1 gain over C.")
    else:
        judg.append("- Scalar F1@0.5 advantage vs C **does not clearly retain** on the new period; "
                    "do not claim reproducible F1 superiority over the strong prior control.")
    if mae_better or gt5_better:
        judg.append("- Scalar may still hold **tail refinement value** (MAE / >5s) even if F1 gain vs C is negligible; "
                    "state this as secondary, pre-registered tail evidence — not a post-hoc primary swap.")
    else:
        judg.append("- Scalar MAE / >5s tail benefits vs C **did not clearly retain** here; "
                    "do not claim robust tail refinement without this support.")
    judg.append("")
    judg.append("## Still not claimed")
    judg.append("- Cross-region generalization")
    judg.append("- Blind / end-to-end picking superiority")
    judg.append("- Equivalence solely from CI crossing zero")
    judg.append("")
    judg.append(f"Full report: `reports/paper_strengthening_v1/INDEPENDENT_PERIOD_VALIDATION_REPORT.md`")
    (REPORT / "UPDATED_PAPER_JUDGMENT.md").write_text("\n".join(judg) + "\n")

    # Replace NOT_COMPLETED marker
    (REPORT / "INDEPENDENT_VALIDATION_STATUS.md").write_text(
        json.dumps(
            {
                "status": "COMPLETED",
                "ready": True,
                "formal_eval": True,
                "report": "INDEPENDENT_PERIOD_VALIDATION_REPORT.md",
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            indent=2,
        )
        + "\n"
    )
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
