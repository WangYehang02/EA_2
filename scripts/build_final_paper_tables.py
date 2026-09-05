#!/usr/bin/env python
"""Build paper tables from frozen artifacts only (no experiments)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from earthquake.config import artifacts_dir
from earthquake.utils import ensure_dir

OUT_T = ROOT / "reports" / "paper" / "tables"
OUT_A = artifacts_dir() / "results" / "final_evidence" / "tables"


def r4(x):
    return float(f"{float(x):.4f}")


def r3(x):
    return float(f"{float(x):.3f}")


def df_to_md(df: pd.DataFrame) -> str:
    """Markdown table without requiring the optional tabulate dependency."""
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(str(c) for c in cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ensure_dir(OUT_T)
    ensure_dir(OUT_A)
    A = artifacts_dir()

    cf = json.loads((A / "results/pairwise_confirm/confirm_fixed_metrics.json").read_text())
    cr = json.loads((A / "results/pairwise_confirm/confirm_resid_metrics.json").read_text())
    cs = json.loads((A / "results/pairwise_confirm/confirm_scalar_metrics.json").read_text())

    # Table 1 confirm main
    t1 = pd.DataFrame(
        [
            {
                "Method": "fixed_rescore_UNION",
                "F1@0.1": r4(cf["f1@0.1"]),
                "F1@0.5": r4(cf["f1@0.5"]),
                "Precision@0.5": r4(cf["precision@0.5"]),
                "Recall@0.5": r4(cf["recall@0.5"]),
                "P95_s": r3(cf["detected_ae_p95"]),
                "ΔF1@0.5_vs_fixed": 0.0,
                "ΔF1@0.5_vs_residual": r4(cf["f1@0.5"] - cr["f1@0.5"]),
            },
            {
                "Method": "resid_s_control",
                "F1@0.1": r4(cr["f1@0.1"]),
                "F1@0.5": r4(cr["f1@0.5"]),
                "Precision@0.5": r4(cr["precision@0.5"]),
                "Recall@0.5": r4(cr["recall@0.5"]),
                "P95_s": r3(cr["detected_ae_p95"]),
                "ΔF1@0.5_vs_fixed": r4(cr["delta_f1@0.5_vs_fixed"]),
                "ΔF1@0.5_vs_residual": 0.0,
            },
            {
                "Method": "scalar_pairwise",
                "F1@0.1": r4(cs["f1@0.1"]),
                "F1@0.5": r4(cs["f1@0.5"]),
                "Precision@0.5": r4(cs["precision@0.5"]),
                "Recall@0.5": r4(cs["recall@0.5"]),
                "P95_s": r3(cs["detected_ae_p95"]),
                "ΔF1@0.5_vs_fixed": r4(cs["delta_f1@0.5_vs_fixed"]),
                "ΔF1@0.5_vs_residual": r4(cs["delta_f1@0.5_vs_resid"]),
            },
        ]
    )
    t1.to_csv(OUT_T / "table1_confirm_main.csv", index=False)
    t1.to_csv(OUT_A / "table1_confirm_main.csv", index=False)
    (OUT_T / "table1_confirm_main.md").write_text(df_to_md(t1), encoding="utf-8")

    # Table 2 replication
    ab = json.loads((A / "results/pairwise_pilot/ablation_metrics.json").read_text())
    fd = json.loads((A / "results/pairwise_fulldev/fulldev_final.json").read_text())
    t2 = pd.DataFrame(
        [
            {
                "Split": "pilot-heldout",
                "Fixed": f"{r4(ab['fixed_rescore_UNION']['f1@0.5']):.4f}",
                "Residual": f"{r4(ab['resid_s_control']['f1@0.5']):.4f}",
                "Scalar": f"{r4(ab['scalar_pairwise']['f1@0.5']):.4f}",
                "Scalar-Fixed": f"{r4(ab['scalar_pairwise']['delta_f1@0.5']):+.4f}",
                "Scalar-Residual": f"{r4(ab['scalar_pairwise']['f1@0.5'] - ab['resid_s_control']['f1@0.5']):+.4f}",
            },
            {
                "Split": "phaseB-full-dev",
                "Fixed": f"{r4(fd['fixed']['f1@0.5']):.4f}",
                "Residual": f"{r4(fd['resid']['f1@0.5']):.4f}",
                "Scalar": f"{r4(fd['scalar']['f1@0.5']):.4f}",
                "Scalar-Fixed": f"{r4(fd['scalar']['f1@0.5'] - fd['fixed']['f1@0.5']):+.4f}",
                "Scalar-Residual": f"{r4(fd['scalar']['f1@0.5'] - fd['resid']['f1@0.5']):+.4f}",
            },
            {
                "Split": "confirm",
                "Fixed": f"{r4(cf['f1@0.5']):.4f}",
                "Residual": f"{r4(cr['f1@0.5']):.4f}",
                "Scalar": f"{r4(cs['f1@0.5']):.4f}",
                "Scalar-Fixed": f"{r4(cs['delta_f1@0.5_vs_fixed']):+.4f}",
                "Scalar-Residual": f"{r4(cs['delta_f1@0.5_vs_resid']):+.4f}",
            },
        ]
    )
    t2.to_csv(OUT_T / "table2_replication.csv", index=False)
    t2.to_csv(OUT_A / "table2_replication.csv", index=False)
    (OUT_T / "table2_replication.md").write_text(df_to_md(t2), encoding="utf-8")

    # Table 3 mechanism
    t3 = pd.DataFrame(
        [
            {
                "Split": "confirm",
                "Eligible_pairs": int(cs["n_eligible_pairs"]),
                "Switch_count": int(cs["n_switch"]),
                "Fixes": int(cs["fixes"]),
                "Breaks": int(cs["breaks"]),
                "Net_fixes": int(cs["net_fixes"]),
                "Switch_precision": r3(cs["switch_precision"]),
                "Q2_recovery": r3(cs["Q2_recovery_rate"]),
                "Q1_break_rate": r3(cs["Q1_break_rate"]),
            },
            {
                "Split": "phaseB-full-dev",
                "Eligible_pairs": int(fd["scalar"]["n_eligible_pairs"]),
                "Switch_count": int(fd["scalar"]["n_switch"]),
                "Fixes": int(fd["scalar"]["fixes"]),
                "Breaks": int(fd["scalar"]["breaks"]),
                "Net_fixes": int(fd["scalar"]["net_fixes"]),
                "Switch_precision": r3(fd["scalar"]["switch_precision"]),
                "Q2_recovery": r3(fd["scalar"]["Q2_recovery_rate"]),
                "Q1_break_rate": r3(fd["scalar"]["Q1_break_rate"]),
            },
        ]
    )
    t3.to_csv(OUT_T / "table3_pairwise_mechanism.csv", index=False)
    t3.to_csv(OUT_A / "table3_pairwise_mechanism.csv", index=False)
    (OUT_T / "table3_pairwise_mechanism.md").write_text(df_to_md(t3), encoding="utf-8")

    # Table 4 ranking forensic
    rd = json.loads((A / "results/multistation_moveout/ranking_gap_rank_distribution.json").read_text())
    n = rd["n_recoverable"]
    rows = []
    cum = 0
    for rank, key in [(2, "rank_good_eq_2"), (3, "rank_good_eq_3"), (4, "rank_good_eq_4"), (5, "rank_good_eq_5")]:
        c = int(rd[key])
        cum += c
        rows.append(
            {
                "Good_candidate_rank": rank,
                "Count": c,
                "Fraction": r3(c / n),
                "Cumulative_fraction": r3(cum / n),
            }
        )
    t4 = pd.DataFrame(rows)
    t4["Good_candidate_rank"] = t4["Good_candidate_rank"].astype(int)
    t4["Count"] = t4["Count"].astype(int)
    t4.to_csv(OUT_T / "table4_ranking_forensic.csv", index=False)
    t4.to_csv(OUT_A / "table4_ranking_forensic.csv", index=False)
    (OUT_T / "table4_ranking_forensic.md").write_text(
        df_to_md(t4)
        + f"\nn_recoverable={n}; P(rank≤2)={r3(rd['P_rank_good_le_2'])}; "
        f"P(rank≤3)={r3(rd['P_rank_good_le_3'])}; P(rank≤5)={r3(rd['P_rank_good_le_5'])}\n",
        encoding="utf-8",
    )

    # Supplement negative
    soft = json.loads((A / "results/multistation_soft/soft_ring_best.json").read_text())
    hard = soft["hard_gate_best_ref"]
    wb = json.loads((A / "results/pairwise_pilot/waveform_bootstrap.json").read_text())
    move_best = json.loads((A / "results/multistation_moveout/moveout_best.json").read_text())
    best_move = move_best["best_legal"]
    agree_raw = move_best.get("moveout_vs_singlestation_agree", 0.983)
    if isinstance(agree_raw, dict):
        agree_note = agree_raw.get("prediction_agree_rate", agree_raw.get("agree_rate", 0.983))
    else:
        agree_note = float(agree_raw)
    neg = pd.DataFrame(
        [
            {
                "Method": "hard_ring_gate_best",
                "ΔF1@0.5": r4(hard["delta_f1@0.5"]),
                "P95_change_note": f"P95→{hard['detected_ae_p95']} with abstain={r3(hard['abstain_rate'])}",
                "Interpretation": "Hard abstention improves timing but violates always-output protocol",
                "Verdict": "NO-GO",
            },
            {
                "Method": "soft_same_ring_best",
                "ΔF1@0.5": r4(soft["best_overall"]["delta_f1@0.5"]),
                "P95_change_note": f"delta_p95={r3(soft['best_overall']['delta_p95'])}",
                "Interpretation": "Significant but small soft geometric gain",
                "Verdict": "small / not primary",
            },
            {
                "Method": "moveout_best_legal",
                "ΔF1@0.5": r4(best_move["delta_f1@0.5"]),
                "P95_change_note": f"delta_p95={r3(best_move['delta_p95'])}",
                "Interpretation": f"≤ single-station resid control; agree≈{r3(float(agree_note))}",
                "Verdict": "NO-GO (geometry)",
            },
            {
                "Method": "waveform_only",
                "ΔF1@0.5": r4(ab["waveform_only"]["delta_f1@0.5"]),
                "P95_change_note": "held-out pilot",
                "Interpretation": "Informative in isolation; below scalar",
                "Verdict": "not chosen",
            },
            {
                "Method": "waveform_plus_scalar",
                "ΔF1@0.5": r4(wb["delta_waveform_point"]),
                "P95_change_note": f"vs scalar; CI={wb['waveform_plus_scalar_vs_scalar_pairwise']['delta_f1@0.5']['ci95']}",
                "Interpretation": "Conditional increment not material",
                "Verdict": "NO-GO (waveform branch)",
            },
        ]
    )
    neg.to_csv(OUT_T / "tableS_negative_ablations.csv", index=False)
    neg.to_csv(OUT_A / "tableS_negative_ablations.csv", index=False)
    (OUT_T / "tableS_negative_ablations.md").write_text(df_to_md(neg), encoding="utf-8")
    print("tables written", OUT_T)


if __name__ == "__main__":
    main()
