#!/usr/bin/env python
"""Build paper figures from frozen artifacts only (no experiments)."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))
from earthquake.config import artifacts_dir
from earthquake.utils import ensure_dir

OUT_A = artifacts_dir() / "results" / "final_evidence" / "figures"
OUT_R = ROOT / "reports" / "paper" / "figures"
SCRIPT = "scripts/build_final_paper_figures.py"


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_source(stem: str, sources: list[dict], notes: str = "") -> None:
    meta = {
        "figure": stem,
        "script": SCRIPT,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sources": sources,
        "notes": notes,
    }
    for out in (OUT_A, OUT_R):
        (out / f"{stem}.source.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def save_fig(stem: str) -> None:
    for out in (OUT_A, OUT_R):
        ensure_dir(out)
        plt.savefig(out / f"{stem}.png", dpi=300, bbox_inches="tight")
        plt.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    plt.close()


def dump_csv(stem: str, df: pd.DataFrame) -> None:
    for out in (OUT_A, OUT_R):
        ensure_dir(out)
        df.to_csv(out / f"{stem}.csv", index=False)


def main() -> None:
    ensure_dir(OUT_A)
    ensure_dir(OUT_R)
    A = artifacts_dir()

    # ----- Figure 1 method flow (markdown structure only) -----
    flow = """# Figure 1 — Method flow (select, not generate)

## Nodes

1. **Waveform** — three-component seismic trace for one station-event pair.
2. **Base pickers** — frozen PhaseNet / EQTransformer (or equivalent) probability streams.
3. **UNION candidates** — merged S-phase candidate set from base pickers (frozen generator).
4. **fixed_score** — deterministic rescoring with `lw=0.5`, `lh=2`, `lp=0` using catalog-assisted expected arrival.
5. **c1 / c2** — highest and second-highest `fixed_score` candidates (tie → `candidate_index`).
6. **scalar_pairwise** — learned comparison of scalar features of (c1, c2) with frozen τ=0.50.
7. **Final candidate** — select c1 or c2 only; never generate a new arrival time.

## Decision rule

- If `<2` candidates: output c1 (no pairwise decision).
- Else: if `P(prefer c2) > τ` switch to c2; else keep c1.
- Forbidden: abstain, rank≥3 selection, new pick generation, arrival-time edit.

## Emphasis

The method **selects** among frozen candidates; it does **not** generate candidates.
"""
    for out in (OUT_A, OUT_R, ROOT / "reports" / "paper"):
        ensure_dir(out)
        (out / "figure1_method_flow.md").write_text(flow, encoding="utf-8")
    write_source(
        "figure1_method_flow",
        [{"artifact": "SCALAR_PAIRWISE.FULLDEV_METHOD_LOCK.json", "role": "method definition"}],
        notes="Structural figure description only; not a network architecture diagram.",
    )

    # ----- Figure 2 ranking forensic -----
    rd_path = A / "results/multistation_moveout/ranking_gap_rank_distribution.json"
    rd = json.loads(rd_path.read_text())
    n = rd["n_recoverable"]
    ranks = [2, 3, 4, 5]
    counts = [rd[f"rank_good_eq_{r}"] for r in ranks]
    fracs = [c / n for c in counts]
    cum = []
    s = 0
    for c in counts:
        s += c
        cum.append(s / n)
    df2 = pd.DataFrame(
        {
            "rank_good": ranks,
            "count": counts,
            "fraction": fracs,
            "cumulative_fraction": cum,
            "n_recoverable": n,
            "P_le2": rd["P_rank_good_le_2"],
            "P_le3": rd["P_rank_good_le_3"],
            "P_le5": rd["P_rank_good_le_5"],
        }
    )
    dump_csv("fig2_rank_distribution", df2)
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.bar([str(r) for r in ranks], counts, color="0.35", edgecolor="black", linewidth=0.6)
    ax.set_xlabel("Rank of correct S candidate (recoverable cases)")
    ax.set_ylabel("Count")
    ax.set_title("Ranking forensic (n_recoverable=%d)" % n)
    txt = "cum@2=%.3f  cum@3=%.3f  cum@5=%.3f" % (rd["P_rank_good_le_2"], rd["P_rank_good_le_3"], rd["P_rank_good_le_5"])
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", fontsize=9)
    save_fig("fig2_rank_distribution")
    write_source(
        "fig2_rank_distribution",
        [{"artifact": str(rd_path), "sha256": sha_file(rd_path)}],
        notes="Among ranking-recoverable errors only; not all S errors.",
    )

    # ----- Figure 3 replication -----
    ab = json.loads((A / "results/pairwise_pilot/ablation_metrics.json").read_text())
    fd = json.loads((A / "results/pairwise_fulldev/fulldev_final.json").read_text())
    cf = json.loads((A / "results/pairwise_confirm/confirm_fixed_metrics.json").read_text())
    cr = json.loads((A / "results/pairwise_confirm/confirm_resid_metrics.json").read_text())
    cs = json.loads((A / "results/pairwise_confirm/confirm_scalar_metrics.json").read_text())
    b_pilot = json.loads((A / "results/pairwise_pilot/bootstrap.json").read_text())
    bsf = json.loads((A / "results/pairwise_confirm/confirm_bootstrap_scalar_vs_fixed.json").read_text())
    bsr = json.loads((A / "results/pairwise_confirm/confirm_bootstrap_scalar_vs_resid.json").read_text())

    rows = []
    # held-out: scalar-fixed has bootstrap; scalar-resid point-only
    sf_p = b_pilot["scalar_pairwise_vs_fixed"]["delta_f1@0.5"]
    rows.append(
        {
            "split": "held-out",
            "contrast": "scalar-fixed",
            "delta_f1@0.5": ab["scalar_pairwise"]["delta_f1@0.5"],
            "ci_lo": sf_p["ci95"][0],
            "ci_hi": sf_p["ci95"][1],
            "ci_available": True,
        }
    )
    rows.append(
        {
            "split": "held-out",
            "contrast": "scalar-resid",
            "delta_f1@0.5": ab["scalar_pairwise"]["f1@0.5"] - ab["resid_s_control"]["f1@0.5"],
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "ci_available": False,
        }
    )
    rows.append(
        {
            "split": "full-dev",
            "contrast": "scalar-fixed",
            "delta_f1@0.5": fd["scalar"]["f1@0.5"] - fd["fixed"]["f1@0.5"],
            "ci_lo": fd["bootstrap_vs_fixed"]["delta_f1@0.5"]["ci95"][0],
            "ci_hi": fd["bootstrap_vs_fixed"]["delta_f1@0.5"]["ci95"][1],
            "ci_available": True,
        }
    )
    rows.append(
        {
            "split": "full-dev",
            "contrast": "scalar-resid",
            "delta_f1@0.5": fd["scalar"]["f1@0.5"] - fd["resid"]["f1@0.5"],
            "ci_lo": fd["bootstrap_vs_resid"]["delta_f1@0.5"]["ci95"][0],
            "ci_hi": fd["bootstrap_vs_resid"]["delta_f1@0.5"]["ci95"][1],
            "ci_available": True,
        }
    )
    rows.append(
        {
            "split": "confirm",
            "contrast": "scalar-fixed",
            "delta_f1@0.5": cs["delta_f1@0.5_vs_fixed"],
            "ci_lo": bsf["delta_f1@0.5"]["ci95"][0],
            "ci_hi": bsf["delta_f1@0.5"]["ci95"][1],
            "ci_available": True,
        }
    )
    rows.append(
        {
            "split": "confirm",
            "contrast": "scalar-resid",
            "delta_f1@0.5": cs["delta_f1@0.5_vs_resid"],
            "ci_lo": bsr["delta_f1@0.5"]["ci95"][0],
            "ci_hi": bsr["delta_f1@0.5"]["ci95"][1],
            "ci_available": True,
        }
    )
    df3 = pd.DataFrame(rows)
    dump_csv("fig3_replication", df3)

    order = ["held-out", "full-dev", "confirm"]
    x = list(range(len(order)))
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for contrast, marker, offset in [("scalar-fixed", "o", -0.08), ("scalar-resid", "s", 0.08)]:
        sub = df3[df3["contrast"] == contrast].set_index("split").loc[order]
        xs = [xi + offset for xi in x]
        y = sub["delta_f1@0.5"].values
        ax.plot(xs, y, marker=marker, linestyle="-", label=contrast, color="0.2" if contrast.endswith("fixed") else "0.55")
        for xi, yi, lo, hi, ok in zip(xs, y, sub["ci_lo"], sub["ci_hi"], sub["ci_available"]):
            if bool(ok) and pd.notna(lo) and pd.notna(hi):
                ax.plot([xi, xi], [lo, hi], color="0.2" if contrast.endswith("fixed") else "0.55", linewidth=1.2)
    ax.axhline(0.0, color="0.7", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("ΔF1@0.5")
    ax.set_xlabel("Evaluation split")
    ax.set_title("Effect replication across splits")
    ax.legend(frameon=False)
    save_fig("fig3_replication")
    write_source(
        "fig3_replication",
        [
            {"artifact": str(A / "results/pairwise_pilot/ablation_metrics.json"), "sha256": sha_file(A / "results/pairwise_pilot/ablation_metrics.json")},
            {"artifact": str(A / "results/pairwise_pilot/bootstrap.json"), "sha256": sha_file(A / "results/pairwise_pilot/bootstrap.json")},
            {"artifact": str(A / "results/pairwise_fulldev/fulldev_final.json"), "sha256": sha_file(A / "results/pairwise_fulldev/fulldev_final.json")},
            {"artifact": str(A / "results/pairwise_confirm/confirm_scalar_metrics.json"), "sha256": sha_file(A / "results/pairwise_confirm/confirm_scalar_metrics.json")},
            {"artifact": str(A / "results/pairwise_confirm/confirm_bootstrap_scalar_vs_fixed.json"), "sha256": sha_file(A / "results/pairwise_confirm/confirm_bootstrap_scalar_vs_fixed.json")},
            {"artifact": str(A / "results/pairwise_confirm/confirm_bootstrap_scalar_vs_resid.json"), "sha256": sha_file(A / "results/pairwise_confirm/confirm_bootstrap_scalar_vs_resid.json")},
        ],
        notes="held-out scalar-resid has point estimate only (no frozen bootstrap CI).",
    )

    # ----- Figure 4 margin mechanism (two panel-ready CSVs + two figures) -----
    mb_path = A / "results/pairwise_confirm/confirm_margin_bins.csv"
    mb = pd.read_csv(mb_path)
    # normalize labels for paper
    label_map = {
        "lowest_10pct": "lowest10",
        "p10_p30": "10-30",
        "p30_p60": "30-60",
        "highest_40pct": "highest40",
    }
    mb = mb.copy()
    mb["margin_bin_label"] = mb["margin_bin"].map(label_map).fillna(mb["margin_bin"])
    df4a = mb[["margin_bin_label", "n", "delta_scalar_fixed", "delta_scalar_resid"]].rename(
        columns={"margin_bin_label": "margin_bin", "delta_scalar_fixed": "ΔF1@0.5_vs_fixed", "delta_scalar_resid": "ΔF1@0.5_vs_resid"}
    )
    df4b = mb[["margin_bin_label", "n", "switch_precision", "fixes", "breaks", "net_fixes"]].rename(
        columns={"margin_bin_label": "margin_bin"}
    )
    dump_csv("fig4_margin_effect", df4a)
    dump_csv("fig4_margin_switch_precision", df4b)

    order4 = ["lowest10", "10-30", "30-60", "highest40"]
    df4a_o = df4a.set_index("margin_bin").loc[order4]
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    ax.plot(order4, df4a_o["ΔF1@0.5_vs_fixed"], marker="o", label="scalar-fixed", color="0.2")
    ax.plot(order4, df4a_o["ΔF1@0.5_vs_resid"], marker="s", label="scalar-resid", color="0.55")
    ax.axhline(0.0, color="0.7", linewidth=0.8)
    ax.set_xlabel("Confirm margin bin (post-hoc)")
    ax.set_ylabel("ΔF1@0.5")
    ax.set_title("Confirm margin mechanism (post-hoc)")
    ax.legend(frameon=False)
    save_fig("fig4_margin_effect")
    write_source(
        "fig4_margin_effect",
        [{"artifact": str(mb_path), "sha256": sha_file(mb_path)}],
        notes="Post-hoc mechanism analysis on confirm; not used for gate design.",
    )

    df4b_o = df4b.set_index("margin_bin").loc[order4]
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    ax.plot(order4, df4b_o["switch_precision"], marker="o", color="0.25")
    ax.set_xlabel("Confirm margin bin (post-hoc)")
    ax.set_ylabel("Switch precision")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Confirm margin vs switch precision (post-hoc)")
    save_fig("fig4_margin_switch_precision")
    write_source(
        "fig4_margin_switch_precision",
        [{"artifact": str(mb_path), "sha256": sha_file(mb_path)}],
        notes="Separate panel; avoid dual-y complexity.",
    )

    # ----- Figure 5 fixes vs breaks -----
    df5 = pd.DataFrame(
        [
            {"outcome": "fixes", "count": int(cs["fixes"])},
            {"outcome": "breaks", "count": int(cs["breaks"])},
            {"outcome": "net", "count": int(cs["net_fixes"])},
        ]
    )
    dump_csv("fig5_switch_outcomes", df5)
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.bar(["fixes", "breaks"], [cs["fixes"], cs["breaks"]], color=["0.35", "0.7"], edgecolor="black", linewidth=0.6)
    ax.set_ylabel("Count")
    ax.set_title("Confirm switch outcomes (net=+%d)" % int(cs["net_fixes"]))
    ax.text(0.5, 0.95, "switch_precision=%.3f  Q2_recovery=%.3f" % (cs["switch_precision"], cs["Q2_recovery_rate"]),
            transform=ax.transAxes, ha="center", va="top", fontsize=9)
    save_fig("fig5_switch_outcomes")
    write_source(
        "fig5_switch_outcomes",
        [{"artifact": str(A / "results/pairwise_confirm/confirm_scalar_metrics.json"), "sha256": sha_file(A / "results/pairwise_confirm/confirm_scalar_metrics.json")}],
    )

    # ----- Supplement negative -----
    soft = json.loads((A / "results/multistation_soft/soft_ring_best.json").read_text())
    hard = soft["hard_gate_best_ref"]
    wb = json.loads((A / "results/pairwise_pilot/waveform_bootstrap.json").read_text())
    move_best = json.loads((A / "results/multistation_moveout/moveout_best.json").read_text())
    best_move = move_best["best_legal"]
    dfs = pd.DataFrame(
        [
            {"method": "hard_ring", "delta_f1@0.5": hard["delta_f1@0.5"], "note": "abstain"},
            {"method": "soft_ring", "delta_f1@0.5": soft["best_overall"]["delta_f1@0.5"], "note": "small"},
            {"method": "moveout_legal", "delta_f1@0.5": float(best_move["delta_f1@0.5"]), "note": "NO-GO"},
            {"method": "waveform_only", "delta_f1@0.5": ab["waveform_only"]["delta_f1@0.5"], "note": "isolation"},
            {"method": "waveform+scalar_vs_scalar", "delta_f1@0.5": wb["delta_waveform_point"], "note": "CI crosses 0"},
        ]
    )
    dump_csv("figS_negative_ablations", dfs)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.bar(dfs["method"], dfs["delta_f1@0.5"], color="0.5", edgecolor="black", linewidth=0.6)
    ax.axhline(0.0, color="0.7", linewidth=0.8)
    ax.set_ylabel("ΔF1@0.5")
    ax.set_title("Negative / non-primary branches (supplement)")
    ax.tick_params(axis="x", rotation=25)
    save_fig("figS_negative_ablations")
    write_source(
        "figS_negative_ablations",
        [
            {"artifact": str(A / "results/multistation_soft/soft_ring_best.json"), "sha256": sha_file(A / "results/multistation_soft/soft_ring_best.json")},
            {"artifact": str(A / "results/multistation_moveout/moveout_metrics.csv"), "sha256": sha_file(A / "results/multistation_moveout/moveout_metrics.csv")},
            {"artifact": str(A / "results/pairwise_pilot/waveform_bootstrap.json"), "sha256": sha_file(A / "results/pairwise_pilot/waveform_bootstrap.json")},
        ],
        notes="Supplement only; do not dominate main story.",
    )

    # silence unused
    _ = (cf, cr)
    print("figures written", OUT_A)


if __name__ == "__main__":
    main()
