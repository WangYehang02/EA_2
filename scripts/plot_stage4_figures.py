#!/usr/bin/env python
"""Paper-style Stage-4 figures (PNG+PDF, 300 dpi)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir
from earthquake.utils import ensure_dir

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.dpi": 300, "savefig.dpi": 300})


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"))
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    out = ensure_dir(artifacts_dir() / "results" / "stage4" / "figures")
    picks_path = artifacts_dir() / "results" / "stage4" / "holdout_picks.parquet"
    if not picks_path.exists():
        raise SystemExit("missing picks")
    df = pd.read_parquet(picks_path)
    sr = df.sampling_rate_hz.to_numpy()
    true = df.true_s_sample.to_numpy()
    e_pn = np.abs(df.pred_s_phasenet - true) / sr
    e_fx = np.abs(df.pred_s_catalog_rescore - true) / sr
    m = np.isfinite(e_pn) & np.isfinite(e_fx)

    # 1 error CDF
    fig, ax = plt.subplots(figsize=(5, 4))
    for err, lab, c in [(e_pn[m], "PhaseNet", "C0"), (e_fx[m], "fixed catalog_rescore", "C1")]:
        x = np.sort(err)
        y = np.linspace(0, 1, len(x), endpoint=False)
        ax.plot(x, y, label=lab, color=c)
    ax.set_xscale("log")
    ax.set_xlabel("Absolute S error (s)")
    ax.set_ylabel("CDF")
    ax.legend()
    ax.set_title("Holdout S error CDF")
    _save(fig, out / "error_cdf_s")

    # 2 F1 vs tolerance
    main = pd.read_csv(artifacts_dir() / "results" / "stage4" / "tables" / "main_results.csv")
    fig, ax = plt.subplots(figsize=(5, 4))
    for method, c in [("phasenet", "C0"), ("fixed_catalog_rescore", "C1"), ("learned_gate", "C2")]:
        row = main[main.method == method]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        ax.plot([0.1, 0.2, 0.5], [r["S_f1@0.1"], r["S_f1@0.2"], r["S_f1@0.5"]], "o-", label=method, color=c)
    ax.set_xlabel("Tolerance (s)")
    ax.set_ylabel("S F1")
    ax.legend()
    ax.set_title("S F1 vs tolerance")
    _save(fig, out / "f1_vs_tolerance")

    # 3 subset gains if available
    hs = artifacts_dir() / "results" / "stage4" / "tables" / "hard_subset_results.csv"
    if hs.exists():
        tab = pd.read_csv(hs)
        for feat, fname in [("residual_s_mad", "gain_by_history_mad"), ("history_count", "gain_by_history_count"), ("s_peak_probability", "gain_by_phasenet_prob"), ("s_n_cands", "gain_by_multipeak")]:
            sub = tab[tab.feature == feat]
            if len(sub) == 0:
                continue
            fig, ax = plt.subplots(figsize=(5.5, 3.8))
            ax.bar(sub["bin"].astype(str), sub["delta_f1_0.5"], color="C1")
            ax.axhline(0, color="k", lw=0.8)
            ax.set_ylabel("Δ S F1@0.5 (fixed−PN)")
            ax.set_title(feat)
            ax.tick_params(axis="x", rotation=20)
            _save(fig, out / fname)

    # 8 bootstrap distributions
    boot = artifacts_dir() / "results" / "stage4" / "bootstrap_samples.parquet"
    if boot.exists():
        b = pd.read_parquet(boot)
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
        for ax, metric, title in [
            (axes[0], "delta_f1_0.5", "Δ S F1@0.5"),
            (axes[1], "delta_e2e_p95", "Δ S e2e P95"),
        ]:
            sub = b[(b.comparison == "fixed_catalog_rescore_vs_phasenet") & (b.metric == metric)]
            if len(sub):
                ax.hist(sub.value, bins=40, color="C1", alpha=0.85)
                ax.axvline(0, color="k", ls="--")
            ax.set_title(title)
        _save(fig, out / "bootstrap_deltas")

    # 9 simple method diagram
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axis("off")
    boxes = ["Waveform", "PhaseNet\n(STEAD)", "Top-K=5\nS candidates", "Residual\nhistory prior", "Fixed\ncatalog_rescore", "S pick"]
    xs = np.linspace(0.05, 0.85, len(boxes))
    for x, t in zip(xs, boxes):
        ax.add_patch(plt.Rectangle((x, 0.35), 0.12, 0.35, fill=False, lw=1.2))
        ax.text(x + 0.06, 0.52, t, ha="center", va="center", fontsize=8)
    for i in range(len(xs) - 1):
        ax.annotate("", xy=(xs[i + 1], 0.52), xytext=(xs[i] + 0.12, 0.52), arrowprops=dict(arrowstyle="->", lw=1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Catalog-assisted S-phase candidate re-picking/refinement")
    _save(fig, out / "method_diagram")
    print({"figures": str(out)}, flush=True)


if __name__ == "__main__":
    main()
