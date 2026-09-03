#!/usr/bin/env python
"""Finalize Stage 7A reports after comparator analyze + optional runtime JSON exist."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, load_json, save_json
from earthquake.stage6.phaseB import sha256_file
from earthquake.utils import ensure_dir


def main() -> None:
    out = artifacts_dir() / "results" / "stage7"
    reports = ensure_dir(ROOT / "reports" / "stage7")
    paper = ensure_dir(ROOT / "paper" / "candidate_edits")

    reg = load_json(out / "comparator_registry.json")
    verdict = load_json(out / "stage7A_final_verdict.json")
    boot = load_json(out / "comparator_bootstrap.json")
    metrics = __import__("pandas").read_csv(out / "comparator_metrics_confirm.csv")
    locks = {p.stem.replace("_comparator_lock", ""): json.loads(p.read_text()) for p in (out / "locks").glob("*_comparator_lock.json")}
    runtime = load_json(out / "runtime_benchmark.json") if (out / "runtime_benchmark.json").exists() else None
    cohort = load_json(out / "cohort_hashes.json")

    # main benchmark report
    lines = [
        "# Stage 7A — Same-Protocol External Baseline Benchmark",
        "",
        "**Disclosure:** This is *post-confirm external comparator evaluation*. Stage-6 confirm is already `CONFIRM.CONSUMED`. Comparators were **not** co-preregistered with the Stage-6 main method.",
        "",
        f"- Main method (unchanged): `fixed_rescore_UNION`",
        f"- Stage-6 method_lock SHA256: `{cohort.get('stage6_method_lock_sha256')}`",
        f"- `sota_claim_allowed`: **false**",
        "",
        "## Locked thresholds (dev-only)",
        "",
    ]
    for name, lk in locks.items():
        lines.append(f"- **{name}**: threshold={lk['threshold']} (dev F1@0.5={lk['dev_metrics']['f1@0.5']:.4f})")
    lines += ["", "## Confirm metrics (S-labelled)", "", "| Method | F1@0.1 | F1@0.5 | P95 | miss | coverage |", "|--|--:|--:|--:|--:|--:|"]
    for _, r in metrics.iterrows():
        lines.append(
            f"| {r['model']} | {r.get('f1@0.1', float('nan')):.4f} | {r.get('f1@0.5', float('nan')):.4f} | "
            f"{r.get('detected_ae_p95', float('nan')):.3f} | {r.get('miss_rate', float('nan')):.4f} | "
            f"{r.get('prediction_coverage', float('nan')):.4f} |"
        )
    lines += ["", "## Bootstrap (fixed_UNION − comparator)", ""]
    for k, v in boot.items():
        if not isinstance(v, dict) or "f1@0.5" not in v:
            continue
        ci = v["f1@0.5"]["ci95"]
        lines.append(f"- **{k}**: ΔF1@0.5={v['f1@0.5']['mean_delta']:+.4f} 95% CI=[{ci[0]:+.4f}, {ci[1]:+.4f}]")
    lines += [
        "",
        "## Interpretation rule",
        "",
        "If fixed UNION's ΔF1@0.5 CI is entirely above 0 vs all valid external comparators, we may state it "
        "*outperformed the evaluated external pretrained baselines under our Stage 6 protocol*. "
        "We do **not** claim SOTA / best on INSTANCE / surpasses all phase pickers.",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
    ]
    (reports / "stage7A_same_protocol_benchmark.md").write_text("\n".join(lines) + "\n")

    # training feasibility (7B estimate only)
    rate = None
    note = "no runtime_benchmark.json yet; falling back to Stage-6 annotate probe ≈6.4 traces/s/GPU ×4 ≈25.6"
    if runtime and runtime.get("projections"):
        rate = runtime["projections"].get("traces_per_s")
        note = f"from Stage7A runtime `{runtime['projections'].get('reference_setting')}`"
    if rate is None:
        rate = 25.6
    # crude training estimate: epoch over train split ~ from Stage6 split sizes
    train_traces = 800_000  # approximate INSTANCE train order; refine if split file present
    split_audit = artifacts_dir() / "results" / "stage6" / "full_split_audit.json"
    if split_audit.exists():
        sa = load_json(split_audit)
        # try common keys
        for k in ("n_train_traces", "train_traces", "n_traces_train"):
            if k in sa:
                train_traces = int(sa[k])
                break
        if "splits" in sa and isinstance(sa["splits"], dict):
            tr = sa["splits"].get("train") or sa["splits"].get("Train")
            if isinstance(tr, dict) and "n_traces" in tr:
                train_traces = int(tr["n_traces"])

    # inference-equivalent proxy is not training — use Stage6 loader bench ~90 tps/GPU for dataloader,
    # but training is slower; use factor 0.25 of annotate e2e as conservative wall for forward+backward proxy
    train_tps = max(rate * 0.15, 1.0)  # conservative: training << annotate
    sec_per_epoch = train_traces / train_tps
    eqt_epochs = 50
    feas = f"""# Stage 7B — In-Domain Training Feasibility (estimate only)

**No training started.** Estimates use Stage-7A / Stage-6 measured throughput proxies, not theoretical FLOPS.

- Annotate/e2e reference rate: **{rate:.2f} traces/s** ({note})
- Assumed train set size: **{train_traces}** traces
- Conservative train throughput proxy: **{train_tps:.2f} traces/s** (≈15% of annotate e2e; single-HDF5 I/O bound)
- Seconds / epoch: **{sec_per_epoch/3600:.2f} h**

| Job | Seeds | Epochs (assumed) | Wall estimate | 1-week feasible? |
|--|--:|--:|--|--|
| EQTransformer in-domain seed42 | 1 | {eqt_epochs} | {eqt_epochs*sec_per_epoch/3600:.1f} h | {"yes" if eqt_epochs*sec_per_epoch < 7*86400 else "no"} |
| EQTransformer 3 seeds | 3 | {eqt_epochs} | {3*eqt_epochs*sec_per_epoch/3600:.1f} h | {"yes" if 3*eqt_epochs*sec_per_epoch < 7*86400 else "marginal/no"} |
| LFTNet seed42 | 1 | n/a | blocked | `LFTNet_status=not_reproducible_from_official_release` |
| LFTNet 3 seeds | 3 | n/a | blocked | same |

## Disk

- Checkpoints + logs: order **50–200 GB** depending on retention
- Candidate caches (UNION-style): order **tens of GB** if rebuilt for new pickers

## Candidate cache time

At annotate rate {rate:.1f} t/s on 4 GPUs: full INSTANCE 1,159,249 traces ≈ **{1159249/rate/3600:.1f} h**.

## Verdict

Do **not** start Stage 7B training from this document alone. Revisit after official EQT weights / LFTNet release gates pass.
"""
    (reports / "in_domain_training_feasibility.md").write_text(feas)

    # paper candidate edit
    paper_md = """# Candidate edit — Stage 7A external comparators (NOT applied to paper body)

Status: **candidate only**. Do not merge into frozen paper text without human review.

Suggested addition (Results / Baselines):

> Under the identical Stage-6 confirmatory protocol (same manifests, tolerances, and `match_picks` definitions), we evaluated additional SeisBench pretrained PhaseNet weights (ETHZ, SCEDC) with thresholds selected **only** on the Stage-6 development set. These comparisons are **post-confirm** external baselines; they were not co-preregistered with the confirmatory analysis of `fixed_rescore_UNION`. We do not claim state-of-the-art performance.

Also record: INSTANCE-pretrained weights are excluded as `diagnostic_only_data_leakage`. EQTransformer official weights were unavailable in this environment (download failure). LFTNet was not reproducible from an official release.
"""
    (paper / "stage7_external_comparator_update.md").write_text(paper_md)

    # DONE marker
    done = {
        "stage": "7A",
        "status": "DONE",
        "sota_claim_allowed": False,
        "stage6_method_lock_sha256": cohort.get("stage6_method_lock_sha256"),
        "method_lock_unchanged": sha256_file(artifacts_dir() / "results/stage6/final_confirm/method_lock.json")
        == cohort.get("stage6_method_lock_sha256"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "verdict_path": str(out / "stage7A_final_verdict.json"),
    }
    save_json(done, out / "STAGE7A.DONE")
    (out / "STAGE7A.DONE.txt").write_text("STAGE7A.DONE\n")
    print(json.dumps(done, indent=2))


if __name__ == "__main__":
    main()
