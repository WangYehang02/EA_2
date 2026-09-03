#!/usr/bin/env python
"""Write Stage-4 confirmatory report + final verdict JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, load_json, save_json
from earthquake.utils import ensure_dir


def main() -> None:
    out = ensure_dir(artifacts_dir() / "results" / "stage4")
    reports = ensure_dir(ROOT / "reports")
    lock = load_json(out / "method_lock.json")
    audit = load_json(out / "holdout_split_audit.json")
    main_tab = pd.read_csv(out / "tables" / "main_results.csv") if (out / "tables" / "main_results.csv").exists() else pd.DataFrame()
    boot = load_json(out / "bootstrap_confirmatory.json") if (out / "bootstrap_confirmatory.json").exists() else {}

    def row(method):
        r = main_tab[main_tab.method == method]
        return r.iloc[0].to_dict() if len(r) else {}

    pn, fx, gate = row("phasenet"), row("fixed_catalog_rescore"), row("learned_gate")
    cmp_pn = boot.get("comparisons", {}).get("fixed_catalog_rescore_vs_phasenet", {})
    cmp_gate = boot.get("comparisons", {}).get("fixed_catalog_rescore_vs_learned_gate", {})

    def beats(d, key, direction="gt0"):
        if not d or key not in d:
            return None
        lo, hi = d[key]["ci95_low"], d[key]["ci95_high"]
        if direction == "gt0":
            return bool(lo > 0)
        return bool(hi < 0)

    # identity fallback check
    safe_fb = None
    if len(main_tab):
        idr = row("history_unavailable")
        if idr and pn:
            # F1@0.5 should be near PhaseNet (candidate argmax path)
            safe_fb = abs(float(idr.get("S_f1@0.5", 0) - pn.get("S_f1@0.5", 0))) < 0.02

    # transfer
    transfer_ok = None
    tr_path = out / "tables" / "transfer_weights_results.csv"
    if tr_path.exists():
        tr = pd.read_csv(tr_path)
        # direction: fixed F1@0.5 >= phasenet for each weight
        ok = []
        for w in tr.weight.unique():
            sub = tr[tr.weight == w]
            try:
                pn_f = float(sub[sub.method == "phasenet"]["S_f1@0.5"].iloc[0])
                fx_f = float(sub[sub.method == "fixed_catalog_rescore"]["S_f1@0.5"].iloc[0])
                ok.append(fx_f >= pn_f - 1e-9)
            except Exception:
                pass
        transfer_ok = bool(ok) and all(ok)

    replicates = None
    if fx and pn:
        # Stage3: fixed 0.718 vs PN 0.695; holdout direction
        replicates = bool(fx.get("S_f1@0.5", 0) >= pn.get("S_f1@0.5", 0))

    underpowered = bool(audit.get("insufficient_unused_events"))
    stable = (
        audit.get("overlap_used_events") == 0
        and lock.get("method_name") == "fixed_catalog_rescore"
        and beats(cmp_pn, "delta_f1_0.5", "gt0") is True
        and safe_fb is not False
        and not underpowered  # cannot claim stable with 9 events
    )

    claim = (
        "Historical travel-time residual guided candidate re-ranking significantly improves an external pretrained picker for catalog-assisted S-phase repicking on INSTANCE."
        if stable
        else (
            "On the Stage-4 unused-test confirmatory holdout, fixed catalog_rescore remains the designated main method, "
            "but the holdout is severely underpowered (only leftover unused test events after Stage-3), so a 'stable improvement' claim is not allowed from Stage-4 alone. "
            "Rely on Stage-2/3 test-only evidence with explicit catalog-assisted scope; do not claim phase-picking SOTA."
        )
    )

    verdict = {
        "holdout_is_event_disjoint": audit.get("overlap_used_events") == 0 and audit.get("overlap_train_events") == 0,
        "holdout_was_unused_before_stage4": audit.get("overlap_used_events") == 0,
        "method_was_frozen_before_inference": True,
        "holdout_underpowered": underpowered,
        "holdout_n_events": audit.get("n_events"),
        "holdout_n_traces": audit.get("n_traces"),
        "fixed_beats_phasenet_f1_01": beats(cmp_pn, "delta_f1_0.1", "gt0"),
        "fixed_beats_phasenet_f1_05": beats(cmp_pn, "delta_f1_0.5", "gt0"),
        "fixed_beats_phasenet_p95": beats(cmp_pn, "delta_e2e_p95", "lt0"),
        "fixed_beats_gate": beats(cmp_gate, "delta_f1_0.5", "gt0") if cmp_gate else None,
        "replicates_stage3": replicates,
        "safe_fallback_without_history": safe_fb,
        "transfer_direction_consistent": transfer_ok,
        "recommended_main_method": "fixed_catalog_rescore",
        "recommended_claim": claim,
        "sota_claim_allowed": False,
        "stable_improvement_claim_allowed": bool(stable),
        "method_role": "catalog-assisted S-phase candidate re-picking/refinement",
    }
    save_json(verdict, out / "stage4_final_verdict.json")

    def fmt(r, k):
        return "NA" if not r or k not in r else f"{float(r[k]):.3f}"

    md = f"""# Stage 4 Confirmatory Report

## 1. Executive summary

- **Main method (frozen):** `fixed_catalog_rescore` — catalog-assisted S-phase candidate re-picking/refinement.
- **Holdout:** {audit.get('n_traces')} traces / {audit.get('n_events')} events (unused chronological test leftovers).
- **Underpowered:** {underpowered} — Stage-3 test-only already consumed most test events; target 20k/1k not reachable without reuse.
- **SOTA claim allowed:** false.
- **Stable improvement claim from Stage-4 alone:** {verdict['stable_improvement_claim_allowed']}.

Key holdout S metrics:

| Method | F1@0.1 | F1@0.5 | e2e P95 |
|--------|-------:|-------:|--------:|
| PhaseNet | {fmt(pn,'S_f1@0.1')} | {fmt(pn,'S_f1@0.5')} | {fmt(pn,'S_e2e_p95')} |
| fixed catalog_rescore | {fmt(fx,'S_f1@0.1')} | {fmt(fx,'S_f1@0.5')} | {fmt(fx,'S_e2e_p95')} |
| learned gate (ablation) | {fmt(gate,'S_f1@0.1')} | {fmt(gate,'S_f1@0.5')} | {fmt(gate,'S_e2e_p95')} |

## 2. Method lock

Locked before inference (`artifacts/results/stage4/method_lock.json`):

- PhaseNet weight: STEAD + UTC remap
- K=5, shrinkage_k=50
- Lambdas from Stage-2 VAL search (not Stage-3/4 test): S lw=0.5, lh=2.0, lp=0.0
- **Disclosure:** Stage-3 test metrics were *viewed* when choosing fixed over gate as main method; lambdas themselves were selected on Stage-2 validation, not on Stage-3 test-only.

## 3. Holdout construction and leakage audit

- Selection used only event_id / origin_time / availability / counts (no labels/SNR/preds).
- Event overlap with Stage1–3 used events: **{audit.get('overlap_used_events')}**
- Overlap train/val: **{audit.get('overlap_train_events')} / {audit.get('overlap_val_events')}**
- Warning: {audit.get('warning')}

## 4. Main results

See `artifacts/results/stage4/tables/main_results.csv` and `ablation_results.csv`.

Interpretation note: gains concentrated on reducing gross wrong-peak / long-tail e2e errors (F1@0.5 / P95) should **not** be phrased as sub-sample-point precision SOTA if F1@0.1 does not improve.

## 5. Bootstrap significance

Event-level paired bootstrap, n={boot.get('n_bootstrap', 'NA')}.

See `artifacts/results/stage4/bootstrap_confirmatory.json` and `tables/bootstrap_results.csv`.

With only {audit.get('n_events')} events, CIs are wide; do not over-claim.

## 6. Ablation

Includes distance-only, raw path, unshrunk/shrunk residual, shuffled/biased history, no-history identity, learned gate, oracles.

## 7. Difficult subsets

Stage-2 predefined bins only: `tables/hard_subset_results.csv`. Groups with `small_n=true` are diagnostic only.

## 8. Weight transfer

`tables/transfer_weights_results.csv` (stead / ethz / scedc if run). Instance weight excluded from formal results.

## 9. Noise / fallback

Catalog rescore without history must identity-fallback toward PhaseNet candidate ranking. This method requires event context and path history; it is **not** a continuous event detector.

## 10. Compute cost

See `tables/compute_cost.csv`.

## 11. Failure cases / figures

See `artifacts/results/stage4/figures/`.

## 12. Limitations

1. Confirmatory holdout is tiny after Stage-3 consumption of test events.
2. Catalog-assisted (needs origin/location/path).
3. Learned gate did not beat fixed on Stage-3; kept as ablation only.
4. No public-benchmark same-protocol SOTA comparison.

## 13. Paper claim boundary

Allowed (if supported by Stage-2/3 + careful language): catalog-assisted residual-guided candidate re-ranking improves an external pretrained picker for S-phase repicking on INSTANCE.

**Forbidden:** blind phase-picking SOTA; continuous detector claims; hiding Stage-3 inspection of test metrics when choosing main method.

## 14. Final recommendation

- **Main method:** fixed catalog_rescore
- **Ablation:** learned gate
- **Next step:** paper writing — do **not** train GNN or retune on holdout
- Verdict JSON: `artifacts/results/stage4/stage4_final_verdict.json`

Recommended claim text:

> {claim}
"""
    (reports / "stage4_confirmatory_report.md").write_text(md)
    print({"report": str(reports / "stage4_confirmatory_report.md"), "verdict": verdict}, flush=True)


if __name__ == "__main__":
    main()
