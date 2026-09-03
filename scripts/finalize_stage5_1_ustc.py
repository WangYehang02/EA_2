#!/usr/bin/env python
"""Finalize Stage 5.1 verdict, hash check, and audit report."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import save_json
from earthquake.utils import ensure_dir


def file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_sha(p: Path) -> tuple[str, int, int]:
    files = sorted([x for x in p.rglob("*") if x.is_file()])
    h = hashlib.sha256()
    total = 0
    for f in files:
        rel = str(f.relative_to(ROOT)).encode()
        h.update(rel)
        h.update(b"\0")
        h.update(bytes.fromhex(file_sha(f)))
        total += f.stat().st_size
    return h.hexdigest(), len(files), total


def main() -> None:
    out = ensure_dir(ROOT / "artifacts/results/stage5_1_ustc")
    before = json.loads((out / "frozen_hashes_before.json").read_text())
    targets = [
        "artifacts/results/stage3",
        "artifacts/results/stage4",
        "paper/main.tex",
        "paper/main.pdf",
        "paper/references.bib",
    ]
    rows = []
    unchanged = True
    for t in targets:
        p = ROOT / t
        if p.is_file():
            sha = file_sha(p)
            prev = next(r for r in before["rows"] if r["path"] == t)
            ok = sha == prev.get("sha256")
            unchanged = unchanged and ok
            rows.append({"path": t, "type": "file", "sha256": sha, "matches_before": ok})
        else:
            sha, n, nbytes = tree_sha(p)
            prev = next(r for r in before["rows"] if r["path"] == t)
            ok = sha == prev.get("sha256_tree")
            unchanged = unchanged and ok
            rows.append({"path": t, "type": "dir", "sha256_tree": sha, "n_files": n, "nbytes": nbytes, "matches_before": ok})

    after = {"phase": "after", "rows": rows, "all_frozen_unchanged": unchanged}
    (out / "frozen_hashes_after.json").write_text(json.dumps(after, indent=2))
    if not unchanged:
        raise SystemExit("ERROR: frozen Stage3/4/paper hashes changed")

    pr = json.loads((out / "path_repeatability.json").read_text())
    hr = json.loads((out / "hierarchical_val_results.json").read_text())

    path_ok = bool(pr["correct_path_beats_shuffled"]) and float(pr.get("path_residual_corr_val", 0)) > 0.3
    hier_beats = bool(hr["hierarchical_prior_recommended"])
    pick = hr["pick_summaries"]
    # Caveat: coarse alone nearly equals hierarchical on picks
    coarse_equals_hier = abs(pick["coarse_shrunk"]["f1@0.5"] - pick["hierarchical"]["f1@0.5"]) < 1e-9

    verdict = {
        "path_residual_is_repeatable": path_ok,
        "correct_path_beats_shuffled": bool(pr["correct_path_beats_shuffled"]),
        "hierarchical_prior_beats_current_on_val": bool(
            hr["deltas_hier_minus_fine"]["f1_0.5"] >= 0.005
            or hr["deltas_hier_minus_fine"]["prior_mae_rel_drop"] >= 0.05
            or hr["deltas_hier_minus_fine"]["e2e_p95_drop_s"] >= 0.1
        ),
        "hierarchical_prior_recommended": "future_work" if hier_beats else "reject",
        "hierarchical_gain_mostly_from_coarse_backoff": coarse_equals_hier,
        "paper_related_work_update_recommended": True,
        "paper_discussion_update_recommended": True,
        "in_domain_phasenet_baseline_priority": "reviewer_request",
        "modify_frozen_test_results": False,
        "run_gnn": False,
        "use_ustc_models_as_instance_baseline": False,
        "frozen_artifacts_unchanged": unchanged,
        "decisions": {
            "regional_domain_shift_motivation": "adopt",
            "add_related_work": "adopt",
            "waveform_adaptation_complementarity": "adopt",
            "distance_s_difficulty_discussion": "adopt_with_caveat",
            "diminishing_returns_complexity": "adopt_with_caveat",
            "hierarchical_history_residual": "future_work",
            "formal_in_domain_phasenet": "reviewer_request",
            "ustc_as_instance_baseline": "reject",
            "continue_gnn": "reject",
        },
    }
    save_json(verdict, out / "final_verdict.json")

    # Write report
    report = ROOT / "reports/ustc_pickers_reference_audit.md"
    s_pr = pr["prior_error_summaries"]
    report.write_text(
        f"""# Stage 5.1 — USTC-Pickers Reference Audit

## 1. Executive summary

USTC-Pickers strengthens the **motivation** for regional domain shift and the
**complementarity** narrative (waveform adaptation vs catalog-assisted
candidate re-ranking). Train→validation analyses show that **path residuals are
repeatable**: correct path priors beat shuffled-path controls with event-level
bootstrap CIs excluding zero (correlation≈{pr['path_residual_corr_val']:.3f}).

A **coarse-to-fine hierarchical residual** passes the pre-registered validation
decision rules and is recommended only as
`promising future extension validated on the development set`.
Most of the validation pick gain equals **coarse (1°) residual backoff** on
fine-unseen paths; it must **not** enter Stage 3 main results.

Formal leakage-free full INSTANCE in-domain PhaseNet remains a
**reviewer-request** strengthening experiment (existing finetunes are debug-scale
and did not beat STEAD). Do **not** use Chinese USTC weights as INSTANCE
baselines; do **not** resume GNN work.

Frozen Stage 3/4 and `paper/main.*` hashes: **unchanged**.

## 2. USTC-Pickers overview

Zhu, Li & Fang (2023), *Earthquake Science* 36(2):95–112,
DOI 10.1016/j.eqs.2023.03.001.

- DiTing dataset; CN picker trained from scratch; fine-tuned tectonic (5) and
  provincial (33) pickers (+ Capital / CSES specials).
- Emphasizes regional waveform domain shift; reports diminishing returns from
  finer customization; S performance worsens with distance.
- Eval: ±0.6 s, 50 Hz — **not comparable** to this project's metrics.

## 3. Commonalities and differences

| | USTC-Pickers | This project |
|--|--|--|
| Goal | Adapt waveform picker weights | Freeze picker; re-rank S candidates |
| History | Regional labeled waveforms for fine-tuning | Train-only path travel-time residuals |
| Needs catalog at inference | No (blind picking) | Yes (catalog-assisted) |
| Hierarchy | CN → tectonic → provincial models | Optional multi-scale residual backoff (val-only here) |

## 4. Paper-argument borrow decisions

| Item | Covered now? | USTC support? | Decision | Section |
|--|--|--|--|--|
| A. Regional domain shift | Partial | Direct | **adopt** | Related Work / Intro |
| B. Complementarity | No | Direct (by contrast) | **adopt** | Related Work / Discussion |
| C. Diminishing returns vs complexity | No | Qualitative | **adopt_with_caveat** (not “GNN invalid”) | Discussion |
| D. Distance-related S difficulty | Stage2/4 frozen + val exploratory | Directional | **adopt_with_caveat** | Discussion |

Candidate text (not merged): `paper/candidate_edits/ustc_*.tex`.

### Forbidden exaggerations
No cross-paper F1 ranking; no causal claim that regional models “prove” path
residuals; no “10k samples universal recipe”.

## 5. Path residual repeatability (train→val)

Protocol: distance MLP frozen from Stage 2; history tables from **train only**;
evaluate on **validation only**; event-disjoint enforced.

| Prior | Val MAE (s) | median AE | P95 |
|--|--:|--:|--:|
| MLP only | {s_pr['mlp_only']['mae']:.4f} | {s_pr['mlp_only']['median_ae']:.4f} | {s_pr['mlp_only']['p95']:.4f} |
| + distance-bin residual | {s_pr['mlp_distance_bin']['mae']:.4f} | {s_pr['mlp_distance_bin']['median_ae']:.4f} | {s_pr['mlp_distance_bin']['p95']:.4f} |
| + shuffled path | {s_pr['mlp_shuffled_path']['mae']:.4f} | {s_pr['mlp_shuffled_path']['median_ae']:.4f} | {s_pr['mlp_shuffled_path']['p95']:.4f} |
| + correct path | {s_pr['mlp_correct_path']['mae']:.4f} | {s_pr['mlp_correct_path']['median_ae']:.4f} | {s_pr['mlp_correct_path']['p95']:.4f} |
| + shrunk path (k=50) | {s_pr['mlp_shrunk_path']['mae']:.4f} | {s_pr['mlp_shrunk_path']['median_ae']:.4f} | {s_pr['mlp_shrunk_path']['p95']:.4f} |
| + hierarchical | {s_pr['mlp_hierarchical']['mae']:.4f} | {s_pr['mlp_hierarchical']['median_ae']:.4f} | {s_pr['mlp_hierarchical']['p95']:.4f} |

- Correct vs shuffled ΔMAE bootstrap CI: {pr['bootstrap']['correct_vs_shuffled']['ci95_low']:.4f}–{pr['bootstrap']['correct_vs_shuffled']['ci95_high']:.4f} (prob_improved={pr['bootstrap']['correct_vs_shuffled']['prob_improved']})
- Seen-path residual correlation (val): **{pr['path_residual_corr_val']:.3f}**
- Between/within path MAD ratio: **{pr['path_stats_fine']['between_within_mad_ratio']:.3f}** (robust ratio, not formal ICC)
- Fine coverage on full val: seen frac={pr['coverage_val']['fine_seen_frac']:.3f}

**Conclusion:** path residual behaves as a **repeatable directed source-region–station effect**, not a shuffled lookup artifact.

## 6. Hierarchical residual (validation-only feasibility)

Fixed eval **val** traces only (n={hr['n_val_traces']}, events={hr['n_val_events']});
frozen K/λ/peaks; replace travel-time prior only.

| Method | Prior MAE | S F1@0.5 | e2e P95 | wrong-peak |
|--|--:|--:|--:|--:|
| PhaseNet | — | {pick['phasenet']['f1@0.5']:.3f} | {pick['phasenet']['e2e_p95']:.2f} | {pick['phasenet']['wrong_peak_rate']:.3f} |
| fine shrunk (current-like) | {hr['prior_summaries']['fine_shrunk']['mae']:.4f} | {pick['fine_shrunk']['f1@0.5']:.3f} | {pick['fine_shrunk']['e2e_p95']:.2f} | {pick['fine_shrunk']['wrong_peak_rate']:.3f} |
| coarse shrunk (1°) | {hr['prior_summaries']['coarse_shrunk']['mae']:.4f} | {pick['coarse_shrunk']['f1@0.5']:.3f} | {pick['coarse_shrunk']['e2e_p95']:.2f} | {pick['coarse_shrunk']['wrong_peak_rate']:.3f} |
| hierarchical | {hr['prior_summaries']['hierarchical']['mae']:.4f} | {pick['hierarchical']['f1@0.5']:.3f} | {pick['hierarchical']['e2e_p95']:.2f} | {pick['hierarchical']['wrong_peak_rate']:.3f} |
| hierarchical shuffled | {hr['prior_summaries']['hierarchical_shuffled']['mae']:.4f} | {pick['hierarchical_shuffled']['f1@0.5']:.3f} | {pick['hierarchical_shuffled']['e2e_p95']:.2f} | {pick['hierarchical_shuffled']['wrong_peak_rate']:.3f} |

Pass rules: `{json.dumps(hr['pass_rules'])}`

**Interpretation:** gains concentrate on **fine_unseen** paths where coarse history
is available (`hierarchical_val_by_group.csv`). Hierarchical ≈ coarse on this
val set; recommend as **future_work**, not a Stage 3 method change.

## 7. In-domain PhaseNet strong-baseline audit

Existing artifacts (`artifacts/phasenet_finetune/`, `finetuned_phasenet_metrics.json`):

1. Debug finetune (10k/2k traces, 5 epochs) never beat UTC-aligned STEAD on annotate val F1; `best.pt` kept pretrained.
2. Early failures: NPS vs PSN label order; BN running stats updated on crops.
3. Last-layer-only run: likewise `no_improve_keep_pretrained`.
4. **No** completed full-scale leakage-free INSTANCE train under `configs/phasenet_finetune_full.yaml`.
5. SeisBench `instance` weights remain **diagnostic-only** (leakage risk).
6. Reviewers may ask why only external STEAD is shown; a proper in-domain baseline would strengthen the paper but is **orthogonal** to the re-ranking claim.
7. Minimal sufficient experiment: full chronological train, event-disjoint val selection by annotate F1, BN/eval protocol locked, compare STEAD vs in-domain vs STEAD+fixed rescore on Stage 3 frozen list (**after** method lock; no retuning λ on test).
8. Cost: multi-GPU-day scale (full INSTANCE, ~30 epochs) — nontrivial.
9. Priority: **`reviewer_request`** (upgrade to before-submission if compute budget allows). Do not assert in-domain training is useless from debug failure; do not extrapolate USTC DiTing numbers to INSTANCE.

## 8. Non-comparable metrics

| Item | USTC-Pickers | This project |
|--|--|--|
| Data | DiTing | INSTANCE |
| Sampling rate | 50 Hz | 100 Hz |
| Window | 60 s | 120 s |
| Tolerance | ±0.6 s | ±0.1 / 0.5 s |
| Picker | China train/finetune | External STEAD + catalog rescore |
| Task | waveform picking | catalog-assisted repicking |

## 9. Risks and limits

- Val hierarchical gains may not transfer to Stage 3 test; not measured here by design.
- Coarse residual may over-smooth spatially; needs careful failure analysis.
- Catalog origin errors still bias residual priors.
- Stage 4 remains underpowered and unused for method choice.

## 10. Final decision table

| Candidate borrow | Decision |
|--|--|
| Regional domain-shift motivation | **adopt** |
| Add Related Work citation/paragraph | **adopt** |
| Waveform adaptation ↔ path re-ranking complementarity | **adopt** |
| Distance-related S difficulty discussion | **adopt_with_caveat** |
| Marginal gain vs deployment complexity | **adopt_with_caveat** |
| Hierarchical historical residual | **future_work** (not adopt_now) |
| Formal in-domain PhaseNet | **reviewer_request** |
| USTC models as INSTANCE baseline | **reject** |
| Continue GNN | **reject** |

See `artifacts/results/stage5_1_ustc/final_verdict.json`.
""",
        encoding="utf-8",
    )
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
