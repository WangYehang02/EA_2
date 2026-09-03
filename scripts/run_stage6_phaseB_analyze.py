#!/usr/bin/env python
"""Phase B analysis: oracle, complementarity, top-1, noise, bootstrap, final verdict."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, save_json
from earthquake.stage6.full_splits import assert_full_confirm_access_allowed
from earthquake.stage6.phaseB import (
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEED,
    complementarity_table,
    event_level_bootstrap_delta,
    oracle_metrics_for_set,
    per_trace_hit,
    sha256_file,
    top1_diagnosis,
    topk_from_cache,
    union_candidates,
)
from earthquake.utils import ensure_dir


def _guard_confirm() -> None:
    seal = artifacts_dir() / "results" / "stage6" / "splits_full" / "CONFIRM_SEALED"
    assert seal.exists()
    try:
        assert_full_confirm_access_allowed(purpose="phaseB_analyze")
        raise SystemExit("method_lock present")
    except RuntimeError:
        pass


def _event_oracle_table(oracle: dict) -> pd.DataFrame:
    eids = oracle["event_ids"]
    pred = oracle["oracle_predictions"]
    true = oracle["true_samples"]
    sr = oracle["sampling_rates"]
    rows = []
    for e in np.unique(eids):
        m = eids == e
        hits01 = per_trace_hit(pred[m], true[m], sr[m], 0.1)
        hits05 = per_trace_hit(pred[m], true[m], sr[m], 0.5)
        rows.append(
            {
                "event_id": e,
                "n_traces": int(m.sum()),
                "oracle_recall@0.1": float(np.nanmean(hits01)),
                "oracle_recall@0.5": float(np.nanmean(hits05)),
            }
        )
    return pd.DataFrame(rows)


def summarize_complement(comp: pd.DataFrame) -> dict:
    n = len(comp)
    out = {}
    for cls in ["both_correct", "stead_only_correct", "ida_only_correct", "neither_correct"]:
        g = comp[comp["class"] == cls]
        out[cls] = {
            "n_traces": int(len(g)),
            "n_events": int(g["event_id"].nunique()) if len(g) else 0,
            "rate": float(len(g) / max(n, 1)),
        }
        if len(g) and "station" in g.columns:
            out[cls]["top_stations"] = g["station"].astype(str).value_counts().head(10).to_dict()
            out[cls]["top_events"] = g["event_id"].astype(str).value_counts().head(10).to_dict()
    # concentration: fraction of ida_only in top 5% events
    io = comp[comp["class"] == "ida_only_correct"]
    if len(io):
        vc = io["event_id"].value_counts()
        top = int(max(1, round(0.05 * len(vc))))
        out["ida_only_concentration"] = {
            "n_events_with_any": int(len(vc)),
            "frac_traces_in_top5pct_events": float(vc.head(top).sum() / max(len(io), 1)),
            "frac_traces_in_top10_events": float(vc.head(10).sum() / max(len(io), 1)),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="artifacts/results/stage6/phaseB_eval_manifest.csv")
    parser.add_argument("--manifest-json", default="artifacts/results/stage6/phaseB_eval_manifest.json")
    parser.add_argument("--stead-cache", default="artifacts/cache/stage6/phaseB/stead_top10/stead_top10.parquet")
    parser.add_argument("--ida-cache", default="artifacts/cache/stage6/phaseB/ida_top10/ida_top10.parquet")
    parser.add_argument("--noise-manifest", default="artifacts/results/stage6/phaseB_noise_manifest.csv")
    parser.add_argument("--noise-stead", default="artifacts/cache/stage6/phaseB/noise_stead_top10/stead_top10.parquet")
    parser.add_argument("--noise-ida", default="artifacts/cache/stage6/phaseB/noise_ida_top10/ida_top10.parquet")
    parser.add_argument("--ida-valid-json", default="artifacts/results/stage6/phaseB_ida_checkpoint_validation.json")
    parser.add_argument("--skip-noise", action="store_true")
    args = parser.parse_args()
    _guard_confirm()

    out = ensure_dir(artifacts_dir() / "results" / "stage6")
    meta = pd.read_csv(ROOT / args.manifest)
    man_json = json.loads((ROOT / args.manifest_json).read_text())
    stead = pd.read_parquet(ROOT / args.stead_cache)
    ida = pd.read_parquet(ROOT / args.ida_cache)

    # same trace set
    mset = set(meta["trace_name"].astype(str))
    assert set(stead["trace_name"].astype(str).unique()) >= mset or len(set(stead["trace_name"].astype(str)) & mset) == len(mset)
    # filter caches to manifest
    stead = stead[stead["trace_name"].astype(str).isin(mset)].copy()
    ida = ida[ida["trace_name"].astype(str).isin(mset)].copy()

    ida_valid = json.loads((ROOT / args.ida_valid_json).read_text())
    if not ida_valid.get("ok"):
        verdict = {
            "decision": "discard_ida_noise_or_instability",
            "ida_checkpoint_valid": False,
            "keep_ida_as_secondary_source": False,
            "recommended_candidate_source": "STEAD",
            "recommended_k": None,
            "ranker_may_start": False,
            "confirm_remains_sealed": True,
            "reason": "ida checkpoint invalid",
        }
        save_json(verdict, out / "phaseB_final_verdict.json")
        print(json.dumps(verdict, indent=2))
        return

    sets = {
        "STEAD_K5": topk_from_cache(stead, 5),
        "STEAD_K10": topk_from_cache(stead, 10),
        "IDA_K5": topk_from_cache(ida, 5),
        "IDA_K10": topk_from_cache(ida, 10),
    }
    union = union_candidates(stead, ida, stead_k=5, ida_k=5, max_union=10)
    assert union.groupby("trace_name").size().max() <= 10
    sets["UNION_K10"] = union

    oracles = {}
    oracle_raw = {}
    for name, cdf in sets.items():
        sample_col = "candidate_sample"
        o = oracle_metrics_for_set(cdf, meta, sample_col=sample_col)
        oracle_raw[name] = o
        oracles[name] = {k: v for k, v in o.items() if k not in {"oracle_predictions", "true_samples", "sampling_rates", "event_ids"}}
        oracles[name]["event_level"] = _event_oracle_table(o).to_dict(orient="records")[:5]  # preview only in json
        # save full event table separately for union/stead
        if name in {"STEAD_K10", "UNION_K10", "STEAD_K5"}:
            _event_oracle_table(o).to_csv(out / f"phaseB_event_oracle_{name}.csv", index=False)

    # complementarity at K=5 and K=10 for three tols
    comp_rows = []
    comp_summaries = {}
    for k in (5, 10):
        for tol in (0.1, 0.2, 0.5):
            tab = complementarity_table(stead, ida, meta, k=k, tol_s=tol)
            comp_rows.append(tab)
            comp_summaries[f"K{k}_tol{tol}"] = summarize_complement(tab)
    comp_all = pd.concat(comp_rows, ignore_index=True)
    comp_all.to_csv(out / "phaseB_candidate_complementarity.csv", index=False)

    # Does ID-A add correct peaks absent from STEAD K=10?
    c10 = complementarity_table(stead, ida, meta, k=10, tol_s=0.5)
    ida_only = c10[c10["class"] == "ida_only_correct"]
    # vs STEAD K=10: ida_only means ID-A has correct and STEAD K=10 does not
    ida_adds_beyond_stead_k10 = {
        "tol": 0.5,
        "k_compared": 10,
        "n_ida_only_correct": int(len(ida_only)),
        "rate": float(len(ida_only) / max(len(c10), 1)),
        "n_events": int(ida_only["event_id"].nunique()) if len(ida_only) else 0,
    }

    # overlap rate among candidates (approx): fraction of STEAD K5 peaks matched in IDA K5 within 0.05s
    overlap_hits = 0
    overlap_tot = 0
    s5 = topk_from_cache(stead, 5)
    i5 = topk_from_cache(ida, 5)
    i5_map = {
        str(tn): g["candidate_sample"].to_numpy(dtype=float)
        for tn, g in i5.groupby(i5["trace_name"].astype(str), sort=False)
    }
    for tn, sg in s5.groupby(s5["trace_name"].astype(str), sort=False):
        isamps = i5_map.get(str(tn), np.array([], dtype=float))
        sr = float(sg["sampling_rate_hz"].iloc[0])
        for samp in sg["candidate_sample"].to_numpy(dtype=float):
            overlap_tot += 1
            if isamps.size and float(np.min(np.abs(isamps - samp) / sr)) <= 0.05:
                overlap_hits += 1
    candidate_overlap_rate = overlap_hits / max(overlap_tot, 1)

    # top-1 diagnosis
    # take rank0 from caches
    def top1_map(df: pd.DataFrame) -> pd.Series:
        g = df.sort_values("candidate_rank").groupby("trace_name").first()
        # prefer stored top1_s_sample if present
        if "top1_s_sample" in g.columns:
            return g["top1_s_sample"]
        return g["candidate_sample"]

    stead_top1 = meta.set_index("trace_name")
    # align
    st1 = top1_map(stead).reindex(meta["trace_name"].astype(str)).to_numpy(dtype=float)
    it1 = top1_map(ida).reindex(meta["trace_name"].astype(str)).to_numpy(dtype=float)
    true = meta["s_arrival_sample"].to_numpy(dtype=float)
    sr = meta["sampling_rate_hz"].to_numpy(dtype=float)
    top1 = {
        "STEAD": top1_diagnosis(st1, true, sr),
        "IDA": top1_diagnosis(it1, true, sr),
    }
    # fix/break counts @0.5
    stead_ok = (np.isfinite(st1) & np.isfinite(true) & (np.abs(st1 - true) / sr <= 0.5))
    ida_ok = (np.isfinite(it1) & np.isfinite(true) & (np.abs(it1 - true) / sr <= 0.5))
    top1["ida_fixes_stead"] = int((~stead_ok & ida_ok).sum())
    top1["ida_breaks_stead"] = int((stead_ok & ~ida_ok).sum())
    top1["disagreement_rate"] = float(np.mean(np.abs(st1 - it1) / sr > 0.05))

    # bootstrap: need per-trace metrics for each set
    def boot_pair(name_a: str, name_b: str, tol: float) -> dict:
        oa, ob = oracle_raw[name_a], oracle_raw[name_b]
        ha = per_trace_hit(oa["oracle_predictions"], oa["true_samples"], oa["sampling_rates"], tol)
        hb = per_trace_hit(ob["oracle_predictions"], ob["true_samples"], ob["sampling_rates"], tol)
        return event_level_bootstrap_delta(oa["event_ids"], ha, hb, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED)

    def boot_p95(name_a: str, name_b: str) -> dict:
        # Event-level bagging of closest-candidate AE P95 (B − A). Pre-index events for O(reps·N) work.
        oa, ob = oracle_raw[name_a], oracle_raw[name_b]
        ae_a = np.abs((oa["oracle_predictions"] - oa["true_samples"]) / oa["sampling_rates"])
        ae_b = np.abs((ob["oracle_predictions"] - ob["true_samples"]) / ob["sampling_rates"])
        eids = oa["event_ids"]
        uniq = np.unique(eids)
        ev_idx: dict[str, np.ndarray] = {}
        for i, e in enumerate(eids):
            ev_idx.setdefault(str(e), []).append(i)
        ev_idx = {e: np.asarray(ix, dtype=np.int64) for e, ix in ev_idx.items()}
        uniq_list = [str(e) for e in uniq]
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        boots = np.empty(BOOTSTRAP_REPS, dtype=float)
        for r in range(BOOTSTRAP_REPS):
            sampled = rng.choice(uniq_list, size=len(uniq_list), replace=True)
            sel = np.concatenate([ev_idx[e] for e in sampled])
            aa = ae_a[sel]
            bb = ae_b[sel]
            aa = aa[np.isfinite(aa)]
            bb = bb[np.isfinite(bb)]
            pa = float(np.percentile(aa, 95)) if aa.size else np.nan
            pb = float(np.percentile(bb, 95)) if bb.size else np.nan
            boots[r] = pb - pa
        aa = ae_a[np.isfinite(ae_a)]
        bb = ae_b[np.isfinite(ae_b)]
        d0 = float(np.percentile(bb, 95) - np.percentile(aa, 95))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return {
            "mean_delta": d0,
            "ci95": [float(lo), float(hi)],
            "n_events": int(len(uniq)),
            "n_traces": int(len(eids)),
            "direction": "b_lower_p95" if d0 < 0 else "b_higher_p95",
            "reps": BOOTSTRAP_REPS,
            "seed": BOOTSTRAP_SEED,
        }

    bootstrap = {
        "union_k10_vs_stead_k10_recall@0.1": boot_pair("STEAD_K10", "UNION_K10", 0.1),
        "union_k10_vs_stead_k10_recall@0.5": boot_pair("STEAD_K10", "UNION_K10", 0.5),
        "ida_k5_vs_stead_k5_recall@0.1": boot_pair("STEAD_K5", "IDA_K5", 0.1),
        "ida_k5_vs_stead_k5_recall@0.5": boot_pair("STEAD_K5", "IDA_K5", 0.5),
        "stead_k10_vs_stead_k5_recall@0.1": boot_pair("STEAD_K5", "STEAD_K10", 0.1),
        "stead_k10_vs_stead_k5_recall@0.5": boot_pair("STEAD_K5", "STEAD_K10", 0.5),
        "union_k10_vs_stead_k10_closest_p95": boot_p95("STEAD_K10", "UNION_K10"),
        "note": "event-level paired bootstrap; not trace-iid",
    }
    # miss rate bootstrap (1 - has any cand with finite oracle pred) — approx 1-recall of having pred
    def miss_vec(o):
        return (~np.isfinite(o["oracle_predictions"]) & np.isfinite(o["true_samples"])).astype(float)

    bootstrap["union_vs_stead_k10_miss_rate"] = event_level_bootstrap_delta(
        oracle_raw["STEAD_K10"]["event_ids"],
        miss_vec(oracle_raw["STEAD_K10"]),
        miss_vec(oracle_raw["UNION_K10"]),
        reps=BOOTSTRAP_REPS,
        seed=BOOTSTRAP_SEED,
    )

    # noise audit
    noise_audit = {"skipped": True}
    if not args.skip_noise and (ROOT / args.noise_stead).exists() and (ROOT / args.noise_ida).exists():
        ns = pd.read_parquet(ROOT / args.noise_stead)
        ni = pd.read_parquet(ROOT / args.noise_ida)
        nm = pd.read_csv(ROOT / args.noise_manifest)

        def noise_stats(df: pd.DataFrame, thr: float = 0.1) -> dict:
            # FPR: top1 prob above thr counts as false pick
            g = df.sort_values("candidate_rank").groupby("trace_name").first()
            s_fpr = float((g["top1_s_prob"] >= thr).mean()) if "top1_s_prob" in g else float("nan")
            p_fpr = float((g["top1_p_prob"] >= thr).mean()) if "top1_p_prob" in g else float("nan")
            n_cand = df.groupby("trace_name").size()
            any_cand = float((n_cand >= 1).mean())
            return {
                "n_traces": int(g.shape[0]),
                "S_FPR_top1_prob_ge_0.1": s_fpr,
                "P_FPR_top1_prob_ge_0.1": p_fpr,
                "mean_n_s_candidates": float(n_cand.mean()),
                "frac_with_ge1_s_candidate": any_cand,
                "top1_s_prob_mean": float(g["top1_s_prob"].mean()) if "top1_s_prob" in g else None,
                "top1_s_prob_p95": float(g["top1_s_prob"].quantile(0.95)) if "top1_s_prob" in g else None,
                "mean_n_cand_k5": float(topk_from_cache(df, 5).groupby("trace_name").size().mean()),
                "mean_n_cand_k10": float(topk_from_cache(df, 10).groupby("trace_name").size().mean()),
            }

        noise_audit = {
            "skipped": False,
            "manifest": str(args.noise_manifest),
            "n_manifest": int(len(nm)),
            "STEAD": noise_stats(ns),
            "IDA": noise_stats(ni),
        }
        # union cand count on noise
        nu = union_candidates(ns, ni, stead_k=5, ida_k=5, max_union=10)
        noise_audit["UNION_mean_n"] = float(nu.groupby("trace_name").size().mean()) if len(nu) else 0.0
        noise_audit["noise_candidate_increase"] = {
            "ida_minus_stead_mean_n_k10": noise_audit["IDA"]["mean_n_cand_k10"] - noise_audit["STEAD"]["mean_n_cand_k10"],
            "union_minus_stead_mean_n": noise_audit["UNION_mean_n"] - noise_audit["STEAD"]["mean_n_cand_k10"],
        }

    save_json(oracles, out / "phaseB_candidate_oracle.json")
    save_json(bootstrap, out / "phaseB_bootstrap.json")
    save_json(noise_audit, out / "phaseB_noise_audit.json")
    save_json(
        {
            "complementarity_summaries": comp_summaries,
            "ida_adds_beyond_stead_k10": ida_adds_beyond_stead_k10,
            "candidate_overlap_rate_steadK5_in_idaK5_0.05s": candidate_overlap_rate,
            "top1": top1,
        },
        out / "phaseB_complementarity_summary.json",
    )

    # ---- decision ----
    u05 = oracles["UNION_K10"]["oracle_f1@0.5"]
    s10_05 = oracles["STEAD_K10"]["oracle_f1@0.5"]
    s5_05 = oracles["STEAD_K5"]["oracle_f1@0.5"]
    delta_u_s10 = u05 - s10_05
    delta_u_s5 = u05 - s5_05
    boot05 = bootstrap["union_k10_vs_stead_k10_recall@0.5"]
    ci = boot05["ci95"]
    stable = (delta_u_s10 > 0 and ci[0] > 0) or (abs(delta_u_s10) < 1e-6)
    # tighten: direction stable if CI excludes 0 in same direction as mean
    direction_stable = (boot05["mean_delta"] >= 0 and ci[0] >= 0) or (boot05["mean_delta"] <= 0 and ci[1] <= 0)

    ida_only_rate = comp_summaries["K10_tol0.5"]["ida_only_correct"]["rate"]
    conc = comp_summaries["K10_tol0.5"].get("ida_only_concentration", {})
    concentrated = conc.get("frac_traces_in_top10_events", 1.0) > 0.5 and ida_only_rate > 0

    u01 = oracles["UNION_K10"]["oracle_f1@0.1"]
    s10_01 = oracles["STEAD_K10"]["oracle_f1@0.1"]
    fine_not_gone = (u01 - s10_01) > -0.005  # at least not fully vanished / mild

    noise_bad = False
    noise_increase = None
    if not noise_audit.get("skipped"):
        noise_increase = noise_audit.get("noise_candidate_increase")
        # unacceptable: mean candidates explode >2x or S FPR jumps >0.2 absolute
        if noise_audit["IDA"]["mean_n_cand_k10"] > 2.0 * max(noise_audit["STEAD"]["mean_n_cand_k10"], 1e-6):
            noise_bad = True
        if (noise_audit["IDA"]["S_FPR_top1_prob_ge_0.1"] - noise_audit["STEAD"]["S_FPR_top1_prob_ge_0.1"]) > 0.2:
            noise_bad = True

    union_n_ok = bool(union.groupby("trace_name").size().max() <= 10)

    decision = None
    keep = False
    if delta_u_s10 < 0.01 and delta_u_s5 >= 0.01:
        decision = "gain_explained_by_larger_candidate_budget"
        keep = False
    elif noise_bad or not ida_valid.get("ok"):
        decision = "discard_ida_noise_or_instability"
        keep = False
    elif delta_u_s10 >= 0.01 and direction_stable and (not concentrated) and union_n_ok and fine_not_gone:
        decision = "keep_ida_for_union"
        keep = True
    elif delta_u_s10 < 0.01:
        decision = "discard_ida_no_complementarity"
        keep = False
    else:
        decision = "inconclusive_insufficient_eval"
        keep = False

    # recommended K
    if keep:
        src = "UNION_STEAD5_IDA5"
        rec_k = 10
    else:
        src = "STEAD"
        gain_k = oracles["STEAD_K10"]["oracle_f1@0.5"] - oracles["STEAD_K5"]["oracle_f1@0.5"]
        rec_k = 10 if gain_k >= 0.005 else 5

    verdict = {
        "eval_scope": man_json.get("sampling", "all_s_labelled"),
        "eval_events": int(man_json.get("n_events", meta["event_id"].nunique())),
        "eval_s_traces": int(man_json.get("n_s_labelled_traces", len(meta))),
        "manifest_sha256": man_json.get("csv_sha256"),
        "stead_cache_sha256": sha256_file(ROOT / args.stead_cache),
        "ida_cache_sha256": sha256_file(ROOT / args.ida_cache),
        "stead_k5_oracle_f1_01": oracles["STEAD_K5"]["oracle_f1@0.1"],
        "stead_k5_oracle_f1_05": oracles["STEAD_K5"]["oracle_f1@0.5"],
        "stead_k10_oracle_f1_05": oracles["STEAD_K10"]["oracle_f1@0.5"],
        "ida_k5_oracle_f1_05": oracles["IDA_K5"]["oracle_f1@0.5"],
        "ida_k10_oracle_f1_05": oracles["IDA_K10"]["oracle_f1@0.5"],
        "union_k10_oracle_f1_01": oracles["UNION_K10"]["oracle_f1@0.1"],
        "union_k10_oracle_f1_05": oracles["UNION_K10"]["oracle_f1@0.5"],
        "union_vs_stead_k10_delta_f1_05": float(delta_u_s10),
        "union_vs_stead_k5_delta_f1_05": float(delta_u_s5),
        "union_vs_stead_k10_ci": boot05["ci95"],
        "ida_only_correct_rate_05": ida_only_rate,
        "noise_candidate_increase": noise_increase,
        "ida_checkpoint_valid": bool(ida_valid.get("ok")),
        "keep_ida_as_secondary_source": bool(keep),
        "recommended_candidate_source": src,
        "recommended_k": int(rec_k),
        "decision": decision,
        "ranker_may_start": False,
        "confirm_remains_sealed": True,
        "gates": {
            "union_vs_stead_k10_delta_ge_0.01": delta_u_s10 >= 0.01,
            "bootstrap_direction_stable": direction_stable,
            "ida_only_not_concentrated": not concentrated,
            "union_n_le_10": union_n_ok,
            "noise_ok": not noise_bad,
            "fine_tol_not_vanished": fine_not_gone,
        },
        "top1_summary": top1,
    }
    save_json(verdict, out / "phaseB_final_verdict.json")

    # markdown report
    md = f"""# Stage 6 Phase B — Candidate Oracle Report

**Decision:** `{decision}`  
**Keep ID-A as secondary source:** `{keep}`  
**Recommended candidate source / K:** `{src}` / `{rec_k}`  
**Ranker may start:** `false`  
**Confirm sealed:** `true`

## Eval scope

- Events: {verdict['eval_events']}
- S-labelled traces: {verdict['eval_s_traces']}
- Manifest SHA256: `{verdict['manifest_sha256']}`
- Sampling: {verdict['eval_scope']}

## Oracle F1 (closest candidate; ≈ hit-rate when misses≈0)

| Set | F1@0.1 | F1@0.5 | miss rate | closest P95 |
|--|--:|--:|--:|--:|
| STEAD K=5 | {oracles['STEAD_K5']['oracle_f1@0.1']:.4f} | {oracles['STEAD_K5']['oracle_f1@0.5']:.4f} | {oracles['STEAD_K5']['candidate_miss_rate']:.4f} | {oracles['STEAD_K5']['closest_candidate_p95']:.4f} |
| STEAD K=10 | {oracles['STEAD_K10']['oracle_f1@0.1']:.4f} | {oracles['STEAD_K10']['oracle_f1@0.5']:.4f} | {oracles['STEAD_K10']['candidate_miss_rate']:.4f} | {oracles['STEAD_K10']['closest_candidate_p95']:.4f} |
| ID-A K=5 | {oracles['IDA_K5']['oracle_f1@0.1']:.4f} | {oracles['IDA_K5']['oracle_f1@0.5']:.4f} | {oracles['IDA_K5']['candidate_miss_rate']:.4f} | {oracles['IDA_K5']['closest_candidate_p95']:.4f} |
| ID-A K=10 | {oracles['IDA_K10']['oracle_f1@0.1']:.4f} | {oracles['IDA_K10']['oracle_f1@0.5']:.4f} | {oracles['IDA_K10']['candidate_miss_rate']:.4f} | {oracles['IDA_K10']['closest_candidate_p95']:.4f} |
| Union K≤10 | {oracles['UNION_K10']['oracle_f1@0.1']:.4f} | {oracles['UNION_K10']['oracle_f1@0.5']:.4f} | {oracles['UNION_K10']['candidate_miss_rate']:.4f} | {oracles['UNION_K10']['closest_candidate_p95']:.4f} |

**Fairness:** Union vs **STEAD K=10** ΔF1@0.5 = **{delta_u_s10:+.4f}** (vs STEAD K=5 = {delta_u_s5:+.4f}).

Oracle F1 definition note: {oracles['STEAD_K5']['oracle_f1_vs_hitrate_note']}

## Complementarity (STEAD vs ID-A at K=10, ±0.5 s)

| class | traces | rate |
|--|--:|--:|
| both_correct | {comp_summaries['K10_tol0.5']['both_correct']['n_traces']} | {comp_summaries['K10_tol0.5']['both_correct']['rate']:.4f} |
| stead_only_correct | {comp_summaries['K10_tol0.5']['stead_only_correct']['n_traces']} | {comp_summaries['K10_tol0.5']['stead_only_correct']['rate']:.4f} |
| ida_only_correct | {comp_summaries['K10_tol0.5']['ida_only_correct']['n_traces']} | {comp_summaries['K10_tol0.5']['ida_only_correct']['rate']:.4f} |
| neither_correct | {comp_summaries['K10_tol0.5']['neither_correct']['n_traces']} | {comp_summaries['K10_tol0.5']['neither_correct']['rate']:.4f} |

ID-A-only correct beyond STEAD K=10: {ida_adds_beyond_stead_k10}

## Bootstrap (event-level, {BOOTSTRAP_REPS} reps)

Union vs STEAD K=10 recall@0.5: mean Δ={boot05['mean_delta']:+.4f}, 95% CI={boot05['ci95']}

## Top-1 (same list)

- STEAD detected_ae_p95={top1['STEAD']['detected_ae_p95']:.3f}, F1@0.5={top1['STEAD']['f1@0.5s']:.4f}
- ID-A detected_ae_p95={top1['IDA']['detected_ae_p95']:.3f}, F1@0.5={top1['IDA']['f1@0.5s']:.4f}
- ID-A fixes STEAD: {top1['ida_fixes_stead']}; breaks: {top1['ida_breaks_stead']}; disagreement: {top1['disagreement_rate']:.4f}

## Gates

{json.dumps(verdict['gates'], indent=2)}

## Artifacts

- `artifacts/results/stage6/phaseB_candidate_oracle.json`
- `artifacts/results/stage6/phaseB_candidate_complementarity.csv`
- `artifacts/results/stage6/phaseB_bootstrap.json`
- `artifacts/results/stage6/phaseB_noise_audit.json`
- `artifacts/results/stage6/phaseB_final_verdict.json`
"""
    (ROOT / "reports/stage6/phaseB_candidate_oracle_report.md").write_text(md)
    print(json.dumps({"decision": decision, "delta_u_s10": delta_u_s10, "keep": keep, "rec_k": rec_k}, indent=2))


if __name__ == "__main__":
    main()
