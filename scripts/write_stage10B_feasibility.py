#!/usr/bin/env python
"""Write Stage10B feasibility + update stage10_state after smoke/overfit."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/results/stage10"
CACHE = Path("/data/mnt_data/yehang/PSdetec/Earthquake_stage10")
REPORTS = ROOT / "reports/stage10"


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def main() -> None:
    reg = load(OUT / "stage10A_protocol_registry.json")
    smoke = load(CACHE / "dkpn_clean/smoke_seed42/smoke_report.json")
    overfit = load(CACHE / "dkpn_clean/overfit_seed42/overfit_report.json")
    thr = load(CACHE / "dkpn_clean/throughput_seed42/throughput_report.json")
    train_man = load(ROOT / "artifacts/results/stage6/full_training_manifest.json")
    frozen = load(OUT / "stage10A_frozen_hashes.json")

    # clean train counts (planned)
    n_ps = train_man["supervision"]["n_ps_both_traces"]
    n_all = train_man["supervision"]["n_picker_train_traces_all"]
    n_p_only = n_all - n_ps  # includes unlabeled? picker_train all - ps_both
    n_noise = int(0.2 * (n_ps + n_p_only))
    planned = {
        "picker_train_events": train_man["supervision"]["n_picker_train_events"],
        "ps_both": n_ps,
        "p_only_masked": n_p_only,
        "noise_ratio_0.2_count": n_noise,
        "n_train_total_planned": n_ps + n_p_only + n_noise,
        "supervision_policy": {
            "ps_both": "supervise P+S+N soft labels",
            "p_only": "supervise P+N only; S channel mask=0; NOT treated as noise",
            "noise": "real noise HDF5; soft target noise-dominant",
        },
        "normalization_stats_from": "picker_train only (per-window STD as DKPN annotate)",
        "init": "random (refuse INSTANCE pretrained)",
    }

    # Corrected ETA: CPU CF bottleneck; GPU remount required for production ETA
    tps_cpu = (thr or {}).get("measured", {}).get("traces_per_sec")
    n_train = planned["n_train_total_planned"]
    # Optimistic GPU with 8 CF workers + CUDA: assume 25–40 t/s (CF parallel + amp)
    # Conservative: 15 t/s
    eta = {}
    for name, tps in [("cpu_measured", tps_cpu or 2.8), ("gpu1_conservative_15tps", 15.0), ("gpu1_optimistic_30tps", 30.0)]:
        if not tps:
            continue
        sec = n_train / tps
        eta[name] = {
            "traces_per_sec": tps,
            "hours_per_epoch": sec / 3600,
            "hours_30ep": 30 * sec / 3600,
        }
    for ng, scale in [(4, 0.85 * 4), (8, 0.75 * 8)]:
        tps = 15.0 * scale  # from conservative 1gpu
        sec = n_train / tps
        eta[f"gpu{ng}_from_15tps_scaled"] = {
            "effective_tps": tps,
            "hours_per_epoch": sec / 3600,
            "hours_30ep": 30 * sec / 3600,
        }

    feas = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "smoke": smoke,
        "overfit": overfit,
        "throughput_raw_cpu": thr,
        "eta_revised": eta,
        "planned_train_counts": planned,
        "gpu_note": "All 8 GPUs occupied at Stage10A/B; smoke/overfit/throughput run on CPU. Seed42 waiter armed after smoke_ok.",
        "cf_bottleneck": "DKPN CF ~50–70ms/crop on CPU; production needs CF cache under /data or many workers before 8-GPU DDP.",
        "official_weights_main": False,
        "sota_claim_allowed": False,
        "confirm_used_for_tuning": False,
        "frozen_lock_unchanged": bool((frozen or {}).get("method_lock_matches_declared")),
        "stop_gate_next": "DKPN seed42 vs strongest waveform-only Stage6/7/9 DEV baseline: ΔF1@0.5≥+0.01 OR (F1 not down & P95≥0.15s better) OR UNION oracle +0.01; else rejected_candidate_source",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stage10B_dkpn_feasibility.json").write_text(json.dumps(feas, indent=2) + "\n")
    REPORTS.mkdir(parents=True, exist_ok=True)
    md = f"""# Stage 10B — Clean DKPN feasibility

**UTC:** {feas['created_utc']}

## Smoke
- smoke_ok: **{smoke and smoke.get('smoke_ok')}**
- device: `{smoke and smoke.get('device')}`
- mean_loss (20 steps): {smoke and smoke.get('train_stats',{}).get('mean_loss')}
- output_finite: {smoke and smoke.get('output_finite')}

## Overfit (32 traces, cached batch, localization)
- overfit_ok: **{overfit and overfit.get('overfit_ok')}**
- loss: {overfit and overfit.get('init_loss')} → {overfit and overfit.get('final_loss')} (drop {overfit and round(overfit.get('loss_drop_frac',0),3)})
- S-peak ±0.2s acc: {overfit and overfit.get('s_peak_acc_init')} → {overfit and overfit.get('s_peak_acc_final')}

## Planned clean train counts
- P+S: **{n_ps}**
- P-only (masked): **{n_p_only}**
- noise @0.2: **{n_noise}**
- total: **{planned['n_train_total_planned']}**

## ETA (revised)
CPU measured E2E (CF+train) ≈ {tps_cpu} t/s → impractical for full train.
Conservative 1×GPU with parallel CF ≈ 15 t/s → ~{eta.get('gpu1_conservative_15tps',{}).get('hours_30ep')} h / 30 ep.
See `stage10B_dkpn_feasibility.json` for 4/8 GPU scaled ETA.
**Must remount on free GPU before trusting production ETA.**

## Policy
- Official INSTANCE DKPN weights: **not** for main table
- Confirm: **not** used for Stage10 tuning
- SOTA claim: **forbidden** (`exact_lftnet_protocol_reproducible=false`)

## Next stop gate
{feas['stop_gate_next']}
"""
    (REPORTS / "stage10B_dkpn_feasibility.md").write_text(md)

    state = {
        "stage": "10B",
        "status": "SMOKE_OVERFIT_DONE_WAITING_GPU_SEED42",
        "smoke_ok": bool(smoke and smoke.get("smoke_ok")),
        "overfit_ok": bool(overfit and overfit.get("overfit_ok")),
        "seed42": "WAITING_FREE_GPU",
        "sota_claim_allowed": False,
        "exact_lftnet_protocol_reproducible": False,
        "confirm_usable_for_stage10_tuning": False,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "next_stop_gate": feas["stop_gate_next"],
    }
    (OUT / "stage10_state.json").write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
