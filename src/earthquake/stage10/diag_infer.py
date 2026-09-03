"""Shared inference/diagnostics for threshold calibration and loss ablation."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from earthquake.stage10.dkpn_clean import dkpn_logits
from earthquake.stage10.phase_balanced_loss import partial_label_group_terms
from earthquake.stage10.threshold_calibration import (
    OFFICIAL_HEIGHT,
    event_bootstrap_f1,
    forced_choice_preds,
    pick_metrics_at_thr,
    select_threshold_on_calibration,
    window_prob_stats,
)


def collate_diag(batch):
    keys_t = ["x", "p_pos", "s_pos", "n_pos", "not_p", "not_s", "pad_mask", "vis_p", "vis_s", "p_c", "s_c"]
    out = {k: torch.stack([b[k] for b in batch]) for k in keys_t}
    out["crop_kind"] = [b["crop_kind"] for b in batch]
    out["trace_name"] = [b["trace_name"] for b in batch]
    out["event_id"] = [b["event_id"] for b in batch]
    out["is_noise"] = torch.tensor([b["is_noise"] for b in batch])
    return out


def force_s_centered(cat):
    c = cat.copy()
    c["crop_kind"] = "s_centered"
    return c


@torch.no_grad()
def infer_arrays(model, ds, device, *, batch_size: int = 8) -> dict[str, np.ndarray]:
    ld = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_diag)
    s_probs, p_probs, n_probs = [], [], []
    true_s, true_p, vis_s, vis_p, eids, names = [], [], [], [], [], []
    kinds = []
    for batch in ld:
        pr = torch.softmax(dkpn_logits(model, batch["x"].to(device)).float(), dim=1).cpu().numpy()
        for i in range(pr.shape[0]):
            s_probs.append(pr[i, 1])
            p_probs.append(pr[i, 0])
            n_probs.append(pr[i, 2])
            true_s.append(float(batch["s_c"][i]))
            true_p.append(float(batch["p_c"][i]))
            vis_s.append(bool(batch["vis_s"][i] > 0.5 and float(batch["s_c"][i]) >= 0))
            vis_p.append(bool(batch["vis_p"][i] > 0.5 and float(batch["p_c"][i]) >= 0))
            eids.append(str(batch["event_id"][i]))
            names.append(str(batch["trace_name"][i]))
            kinds.append(batch["crop_kind"][i])
    return {
        "s_probs": np.stack(s_probs) if s_probs else np.zeros((0, 1)),
        "p_probs": np.stack(p_probs) if p_probs else np.zeros((0, 1)),
        "n_probs": np.stack(n_probs) if n_probs else np.zeros((0, 1)),
        "true_s": np.asarray(true_s, dtype=float),
        "true_p": np.asarray(true_p, dtype=float),
        "vis_s": np.asarray(vis_s, dtype=bool),
        "vis_p": np.asarray(vis_p, dtype=bool),
        "event_id": np.asarray(eids),
        "trace_name": np.asarray(names),
        "crop_kind": np.asarray(kinds),
    }


def psn_distribution(arrs: dict[str, np.ndarray]) -> dict[str, float]:
    if arrs["s_probs"].size == 0:
        return {}
    p, s, n = arrs["p_probs"], arrs["s_probs"], arrs["n_probs"]
    return {
        "p_mean": float(p.mean()),
        "s_mean": float(s.mean()),
        "n_mean": float(n.mean()),
        "p_max_mean": float(p.max(axis=-1).mean()),
        "s_max_mean": float(s.max(axis=-1).mean()),
        "n_max_mean": float(n.max(axis=-1).mean()),
        "p_std": float(p.std()),
        "s_std": float(s.std()),
        "n_std": float(n.std()),
    }


def collapse_flag(dist: dict[str, float]) -> bool:
    if not dist:
        return False
    return bool(dist.get("n_mean", 0) > 0.95 and dist.get("s_max_mean", 1) < 0.02 and dist.get("p_max_mean", 1) < 0.02)


def summarize_split(arrs: dict[str, np.ndarray], *, selected_thr: float | None = None) -> dict[str, Any]:
    dist = psn_distribution(arrs)
    wstats = window_prob_stats(arrs["s_probs"], arrs["p_probs"], arrs["true_s"], arrs["true_p"], arrs["vis_s"], arrs["vis_p"])
    official = pick_metrics_at_thr(arrs["s_probs"], arrs["true_s"], arrs["vis_s"], OFFICIAL_HEIGHT)
    out: dict[str, Any] = {
        "n": int(len(arrs["true_s"])),
        "n_vis_s": int(arrs["vis_s"].sum()),
        "official_height_0.2": official,
        "psn": dist,
        **wstats,
        "all_n_collapse": collapse_flag(dist),
    }
    if selected_thr is not None:
        frozen = pick_metrics_at_thr(arrs["s_probs"], arrs["true_s"], arrs["vis_s"], selected_thr)
        pred = forced_choice_preds(arrs["s_probs"], selected_thr)
        boot = event_bootstrap_f1(pred, arrs["true_s"], arrs["vis_s"], arrs["event_id"], tol_samples=50, n_boot=500, seed=42)
        out["at_selected_thr"] = frozen
        out["bootstrap_f1@0.5s_selected"] = boot
        pred02 = forced_choice_preds(arrs["s_probs"], OFFICIAL_HEIGHT)
        out["bootstrap_f1@0.5s_height0.2"] = event_bootstrap_f1(
            pred02, arrs["true_s"], arrs["vis_s"], arrs["event_id"], tol_samples=50, n_boot=500, seed=42
        )
    return out


def calibration_audit(init_cal, init_eval, trained_cal, trained_eval) -> dict[str, Any]:
    """Same pre-registered protocol on init and trained. Never reselect on evaluation."""
    proto_init = select_threshold_on_calibration(init_cal["s_probs"], init_cal["true_s"], init_cal["vis_s"])
    proto_tr = select_threshold_on_calibration(trained_cal["s_probs"], trained_cal["true_s"], trained_cal["vis_s"])
    return {
        "protocol": proto_init["protocol"],
        "same_protocol_for_init_and_trained": proto_init["protocol"] == proto_tr["protocol"],
        "never_reselect_on_evaluation": True,
        "init": {
            "threshold_sweep_calibration": proto_init["grid"],
            "selected_thr": proto_init["selected_thr"],
            "selected_from_eligible_picks_cap": proto_init["selected_from_eligible_picks_cap"],
            "calibration": summarize_split(init_cal, selected_thr=proto_init["selected_thr"]),
            "evaluation_frozen_thr": summarize_split(init_eval, selected_thr=proto_init["selected_thr"]),
            "official_height_0.2_evaluation": pick_metrics_at_thr(
                init_eval["s_probs"], init_eval["true_s"], init_eval["vis_s"], OFFICIAL_HEIGHT
            ),
        },
        "trained": {
            "threshold_sweep_calibration": proto_tr["grid"],
            "selected_thr": proto_tr["selected_thr"],
            "selected_from_eligible_picks_cap": proto_tr["selected_from_eligible_picks_cap"],
            "calibration": summarize_split(trained_cal, selected_thr=proto_tr["selected_thr"]),
            "evaluation_frozen_thr": summarize_split(trained_eval, selected_thr=proto_tr["selected_thr"]),
            "official_height_0.2_evaluation": pick_metrics_at_thr(
                trained_eval["s_probs"], trained_eval["true_s"], trained_eval["vis_s"], OFFICIAL_HEIGHT
            ),
        },
        "forbidden_comparison_trained_best_vs_init_fixed_0.2": False,
    }


def group_param_grad_norms(model, logits, means: dict[str, torch.Tensor]) -> dict[str, float]:
    params = [p for p in model.parameters() if p.requires_grad]
    out = {}
    for name, t in means.items():
        if t is None or (not torch.isfinite(t)):
            out[name] = 0.0
            continue
        if float(t.detach()) == 0.0 and t.grad_fn is None:
            out[name] = 0.0
            continue
        grads = torch.autograd.grad(t, params, retain_graph=True, allow_unused=True)
        acc = 0.0
        for g in grads:
            if g is not None:
                acc += float(g.detach().float().pow(2).sum().cpu())
        out[name] = acc ** 0.5
    g_phase = out.get("p", 0.0) + out.get("s", 0.0)
    g_n = out.get("certified_noise", 0.0) + out.get("complete_n", 0.0)
    out["phase_over_n"] = g_phase / max(g_n, 1e-12)
    return out


def token_mean_group_terms(logits, batch) -> dict[str, torch.Tensor]:
    """A-loss group contributions: (nll*w).sum()/total_denom — actual terms inside token-mean NLL."""
    g = partial_label_group_terms(
        logits,
        p_pos=batch["p_pos"],
        s_pos=batch["s_pos"],
        n_pos=batch["n_pos"],
        not_p=batch["not_p"],
        not_s=batch["not_s"],
        pad_mask=batch["pad_mask"],
        is_noise=batch["is_noise"],
    )
    denom = sum(g[k]["w"].sum() for k in g).clamp_min(1.0)
    return {k: (g[k]["nll"] * g[k]["w"]).sum() / denom for k in g}
