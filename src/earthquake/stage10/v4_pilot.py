"""v4 3-epoch pilot gate. No 30-epoch continuation."""

from __future__ import annotations

import numpy as np

from earthquake.stage10.checkpoint_policy import MIN_DELTA_F1


def pilot_gate_v4(
    history,
    first500,
    kinds_per_epoch,
    bg_pseudo_n,
    nan_inf,
    silent_skips,
    ever_valid_best,
    divergence,
    confirm_read,
) -> tuple[bool, list[str]]:
    reasons = []
    if len(history) < 3:
        return False, ["need_3_virtual_epochs"]
    h0, h1, h2 = history[0], history[1], history[2]
    losses = [h["train_loss"] for h in history]
    if first500 is None or not np.isfinite(first500):
        reasons.append("no_first500")
    elif losses[-1] >= first500 * 0.98:
        reasons.append("train_loss_not_down_vs_first500")
    init_f1 = float(h0.get("init_s_f1_fixed0p2", 0))
    f1s = [float(h["val_s_f1_fixed0p2"]) for h in history]
    if f1s[2] <= init_f1 + MIN_DELTA_F1:
        reasons.append("epoch2_dev_f1_0p2_not_above_init")
    if f1s[2] < f1s[1]:
        reasons.append("epoch2_below_epoch1")
    ora0 = float(h0.get("init_oracle_acc@0.5s", 0))
    ora2 = float(h2.get("oracle_acc@0.5s", 0))
    if ora2 <= ora0:
        reasons.append("oracle_not_above_init")
    if any(h.get("all_n_collapse") for h in history):
        reasons.append("all_n_collapse")
    need = {"p_centered", "s_centered", "background"}
    for ep, kinds in enumerate(kinds_per_epoch):
        if not need.issubset(set(kinds)):
            reasons.append(f"missing_crop_kinds_epoch{ep}:{sorted(need - set(kinds))}")
    if bg_pseudo_n > 0:
        reasons.append("event_background_pseudo_n")
    if silent_skips > 0:
        reasons.append("silent_skip")
    if nan_inf or any(not np.isfinite(h["train_loss"]) for h in history):
        reasons.append("nan_inf")
    if not ever_valid_best:
        reasons.append("best_metric_not_above_init_plus_min_delta")
    if not (divergence or {}).get("stable", False):
        reasons.append("activation_param_or_loss_not_stable")
    if confirm_read:
        reasons.append("confirm_read")
    if h2.get("grad_norm", 0) <= 0 or not np.isfinite(h2.get("grad_norm", 0)):
        reasons.append("bad_grad")
    return (len(reasons) == 0), reasons


# --- Protocol amendment AFTER v4 original pilot ended (2026-09-02 05:20 local).
# This is NOT the v4 pre-registered gate. Original v4 required epoch2 >= epoch1.
# Future unstarted runs may replace that with paired-event-bootstrap F1
# non-inferiority at margin 0.01 (MIN_DELTA_F1 / full-dev stop-gate).
FUTURE_GATE_AMENDMENT_UTC = "2026-09-02T05:21:00+08:00"
FUTURE_F1_NONINFERIORITY_MARGIN = 0.01
FUTURE_GATE_NOT_V4_PREREGISTERED = True
