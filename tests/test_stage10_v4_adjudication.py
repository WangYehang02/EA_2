"""Adjudication helpers. No confirm, no training."""

from earthquake.stage10.paired_event_bootstrap import F1_NONINFERIORITY_MARGIN, adjudicate_e2_vs_e1
from earthquake.stage10.v4_pilot import FUTURE_F1_NONINFERIORITY_MARGIN, FUTURE_GATE_NOT_V4_PREREGISTERED, pilot_gate_v4


def test_margin_is_preregistered_0p01_not_0p006():
    assert F1_NONINFERIORITY_MARGIN == 0.01
    assert FUTURE_F1_NONINFERIORITY_MARGIN == 0.01
    assert FUTURE_GATE_NOT_V4_PREREGISTERED is True


def test_original_v4_gate_still_fails_on_epoch2_below_epoch1():
    h = [
        {"train_loss": 0.01, "val_s_f1_fixed0p2": 0.94, "init_s_f1_fixed0p2": 0.0, "init_oracle_acc@0.5s": 0.0, "oracle_acc@0.5s": 0.97, "all_n_collapse": False, "grad_norm": 0.2},
        {"train_loss": 0.007, "val_s_f1_fixed0p2": 0.963, "init_s_f1_fixed0p2": 0.0, "init_oracle_acc@0.5s": 0.0, "oracle_acc@0.5s": 0.986, "all_n_collapse": False, "grad_norm": 0.2},
        {"train_loss": 0.006, "val_s_f1_fixed0p2": 0.957, "init_s_f1_fixed0p2": 0.0, "init_oracle_acc@0.5s": 0.0, "oracle_acc@0.5s": 0.975, "all_n_collapse": False, "grad_norm": 0.2},
    ]
    kinds = [{"p_centered", "s_centered", "background"}] * 3
    ok, reasons = pilot_gate_v4(h, 0.16, kinds, 0, False, 0, True, {"stable": True}, False)
    assert ok is False
    assert "epoch2_below_epoch1" in reasons


def test_adjudicate_passes_small_f1_drop_inside_margin():
    point = {"f1@0.5s": -0.006, "oracle@0.5s": -0.011, "frac_no_s_peak": 0.004}
    boot = {"f1@0.5s": {"ci95_lo": -0.009, "ci95_hi": 0.002, "mean_delta": -0.006}}
    vs = {"f1@0.5s": {"ci95_lo": 0.9, "ci95_hi": 0.96, "mean_delta": 0.95}}
    ok, reasons, checks = adjudicate_e2_vs_e1(point=point, boot=boot, vs_init_e1=vs, vs_init_e2=vs, finite=True, confirm_read=False)
    assert ok, (reasons, checks)


def test_adjudicate_fails_if_ci_crosses_margin():
    point = {"f1@0.5s": -0.006, "oracle@0.5s": 0.0, "frac_no_s_peak": 0.0}
    boot = {"f1@0.5s": {"ci95_lo": -0.015, "ci95_hi": 0.002, "mean_delta": -0.006}}
    vs = {"f1@0.5s": {"ci95_lo": 0.9, "ci95_hi": 0.96, "mean_delta": 0.95}}
    ok, reasons, _ = adjudicate_e2_vs_e1(point=point, boot=boot, vs_init_e1=vs, vs_init_e2=vs, finite=True, confirm_read=False)
    assert ok is False
    assert "ci_lo_f1_gt_neg_margin" in reasons
