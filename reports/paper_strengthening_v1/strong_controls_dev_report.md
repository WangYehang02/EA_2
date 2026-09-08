# Paper strengthening v1 — strong controls (dev)

- Main strong control (calibration-selected): **C_base_tau_c1c2**
- Detail: `{"config_id": "sig0.25_lam4.0", "lambda_history": 4.0, "sigma_s": 0.25}`
- Label isolation smoke: **True**

## Calibration selection

| Family | F1@0.5 | F1@0.1 | Config |
| --- | ---: | ---: | --- |
| B resid | 0.9324 | 0.6385 | lam=1,σ=0.5 |
| C best | 0.9365 | 0.6504 | sig0.25_lam4.0 |
| D best | 0.9275 | 0.6382 | C0.1_tau0.8 |

## A–E F1@0.5 on historical splits (NOT independent)

| method | calibration | fulldev_phaseB | heldout_eval |
| --- | ---: | ---: | ---: |
| A_fixed | 0.9161 | 0.8668 | 0.9196 |
| B_resid | 0.9324 | 0.8732 | 0.9350 |
| C_best | 0.9365 | 0.8800 | 0.9384 |
| C_registered_sig0p5_lam2 | 0.9357 | 0.8782 | 0.9375 |
| D_best | 0.9275 | 0.8701 | 0.9306 |
| E_scalar | 0.9391 | 0.8805 | 0.9403 |

## Primary comparison: E_scalar − main_strong_control

- **calibration**: Δ=+0.0026, boot mean=+0.0026, CI=[+0.0014, +0.0042] (calibration_used_for_control_selection_not_independent_test)
- **heldout_eval**: Δ=+0.0019, boot mean=+0.0019, CI=[+0.0007, +0.0033] (historical_research_split_not_independent)
- **fulldev_phaseB**: Δ=+0.0005, boot mean=+0.0005, CI=[+0.0000, +0.0011] (historical_research_split_not_independent)

## Independent validation

**INDEPENDENT_VALIDATION_NOT_COMPLETED** — see DATA_REQUIREMENTS.md / PAPER_JUDGMENT.md.

