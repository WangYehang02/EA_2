# External adapter tests (synthetic + old dev)

all_ok=True

```json
{
  "time_sample_roundtrip": {
    "ok": true,
    "sr": 100.0,
    "tau_s": 25.3,
    "sample": 1530.0,
    "back_tau": 25.3
  },
  "channel_order": {
    "expected_components_doc": [
      "Z",
      "N",
      "E"
    ],
    "note": "Adapter must map BH?/HH?/EH? to ZNE without using manual S to choose window.",
    "ok": true
  },
  "schema_old_dev": {
    "missing_columns": [],
    "scalar_feature_names": [
      "prob",
      "stead_prob",
      "ida_prob",
      "fixed_score",
      "resid_s",
      "resid_sp",
      "delta_sp",
      "src_stead",
      "src_ida",
      "src_both"
    ],
    "fulldev_feature_names_match": true,
    "X1_shape": [
      200,
      10
    ],
    "diff_shape": [
      200,
      10
    ],
    "single_candidate_fallback_ok": true,
    "label_isolation_C_ok": true,
    "fixed_equals_c1": true,
    "ok": true
  },
  "missing_input_rules": {
    "missing_waveform": "record reason=MISSING_WAVEFORM; do not silently drop after seeing labels",
    "missing_metadata": "record reason=MISSING_METADATA; exclude only by pre-registered rule",
    "single_candidate": "output c1; keep in overall metrics; exclude from switch-risk denominators",
    "history_miss_on_new_path": "use registered train-only fallback (no test-label history fill)",
    "ok": true
  },
  "synthetic": {
    "n": 50,
    "n_switched": 0,
    "ok": true
  },
  "all_ok": true
}
```

Note: These tests do **not** score BSI 2021–22 model performance.
