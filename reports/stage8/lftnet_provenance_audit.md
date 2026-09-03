# Stage 8 — LFTNet Provenance Audit

**Created (UTC):** 2026-08-19T13:39:03.429043+00:00  
**Provenance class:** `E. incomplete_or_unusable`  
**Auto-continue full eval:** `False`

## Path resolution

- User path `~/baseline`: **does not exist** on this server.
- Actual drop found at: `/home/yehang/yehang/Earthquake/baseline/LFTNet`
- Realpath used for INSTANCE data (unchanged): via project symlink `data/INSTANCE` → do **not** copy ~156GB HDF5.

## Folder hint

- Directory name: `LFTNet`
- Interpretation: Likely GitHub archive fragment Tianjiyu1/LFTNet commit prefix 79994f6 (not verified as full release)
- Name alone is NOT sufficient for official_complete_release

## File tree

```
LFTNet/
  README.md
  se-tcn-Eqt_utils.py
```

## Git

```
{
  "is_git_repo": false
}
```

## Component checklist

| Component | Present? |
|--|--|
| LFTNet_model_class | `True` |
| RSDB | `False` |
| MSSE_TCN | `True` |
| detection_P_S_multitask | `True` |
| softmax_or_sigmoid | `True` |
| input_length_documented | `False` |
| sampling_rate_documented | `True` |
| component_order_documented | `False` |
| normalization_documented | `True` |
| label_order_documented | `True` |
| checkpoint_file_present | `False` |
| checkpoint_training_metadata | `False` |
| peak_extraction_code | `False` |
| threshold_config | `False` |
| INSTANCE_eval_manifest | `False` |
| paper_10k_to_20k_segment_code | `False` |
| train_script | `False` |
| infer_script | `False` |
| license_file | `False` |
| readme | `True` |
| requirements | `False` |

## Evidence for class `E. incomplete_or_unusable`

- only 2 files under drop; no checkpoint
- fragment contains MSSE-TCN-like blocks and detector/P/S heads (cred2 class)
- missing imports (keras layers, SeqSelfAttention, FeedForward, f1) — file not self-contained
- README.md is a requirements list (tensorflow~=2.5.0), not project documentation
- no LICENSE, no train/infer scripts, no configs, no paper 10k/20k construction
- directory name suggests GitHub Tianjiyu1/LFTNet @79994f6 but is not a git checkout

## Dependencies vs conda `PS`

- Drop README lists **TensorFlow ~2.5 / Keras ~2.3 / numpy ~1.19**.
- `PS` has **PyTorch 2.5**; **TensorFlow/Keras absent**.
- Recommendation: **do not** mutate `PS`. Create `LFTNet_eval` only after a complete official release is available.

## Gate (must all pass to continue)

| Gate | Status |
|--|--|
| code_complete | `False` |
| checkpoint_verifiable | `False` |
| no_obvious_confirm_leakage | `unknown_no_checkpoint` |
| utc_sample_alignment_passed | `False` |
| smoke_reasonable | `False` |

**Stop reason:** `incomplete_or_unusable_drop_no_checkpoint`

## Paper numbers (NOT Stage-6 ranking)

Paper (Guo et al. 2025) reports on *their* INSTANCE protocol: S F1@0.5=0.844, P=0.822, R=0.867, MAE=0.232 s.  
These **must not** be ranked against Stage-6/7A confirm metrics.

## Next action for user

Please place the **full** Zenodo release (`10.5281/zenodo.15710535`) or complete GitHub tree **including official pretrained checkpoint + LICENSE + infer/train scripts + eval manifests** under a readable path (e.g. recreate `~/baseline/LFTNet`).  
Until then: **no training, no confirm comparator run, no main-method change**.
