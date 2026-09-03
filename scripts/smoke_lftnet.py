#!/usr/bin/env python
"""Stage 8 smoke — refuses to invent weights; documents gate failure."""

from __future__ import annotations

import ast
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.stage6.phaseB import sha256_file  # noqa: E402


def main() -> int:
    out = ROOT / "artifacts/results/stage8"
    reports = ROOT / "reports/stage8"
    out.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((out / "lftnet_release_manifest.json").read_text())
    primary = Path(manifest["resolved_primary_root"]) if manifest.get("resolved_primary_root") else None
    utils = primary / "se-tcn-Eqt_utils.py" if primary else None

    parse_ok = False
    parse_err = None
    missing_names = []
    if utils and utils.exists():
        src = utils.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(src)
            parse_ok = True
            # referenced but undefined at module level (heuristic)
            defined = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))}
            # also assignments
            for n in tree.body:
                if isinstance(n, ast.Assign):
                    for t in n.targets:
                        if isinstance(t, ast.Name):
                            defined.add(t.id)
            needed = [
                "BatchNormalization",
                "Activation",
                "SpatialDropout1D",
                "SeparableConv1D",
                "add",
                "SeqSelfAttention",
                "Add",
                "LayerNormalization",
                "FeedForward",
                "GlobalAveragePooling1D",
                "Dense",
                "Multiply",
                "Conv1D",
                "Concatenate",
                "MaxPooling1D",
                "UpSampling1D",
                "Cropping1D",
                "Model",
                "Adam",
                "Precision",
                "Recall",
                "MeanAbsoluteError",
                "K",
                "keras",
                "f1",
            ]
            missing_names = [n for n in needed if n not in defined and n not in src.split("import")[0]]
            # better: names used but never imported — file has ZERO import statements
            has_import = any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in tree.body)
            if not has_import:
                missing_names = ["NO_IMPORT_STATEMENTS"] + needed
        except SyntaxError as e:
            parse_err = repr(e)

    smoke = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "blocked",
        "reason": "incomplete_release_no_checkpoint_not_importable",
        "provenance_class": manifest.get("provenance_class"),
        "checkpoint_loaded": False,
        "random_init_refused": True,
        "parse_ok": parse_ok,
        "parse_err": parse_err,
        "missing_symbols_or_imports": missing_names[:40],
        "utils_sha256": sha256_file(utils) if utils and utils.exists() else None,
        "tests": {
            "input_shape": "not_run",
            "sampling_rate_100hz": "not_run",
            "component_order_ENZ_remap": "not_run",
            "output_class_order": "not_run",
            "output_length": "not_run",
            "sample_offset": "not_run",
            "60s_120s_window_map": "not_run",
            "UTC_remap": "not_run",
            "batch_consistency": "not_run",
            "cpu_gpu_consistency": "not_run",
            "determinism": "not_run",
        },
        "alignment_gate": "FAIL_STOP_NO_FULL_INFER",
        "sota_claim_allowed": False,
    }
    (out / "lftnet_smoke_metrics.json").write_text(json.dumps(smoke, indent=2) + "\n")
    (out / "lftnet_alignment_cases.csv").write_text(
        "case_id,status,note\n"
        "BOUNDARY_60S,not_run,blocked_no_model\n"
        "UTC_OFFSET,not_run,blocked_no_model\n"
        "BATCH_1_VS_N,not_run,blocked_no_model\n"
    )

    md = f"""# Stage 8 — LFTNet Alignment / Smoke Sanity

**Status:** `BLOCKED` — full inference **not started**.

## Why

Provenance class `{manifest.get('provenance_class')}`: drop is not a runnable release.

- No official checkpoint under the drop.
- `se-tcn-Eqt_utils.py` has **no import statements** and references Keras/custom layers (`SeqSelfAttention`, `FeedForward`, `f1`, …) that are not defined in-file.
- Refusing to run with **random initialization** (would be scientifically invalid).

## Checklist (all not_run / fail)

| Check | Result |
|--|--|
| checkpoint loads (not random) | FAIL |
| input shape / 100 Hz / ENZ order | not_run |
| output class order / length / sample offset | not_run |
| 60s↔120s / UTC remap / boundary | not_run |
| batch consistency / CPU-GPU / determinism | not_run |

## Gate

`alignment_gate = FAIL_STOP_NO_FULL_INFER`

Per Stage-8 rules: **do not** run Stage-6-protocol confirm comparator or candidate complementarity GPU jobs until alignment passes.
"""
    (reports / "lftnet_alignment_sanity.md").write_text(md)
    print(json.dumps({"smoke": smoke["status"], "alignment_gate": smoke["alignment_gate"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
