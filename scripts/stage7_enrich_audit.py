#!/usr/bin/env python
"""Enrich comparator audit markdown from registry (no f-string backslash issues)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    reg = json.loads((ROOT / "artifacts/results/stage7/comparator_registry.json").read_text())
    pn = Path.home() / ".seisbench/models/v3/phasenet"
    for w, key in [("ethz", "PhaseNet-ETHZ"), ("scedc", "PhaseNet-SCEDC"), ("stead", "PhaseNet-STEAD")]:
        jpath = pn / f"{w}.json.v2"
        if not jpath.exists():
            continue
        j = json.loads(jpath.read_text())
        m = reg["models"][key]
        for k, v in j.items():
            if "threshold" in k.lower() or k in ("sampling_rate", "component_order", "phases"):
                m[k] = v
        m["window_length_note"] = "SeisBench PhaseNet annotate default"
        m["preprocessing"] = "SeisBench annotate + project UTC remap ENZ->ZNE / PSN"
        m["license"] = "SeisBench model zoo / original dataset licenses"
    (ROOT / "artifacts/results/stage7/comparator_registry.json").write_text(json.dumps(reg, indent=2) + "\n")

    lines = [
        "# Stage 7A — Comparator Reproducibility Audit",
        "",
        f"**SeisBench:** {reg.get('seisbench_version')}  ",
        f"**New valid comparators:** {reg['n_new_valid_comparators']} → `{reg['new_valid_comparators']}`  ",
        f"**Proceed GPU batch:** `{reg['proceed_gpu_batch']}`",
        "",
        "## Per-model registry",
        "",
    ]
    for name, m in reg["models"].items():
        lines += [
            f"### {name}",
            "",
            f"- official_source: `{m.get('official_source')}`",
            f"- weight_name: `{m.get('weight_name')}`",
            f"- training_dataset: `{m.get('training_dataset')}`",
            f"- possible_confirm_leakage: `{m.get('possible_confirm_leakage')}`",
            f"- sampling_rate: `{m.get('sampling_rate')}`",
            f"- component_order: `{m.get('component_order')}`",
            f"- window_length: `{m.get('window_length_note')}`",
            f"- preprocessing: `{m.get('preprocessing')}`",
            f"- license: `{m.get('license')}`",
            f"- valid_comparator: `{m.get('valid_comparator')}`",
            f"- exclusion_reason: `{m.get('exclusion_reason')}`",
            f"- weight_sha256: `{m.get('weight_sha256')}`",
            "",
        ]
    (ROOT / "reports/stage7/comparator_reproducibility_audit.md").write_text("\n".join(lines) + "\n")
    print("audit_enriched")


if __name__ == "__main__":
    main()
