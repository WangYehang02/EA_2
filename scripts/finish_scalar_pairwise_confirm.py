#!/usr/bin/env python
"""Print confirm final summary."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from earthquake.config import artifacts_dir

OUT = artifacts_dir() / "results" / "pairwise_confirm"


def main() -> None:
    p = OUT / "confirm_final.json"
    if not p.exists():
        raise SystemExit("confirm_final.json missing")
    d = json.loads(p.read_text())
    print(d["status"])
    print(d["interpretation"])
    print("Δfixed", d["scalar"].get("delta_f1@0.5_vs_fixed"), "Δresid", d["scalar"].get("delta_f1@0.5_vs_resid"))
    print("boot_sf", d["bootstrap_vs_fixed"]["delta_f1@0.5"])
    print("boot_sr", d["bootstrap_vs_resid"]["delta_f1@0.5"])


if __name__ == "__main__":
    main()
