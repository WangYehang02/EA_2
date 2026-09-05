#!/usr/bin/env python
"""Post-process full-dev artifacts into a short summary (optional)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from earthquake.config import artifacts_dir

OUT = artifacts_dir() / "results" / "pairwise_fulldev"


def main() -> None:
    final = OUT / "fulldev_final.json"
    if not final.exists():
        raise SystemExit("fulldev_final.json missing")
    d = json.loads(final.read_text())
    print(json.dumps({k: d[k] for k in ["passed", "gates", "tau", "recommend_confirm"] if k in d}, indent=2))
    print("fixed", d.get("fixed", {}).get("f1@0.5"))
    print("resid", d.get("resid", {}).get("f1@0.5"))
    print("scalar", d.get("scalar", {}).get("f1@0.5"))
    print("Δfixed", d.get("scalar", {}).get("delta_f1@0.5_vs_fixed"))
    print("Δresid", d.get("scalar", {}).get("delta_f1@0.5_vs_resid"))


if __name__ == "__main__":
    main()
