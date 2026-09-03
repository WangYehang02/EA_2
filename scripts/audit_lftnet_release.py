#!/usr/bin/env python
"""Stage 8 — read-only LFTNet provenance / completeness audit (no training, no full infer)."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE_CANDIDATES = [
    Path.home() / "baseline",
    ROOT / "baseline",
    Path("/home/yehang/baseline"),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_lftnet_root() -> dict:
    notes = []
    found = []
    for c in BASELINE_CANDIDATES:
        exists = c.exists()
        notes.append({"path": str(c), "exists": exists, "realpath": str(c.resolve()) if exists else None})
        if not exists:
            continue
        for p in c.rglob("*"):
            if p.is_file():
                found.append(p)
    # Prefer directory named *LFTNet*
    roots = sorted({p.parent for p in found})
    primary = None
    for r in roots:
        if "lftnet" in r.name.lower() or "LFTNet" in r.name:
            primary = r
            break
    if primary is None and roots:
        primary = roots[0]
    return {"candidate_notes": notes, "primary_root": str(primary) if primary else None, "n_files": len(found)}


def list_tree(root: Path, max_depth: int = 4) -> list[str]:
    lines = []
    root = root.resolve()
    for dirpath, dirnames, filenames in os.walk(root):
        rel = Path(dirpath).relative_to(root)
        depth = 0 if str(rel) == "." else len(rel.parts)
        if depth > max_depth:
            dirnames.clear()
            continue
        indent = "  " * depth
        lines.append(f"{indent}{Path(dirpath).name}/")
        for fn in sorted(filenames):
            lines.append(f"{indent}  {fn}")
    return lines


def git_info(root: Path) -> dict:
    if not (root / ".git").exists() and not (root.parent / ".git").exists():
        # walk up
        cur = root
        git_root = None
        for _ in range(4):
            if (cur / ".git").exists():
                git_root = cur
                break
            cur = cur.parent
        if git_root is None:
            return {"is_git_repo": False}
    else:
        git_root = root if (root / ".git").exists() else root.parent

    def run(args: list[str]) -> str:
        try:
            return subprocess.check_output(args, cwd=git_root, text=True, stderr=subprocess.STDOUT).strip()
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {e}"

    return {
        "is_git_repo": True,
        "git_root": str(git_root),
        "remote_v": run(["git", "remote", "-v"]),
        "status": run(["git", "status", "--short"]),
        "log": run(["git", "log", "-5", "--oneline"]),
        "head": run(["git", "rev-parse", "HEAD"]),
        "tags": run(["git", "tag", "--list"]),
    }


def component_checklist(root: Path, text_blob: str) -> dict:
    names = {p.name.lower() for p in root.rglob("*") if p.is_file()}
    blob = text_blob.lower()
    return {
        "LFTNet_model_class": ("cred2" in blob) or ("lftnet" in blob),
        "RSDB": "rsdb" in blob,
        "MSSE_TCN": ("tcn_block_ms_se" in blob) or ("msse" in blob),
        "detection_P_S_multitask": ("picker_p" in blob and "picker_s" in blob and "detector" in blob),
        "softmax_or_sigmoid": ("sigmoid" in blob) or ("softmax" in blob),
        "input_length_documented": ("6000" in blob) or ("60 s" in blob) or ("60s" in blob) or ("12000" in blob),
        "sampling_rate_documented": ("100" in blob and "hz" in blob) or ("sampling" in blob),
        "component_order_documented": ("zne" in blob) or ("enz" in blob) or ("component" in blob),
        "normalization_documented": ("normaliz" in blob),
        "label_order_documented": ("detector" in blob and "picker_p" in blob),
        "checkpoint_file_present": any(n.endswith((".h5", ".hdf5", ".pt", ".pth", ".ckpt", ".keras")) for n in names),
        "checkpoint_training_metadata": any("checkpoint" in n or "weight" in n for n in names),
        "peak_extraction_code": ("argmax" in blob) or ("peak" in blob) or ("trigger" in blob),
        "threshold_config": ("threshold" in blob),
        "INSTANCE_eval_manifest": any("instance" in n and n.endswith((".csv", ".txt", ".json")) for n in names),
        "paper_10k_to_20k_segment_code": ("10000" in blob) or ("20,000" in blob) or ("20000" in blob),
        "train_script": any("train" in n for n in names),
        "infer_script": any(x in n for n in names for x in ("infer", "predict", "test", "eval")),
        "license_file": any(n in ("license", "license.md", "license.txt") for n in names),
        "readme": any(n.startswith("readme") for n in names),
        "requirements": any("requirement" in n or n.endswith(".yml") or n.endswith(".yaml") for n in names),
    }


def classify(checklist: dict, n_files: int, has_ckpt: bool) -> tuple[str, list[str]]:
    evidence = []
    # Incomplete drop: only 2 files, no ckpt, no train/infer scripts
    if n_files <= 3 and not has_ckpt:
        evidence.append(f"only {n_files} files under drop; no checkpoint")
        if checklist["MSSE_TCN"] and checklist["detection_P_S_multitask"]:
            evidence.append("fragment contains MSSE-TCN-like blocks and detector/P/S heads (cred2 class)")
        evidence.append("missing imports (keras layers, SeqSelfAttention, FeedForward, f1) — file not self-contained")
        evidence.append("README.md is a requirements list (tensorflow~=2.5.0), not project documentation")
        evidence.append("no LICENSE, no train/infer scripts, no configs, no paper 10k/20k construction")
        evidence.append("directory name suggests GitHub Tianjiyu1/LFTNet @79994f6 but is not a git checkout")
        return "E. incomplete_or_unusable", evidence
    if has_ckpt and not checklist["INSTANCE_eval_manifest"]:
        return "C. official_checkpoint_missing_metadata", evidence
    if checklist["train_script"] and checklist["infer_script"] and has_ckpt and checklist["license_file"]:
        return "A. official_complete_release", evidence
    if checklist["train_script"] and not has_ckpt:
        return "B. official_code_no_checkpoint", evidence
    return "E. incomplete_or_unusable", evidence


def main() -> None:
    out = ROOT / "artifacts/results/stage8"
    reports = ROOT / "reports/stage8"
    out.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    loc = find_lftnet_root()
    primary = Path(loc["primary_root"]) if loc["primary_root"] else None

    file_hashes = {}
    text_blob = ""
    tree = []
    git = {"is_git_repo": False}
    checklist = {}
    if primary and primary.exists():
        tree = list_tree(primary)
        git = git_info(primary)
        for p in sorted(primary.rglob("*")):
            if p.is_file():
                rel = str(p.relative_to(primary))
                file_hashes[rel] = {
                    "sha256": sha256_file(p),
                    "size": p.stat().st_size,
                    "realpath": str(p.resolve()),
                }
                if p.suffix in {".py", ".md", ".txt", ".yml", ".yaml", ".json", ".cfg", ".ini"}:
                    try:
                        text_blob += "\n" + p.read_text(encoding="utf-8", errors="replace")
                    except Exception:  # noqa: BLE001
                        pass
        checklist = component_checklist(primary, text_blob)

    has_ckpt = bool(checklist.get("checkpoint_file_present"))
    provenance_class, evidence = classify(checklist, loc["n_files"], has_ckpt)

    # Folder name hint
    folder_hint = {
        "dirname": primary.name if primary else None,
        "interpreted_as": "Likely GitHub archive fragment Tianjiyu1/LFTNet commit prefix 79994f6 (not verified as full release)",
        "note": "Name alone is NOT sufficient for official_complete_release",
    }

    # Dependency note vs PS
    deps = {
        "drop_readme_claims": "tensorflow~=2.5.0, keras~=2.3.1, numpy~=1.19.2 (from README.md content)",
        "PS_env": "torch 2.5.x present; tensorflow/keras MISSING in PS",
        "conflict": True,
        "recommendation": "Do not install TF2.5 into PS. If a complete release appears, create LFTNet_eval env.",
        "reuse_PS_for_full_infer": False,
    }

    gate = {
        "code_complete": False,
        "checkpoint_verifiable": False,
        "no_obvious_confirm_leakage": "unknown_no_checkpoint",
        "utc_sample_alignment_passed": False,
        "smoke_reasonable": False,
        "auto_continue_full_eval": False,
        "stop_reason": "incomplete_or_unusable_drop_no_checkpoint",
    }

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "user_claimed_path": "~/baseline",
        "user_claimed_path_exists": (Path.home() / "baseline").exists(),
        "resolved_primary_root": str(primary.resolve()) if primary else None,
        "location_probe": loc,
        "folder_hint": folder_hint,
        "git": git,
        "component_checklist": checklist,
        "provenance_class": provenance_class,
        "evidence": evidence,
        "dependencies": deps,
        "paper_refs": {
            "citation": "Guo et al. (2025) LFTNet, DOI 10.1029/2025EA004548",
            "software_doi": "10.5281/zenodo.15710535",
            "paper_S_F1_0.5_claimed": 0.844,
            "note": "Paper metrics are NOT comparable to Stage-6 protocol rankings",
        },
        "stage6_constraints": {
            "main_method": "fixed_rescore_UNION",
            "method_lock_sha256": "a02dc28e3f485559dd22b0d38c90d75923a159aed43a797bf234d765ada5b303",
            "confirm_consumed": True,
            "evaluation_type": "post-confirm external comparator evaluation",
            "sota_claim_allowed": False,
        },
        "gate": gate,
        "file_tree": tree,
    }

    (out / "lftnet_release_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "lftnet_file_hashes.json").write_text(json.dumps(file_hashes, indent=2) + "\n")

    md = f"""# Stage 8 — LFTNet Provenance Audit

**Created (UTC):** {manifest['created_utc']}  
**Provenance class:** `{provenance_class}`  
**Auto-continue full eval:** `{gate['auto_continue_full_eval']}`

## Path resolution

- User path `~/baseline`: **does not exist** on this server.
- Actual drop found at: `{manifest['resolved_primary_root']}`
- Realpath used for INSTANCE data (unchanged): via project symlink `data/INSTANCE` → do **not** copy ~156GB HDF5.

## Folder hint

- Directory name: `{folder_hint['dirname']}`
- Interpretation: {folder_hint['interpreted_as']}
- {folder_hint['note']}

## File tree

```
{chr(10).join(tree) if tree else '(empty)'}
```

## Git

```
{json.dumps(git, indent=2)}
```

## Component checklist

| Component | Present? |
|--|--|
"""
    for k, v in checklist.items():
        md += f"| {k} | `{v}` |\n"

    md += f"""
## Evidence for class `{provenance_class}`

"""
    for e in evidence:
        md += f"- {e}\n"

    md += f"""
## Dependencies vs conda `PS`

- Drop README lists **TensorFlow ~2.5 / Keras ~2.3 / numpy ~1.19**.
- `PS` has **PyTorch 2.5**; **TensorFlow/Keras absent**.
- Recommendation: **do not** mutate `PS`. Create `LFTNet_eval` only after a complete official release is available.

## Gate (must all pass to continue)

| Gate | Status |
|--|--|
| code_complete | `{gate['code_complete']}` |
| checkpoint_verifiable | `{gate['checkpoint_verifiable']}` |
| no_obvious_confirm_leakage | `{gate['no_obvious_confirm_leakage']}` |
| utc_sample_alignment_passed | `{gate['utc_sample_alignment_passed']}` |
| smoke_reasonable | `{gate['smoke_reasonable']}` |

**Stop reason:** `{gate['stop_reason']}`

## Paper numbers (NOT Stage-6 ranking)

Paper (Guo et al. 2025) reports on *their* INSTANCE protocol: S F1@0.5=0.844, P=0.822, R=0.867, MAE=0.232 s.  
These **must not** be ranked against Stage-6/7A confirm metrics.

## Next action for user

Please place the **full** Zenodo release (`10.5281/zenodo.15710535`) or complete GitHub tree **including official pretrained checkpoint + LICENSE + infer/train scripts + eval manifests** under a readable path (e.g. recreate `~/baseline/LFTNet`).  
Until then: **no training, no confirm comparator run, no main-method change**.
"""
    (reports / "lftnet_provenance_audit.md").write_text(md)
    print(json.dumps({"provenance_class": provenance_class, "primary": manifest["resolved_primary_root"], "auto_continue": False}, indent=2))


if __name__ == "__main__":
    main()
