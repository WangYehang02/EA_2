"""Future blind-test runner: lock-verified, fail-closed, no confirm I/O.

Primary methods only: fixed_rescore_UNION and frozen STEAD_top1.
Neural inference / waveform reads are out of scope here; this module rescores
already-extracted candidate tables (dev/synthetic). Confirm and DiTing paths
are refused by path guard.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from earthquake.fusion.candidate_rescorer import rescore_phase_candidates
from earthquake.fusion.peak_candidates import PeakCandidate
from earthquake.gating.cache_io import attach_expected_s
from earthquake.stage10.preconfirm_readiness import sha256_file

ALLOWED_PRIMARY_METHODS = ("fixed_rescore_UNION", "STEAD_top1")
FORBIDDEN_SEARCH_TOKENS = (
    "threshold_grid",
    "search_threshold",
    "tune_threshold",
    "best_threshold",
    "search_k",
    "select_k",
    "best_k",
    "K_grid",
    "choose_k",
    "glob_checkpoints",
    "replace_primary",
)
CONFIRM_PATH_MARKERS = (
    "final_confirm",
    "stage6_internal_confirm",
    "confirm_predictions",
    "confirm_method_metrics",
    "/diting",
    "diting",
)


class BlindRunnerError(RuntimeError):
    pass


def refuse_confirm_or_diting_path(path: Path | str) -> None:
    s = str(path).replace("\\", "/").lower()
    for m in CONFIRM_PATH_MARKERS:
        if m in s:
            raise BlindRunnerError(f"refusing confirm/DiTing path: {path}")


def _require_finite(name: str, value: float) -> float:
    v = float(value)
    if not np.isfinite(v):
        raise BlindRunnerError(f"fail_closed: nonfinite {name}={value!r}")
    return v


def verify_final_method_lock(lock_dir: Path) -> dict[str, Any]:
    lock_path = lock_dir / "FINAL_METHOD.LOCK.json"
    sidecar = lock_dir / "FINAL_METHOD.LOCK.sha256"
    if not lock_path.exists() or not sidecar.exists():
        raise BlindRunnerError("FINAL_METHOD.LOCK.json or .sha256 missing")
    digest = sha256_file(lock_path)
    expected = sidecar.read_text().strip()
    if digest != expected:
        raise BlindRunnerError("FINAL_METHOD.LOCK sha256 mismatch vs sidecar")
    lock = json.loads(lock_path.read_text())
    if lock.get("primary_lock_name") != "fixed_rescore_UNION":
        raise BlindRunnerError("lock primary is not fixed_rescore_UNION")
    if lock.get("contains_DKPN"):
        raise BlindRunnerError("DKPN must not be in deployable primary")
    ida = Path(lock["IDA_candidates"]["checkpoint"]["path"])
    stead = Path(lock["STEAD_candidates"]["checkpoint"]["path"])
    if sha256_file(ida) != lock["IDA_candidates"]["checkpoint"]["sha256"]:
        raise BlindRunnerError("IDA checkpoint sha256 mismatch")
    stead_sha = lock["STEAD_candidates"]["checkpoint"]["sha256_rehashed_this_audit"]
    if sha256_file(stead) != stead_sha:
        raise BlindRunnerError("STEAD checkpoint sha256 mismatch")
    cfg = lock["config_hash"]
    stage6 = Path(lock["stage6_method_lock_path"])
    if sha256_file(stage6) != cfg["stage6_method_lock_sha256"]:
        raise BlindRunnerError("stage6 method_lock sha256 mismatch")
    return lock


def candidate_probability(row: pd.Series) -> float:
    ps = row.get("stead_probability", np.nan)
    pi = row.get("ida_probability", np.nan)
    ps = float(ps) if pd.notna(ps) else float("nan")
    pi = float(pi) if pd.notna(pi) else float("nan")
    vals = [x for x in (ps, pi) if np.isfinite(x)]
    if not vals:
        raise BlindRunnerError("fail_closed: no finite source probability")
    return max(max(vals), 1e-6)


def rescore_one_trace(
    cand_rows: pd.DataFrame,
    *,
    expected_s_sample: float,
    sigma_samples: float,
    history_available: bool,
    lw: float,
    lh: float,
    lp: float,
) -> float:
    if cand_rows is None or len(cand_rows) == 0:
        raise BlindRunnerError("fail_closed: no UNION candidates")
    expected = _require_finite("expected_s_sample", expected_s_sample)
    sigma = _require_finite("sigma_samples", sigma_samples)
    cands: list[PeakCandidate] = []
    for r in cand_rows.itertuples(index=False):
        samp = _require_finite("candidate_sample", float(r.candidate_sample))
        ps = getattr(r, "stead_probability", np.nan)
        pi = getattr(r, "ida_probability", np.nan)
        ps = float(ps) if pd.notna(ps) else float("nan")
        pi = float(pi) if pd.notna(pi) else float("nan")
        vals = [x for x in (ps, pi) if np.isfinite(x)]
        if not vals:
            raise BlindRunnerError("fail_closed: no finite source probability")
        p = max(max(vals), 1e-6)
        rank = int(getattr(r, "candidate_index", getattr(r, "candidate_rank", 0)))
        cands.append(
            PeakCandidate(
                sample_index=int(samp),
                absolute_utc=None,
                peak_probability=p,
                prominence=p,
                peak_width=float("nan"),
                local_entropy=0.0,
                rank=rank,
                fallback_peak=False,
                phase="S",
            )
        )
    best, rows = rescore_phase_candidates(
        cands,
        expected_sample=expected,
        sigma_samples=sigma,
        lambda_wave=lw,
        lambda_history=lh,
        lambda_prominence=lp,
        history_available=bool(history_available),
    )
    if best is None:
        raise BlindRunnerError("fail_closed: rescore returned no candidate")
    for row in rows:
        _require_finite("score", float(row["score"]))
    return float(_require_finite("pick", best.sample_index))


def expected_s_maps(
    meta: pd.DataFrame,
    hist: pd.DataFrame,
    global_res: dict[str, float],
    *,
    shrink_k: float,
    min_history: int,
    mad_disable_s: float,
    min_sigma_s: float,
    max_sigma_s: float,
) -> tuple[dict[str, float], dict[str, float], dict[str, bool]]:
    hist_ix = hist.drop_duplicates("trace_name").set_index("trace_name")
    hist_map = hist_ix.to_dict("index")
    exp_s: dict[str, float] = {}
    sigma: dict[str, float] = {}
    hist_ok: dict[str, bool] = {}
    for rec in meta.to_dict("records"):
        tn = str(rec["trace_name"])
        if tn in hist_map:
            rec = {**rec, **hist_map[tn]}
        merged = pd.Series(rec)
        e = attach_expected_s(
            merged,
            global_res=global_res,
            shrink_k=shrink_k,
            min_history=min_history,
            mad_disable_s=mad_disable_s,
            min_sigma_s=min_sigma_s,
            max_sigma_s=max_sigma_s,
        )
        exp_s[tn] = _require_finite("expected_s_sample", float(e["expected_s_sample"]))
        sigma[tn] = _require_finite("history_sigma_samples", float(e["history_sigma_samples"]))
        hist_ok[tn] = bool(e["gate_history_available"])
    return exp_s, sigma, hist_ok


def predict_fixed_rescore_union(
    union: pd.DataFrame,
    names: Iterable[str],
    *,
    exp_s: dict[str, float],
    sigma: dict[str, float],
    hist_ok: dict[str, bool],
    lw: float,
    lh: float,
    lp: float,
) -> np.ndarray:
    names = [str(x) for x in names]
    by = {str(tn): g for tn, g in union.groupby(union["trace_name"].astype(str), sort=False)}
    out = np.empty(len(names), dtype=np.float64)
    for i, tn in enumerate(names):
        if tn not in by:
            raise BlindRunnerError(f"fail_closed: missing UNION rows for {tn}")
        if tn not in exp_s or tn not in sigma:
            raise BlindRunnerError(f"fail_closed: missing history maps for {tn}")
        out[i] = rescore_one_trace(
            by[tn],
            expected_s_sample=exp_s[tn],
            sigma_samples=sigma[tn],
            history_available=hist_ok[tn],
            lw=lw,
            lh=lh,
            lp=lp,
        )
    if not np.isfinite(out).all():
        raise BlindRunnerError("fail_closed: nonfinite primary prediction")
    return out


def predict_stead_top1(stead_cache: pd.DataFrame, names: Iterable[str]) -> np.ndarray:
    names = [str(x) for x in names]
    g = stead_cache.sort_values("candidate_rank").groupby("trace_name", sort=False).first()
    col = "top1_s_sample" if "top1_s_sample" in g.columns else "candidate_sample"
    raw = g[col].reindex(names).to_numpy(float)
    if raw.shape[0] != len(names) or not np.isfinite(raw).all():
        raise BlindRunnerError("fail_closed: STEAD_top1 missing or nonfinite")
    return raw


def atomic_freeze_predictions(path: Path, arr: np.ndarray) -> str:
    refuse_confirm_or_diting_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp.{os.getpid()}.npy"
    np.save(tmp, np.asarray(arr, dtype=np.float64))
    os.replace(tmp, path)
    digest = sha256_file(path)
    (path.with_suffix(path.suffix + ".sha256")).write_text(digest + "\n")
    return digest


@dataclass
class PrimaryFreeze:
    method: str
    path: Path
    sha256: str
    frozen: bool


def run_primary(
    method: str,
    *,
    lock: dict[str, Any],
    union: pd.DataFrame | None,
    stead_cache: pd.DataFrame | None,
    names: Iterable[str],
    exp_s: dict[str, float] | None,
    sigma: dict[str, float] | None,
    hist_ok: dict[str, bool] | None,
    out_path: Path,
) -> PrimaryFreeze:
    if method not in ALLOWED_PRIMARY_METHODS:
        raise BlindRunnerError(f"method not allowed: {method}")
    refuse_confirm_or_diting_path(out_path)
    if method == "fixed_rescore_UNION":
        if union is None or exp_s is None or sigma is None or hist_ok is None:
            raise BlindRunnerError("UNION inputs missing")
        lam = lock["fixed_rescore"]["lambdas_s"]
        pred = predict_fixed_rescore_union(
            union,
            names,
            exp_s=exp_s,
            sigma=sigma,
            hist_ok=hist_ok,
            lw=float(lam["lw"]),
            lh=float(lam["lh"]),
            lp=float(lam["lp"]),
        )
    else:
        if stead_cache is None:
            raise BlindRunnerError("STEAD cache missing")
        pred = predict_stead_top1(stead_cache, names)
    digest = atomic_freeze_predictions(out_path, pred)
    return PrimaryFreeze(method=method, path=out_path, sha256=digest, frozen=True)


def run_preregistered_diagnostics(
    *,
    primary: PrimaryFreeze,
    oracle_fn=None,
) -> dict[str, Any]:
    """Diagnostics only after primary freeze. Oracle must not select the primary."""
    if not primary.frozen:
        raise BlindRunnerError("diagnostics forbidden before primary freeze")
    if oracle_fn is not None:
        # Explicitly not used for output or selection.
        raise BlindRunnerError("oracle_fn must not be passed into future runner diagnostics")
    return {
        "primary_method": primary.method,
        "primary_sha256": primary.sha256,
        "oracle_used_for_prediction": False,
        "oracle_used_for_selection": False,
    }


def wrap_missing_channel_read(read_fn, *args, **kwargs):
    try:
        arr = read_fn(*args, **kwargs)
    except (KeyError, ValueError) as exc:
        raise BlindRunnerError(f"fail_closed: missing/corrupt trace: {exc}") from exc
    if arr is None:
        raise BlindRunnerError("fail_closed: empty waveform")
    x = np.asarray(arr)
    if x.ndim != 2 or x.shape[0] != 3:
        raise BlindRunnerError(f"fail_closed: bad waveform shape {x.shape}")
    if not np.isfinite(x).all():
        raise BlindRunnerError("fail_closed: nonfinite waveform samples")
    return x
