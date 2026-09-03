"""Train/val-only hierarchical path-residual utilities for Stage 5.1 audits.

Does not touch Stage 3/4 test artifacts. History tables are built from train only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from earthquake.history.region import depth_bin, path_key, source_region, station_id
from earthquake.history.shrinkage import shrink_residual, shrinkage_weight
from earthquake.utils import mad


def directed_path_key(row: pd.Series, grid_size: float) -> str:
    region = source_region(float(row["source_latitude"]), float(row["source_longitude"]), float(grid_size))
    dbin = depth_bin(float(row["source_depth_km"]) if pd.notna(row.get("source_depth_km")) else float("nan"))
    sid = station_id(
        str(row["network"]),
        str(row["station"]),
        row.get("location", ""),
        str(row["channel_prefix"]),
    )
    return path_key(region, dbin, sid)


@dataclass
class PathResidualTable:
    grid_size: float
    stats: dict[str, dict[str, float]]
    global_median: float
    global_mad: float
    n_train_rows: int

    def query(self, key: str) -> dict[str, float]:
        return self.stats.get(
            key,
            {
                "n": 0.0,
                "median": float("nan"),
                "mad": float("nan"),
                "history_available": 0.0,
            },
        )


def build_path_residual_table(
    train_df: pd.DataFrame,
    *,
    residual_col: str,
    grid_size: float,
    min_history: int = 5,
) -> PathResidualTable:
    """Aggregate train residuals by directed source-cell–station path key."""
    if "split" in train_df.columns and not (train_df["split"] == "train").all():
        raise ValueError("build_path_residual_table requires train-only rows")
    keys = [directed_path_key(r, grid_size) for _, r in train_df.iterrows()]
    tmp = train_df.copy()
    tmp["_path_key"] = keys
    tmp = tmp[np.isfinite(tmp[residual_col].to_numpy(dtype=float))]
    stats: dict[str, dict[str, float]] = {}
    for key, sub in tmp.groupby("_path_key", sort=False):
        x = sub[residual_col].to_numpy(dtype=float)
        n = int(x.size)
        stats[str(key)] = {
            "n": float(n),
            "median": float(np.median(x)),
            "mad": float(mad(x)),
            "history_available": float(n >= min_history),
        }
    g = tmp[residual_col].to_numpy(dtype=float)
    return PathResidualTable(
        grid_size=float(grid_size),
        stats=stats,
        global_median=float(np.median(g)) if g.size else 0.0,
        global_mad=float(mad(g)) if g.size else float("nan"),
        n_train_rows=int(len(tmp)),
    )


def hierarchical_residual(
    fine: dict[str, float],
    coarse: dict[str, float],
    global_median: float,
    *,
    k_fine: float,
    k_coarse: float,
    min_history: int = 5,
) -> dict[str, Any]:
    """Nested shrinkage: fine -> coarse -> global (train-derived medians)."""
    n_f = float(fine.get("n", 0) or 0)
    n_c = float(coarse.get("n", 0) or 0)
    r_f = float(fine.get("median", np.nan))
    r_c = float(coarse.get("median", np.nan))
    r_g = float(global_median)

    if n_c >= min_history and np.isfinite(r_c):
        w_c = shrinkage_weight(n_c, k_coarse)
        r_mid = shrink_residual(r_c, r_g, n_c, k_coarse)
        coarse_ok = True
    else:
        w_c = 0.0
        r_mid = r_g
        coarse_ok = False

    if n_f >= min_history and np.isfinite(r_f):
        w_f = shrinkage_weight(n_f, k_fine)
        r_hat = shrink_residual(r_f, r_mid, n_f, k_fine)
        fine_ok = True
    else:
        w_f = 0.0
        r_hat = r_mid
        fine_ok = False

    history_available = bool(fine_ok or coarse_ok)
    return {
        "r_hat": float(r_hat),
        "w_fine": float(w_f),
        "w_coarse": float(w_c),
        "fine_ok": fine_ok,
        "coarse_ok": coarse_ok,
        "history_available": history_available,
        "n_fine": n_f,
        "n_coarse": n_c,
        "mad_fine": float(fine.get("mad", np.nan)),
        "mad_coarse": float(coarse.get("mad", np.nan)),
    }


def assert_no_test_split(df: pd.DataFrame, *, context: str) -> None:
    if "split" not in df.columns:
        raise ValueError(f"{context}: missing split column")
    bad = set(df["split"].astype(str).unique()) - {"train", "val"}
    if bad:
        raise ValueError(f"{context}: disallowed splits present: {sorted(bad)}")


def assert_event_disjoint(history_events: set[str], eval_events: set[str], *, context: str) -> None:
    leak = history_events & eval_events
    if leak:
        raise ValueError(f"{context}: event leakage ({len(leak)} events), e.g. {next(iter(leak))}")


def ae_summary(err: np.ndarray) -> dict[str, float]:
    x = np.asarray(err, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mae": float("nan"), "median_ae": float("nan"), "p95": float("nan"), "n": 0}
    return {
        "mae": float(np.mean(np.abs(x))),
        "median_ae": float(np.median(np.abs(x))),
        "p95": float(np.percentile(np.abs(x), 95)),
        "n": int(x.size),
    }


def event_bootstrap_delta_mae(
    event_ids: np.ndarray,
    err_baseline: np.ndarray,
    err_improved: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Paired event bootstrap of MAE(baseline)-MAE(improved); >0 means improved better."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "event_id": event_ids,
            "base": np.abs(err_baseline),
            "imp": np.abs(err_improved),
        }
    )
    df = df[np.isfinite(df["base"]) & np.isfinite(df["imp"])]
    groups = {eid: g for eid, g in df.groupby("event_id", sort=False)}
    eids = np.array(list(groups.keys()))
    if eids.size == 0:
        return {
            "mean": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "prob_improved": float("nan"),
            "n_events": 0,
        }

    def delta(sample_eids):
        ba, im = [], []
        for e in sample_eids:
            g = groups[e]
            ba.append(g["base"].to_numpy())
            im.append(g["imp"].to_numpy())
        return float(np.mean(np.concatenate(ba)) - np.mean(np.concatenate(im)))

    point = delta(eids)
    boots = np.array([delta(rng.choice(eids, size=eids.size, replace=True)) for _ in range(n_boot)], dtype=float)
    return {
        "mean": point,
        "boot_mean": float(np.mean(boots)),
        "ci95_low": float(np.percentile(boots, 2.5)),
        "ci95_high": float(np.percentile(boots, 97.5)),
        "prob_improved": float(np.mean(boots > 0)),
        "n_events": int(eids.size),
    }
