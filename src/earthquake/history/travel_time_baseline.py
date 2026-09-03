"""Train-only distance / depth travel-time baselines for Stage 2 residual priors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

ModelKind = Literal["dist_1d", "dist_depth_2d", "quantile_spline", "mlp"]


def _observed_taus(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "path_travel_time_p_s" in out.columns and "path_travel_time_s_s" in out.columns:
        out["obs_tau_p"] = pd.to_numeric(out["path_travel_time_p_s"], errors="coerce")
        out["obs_tau_s"] = pd.to_numeric(out["path_travel_time_s_s"], errors="coerce")
    else:
        # derive from samples
        origin = pd.to_datetime(out["origin_time"], utc=True)
        start = pd.to_datetime(out["trace_start_time"], utc=True)
        sr = out["sampling_rate_hz"].astype(float)
        out["obs_tau_p"] = (
            start + pd.to_timedelta(out["p_arrival_sample"] / sr, unit="s") - origin
        ).dt.total_seconds()
        out["obs_tau_s"] = (
            start + pd.to_timedelta(out["s_arrival_sample"] / sr, unit="s") - origin
        ).dt.total_seconds()
    out["obs_delta_sp"] = out["obs_tau_s"] - out["obs_tau_p"]
    return out


@dataclass
class TravelTimeBaseline:
    kind: str
    meta: dict[str, Any]
    # filled per kind
    table: pd.DataFrame | None = None
    models: dict[str, Any] | None = None

    def predict_row(self, row: pd.Series) -> dict[str, float]:
        d = float(row.get("distance_km", np.nan))
        z = float(row.get("source_depth_km", np.nan))
        elev = float(row.get("station_elevation_m", np.nan))
        if self.kind == "dist_1d":
            return self._predict_1d(d)
        if self.kind == "dist_depth_2d":
            return self._predict_2d(d, z)
        if self.kind == "quantile_spline":
            return self._predict_spline(d)
        if self.kind == "mlp":
            return self._predict_mlp(d, z, elev)
        raise ValueError(self.kind)

    def predict_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.kind == "mlp":
            return self._predict_mlp_frame(df)
        rows = [self.predict_row(r) for _, r in df.iterrows()]
        return pd.DataFrame(rows, index=df.index)

    def _predict_mlp_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        models = self.models or {}
        med = self.meta.get("feature_medians", [0.0, 0.0, 0.0])
        d = pd.to_numeric(df.get("distance_km"), errors="coerce").to_numpy(dtype=np.float64)
        z = pd.to_numeric(df.get("source_depth_km"), errors="coerce").to_numpy(dtype=np.float64)
        elev = pd.to_numeric(df.get("station_elevation_m"), errors="coerce").to_numpy(dtype=np.float64)
        x = np.column_stack([d, z, elev])
        for j in range(3):
            bad = ~np.isfinite(x[:, j])
            x[bad, j] = float(med[j])
        out = {}
        for key, target in [("base_tau_p", "p"), ("base_tau_s", "s"), ("base_delta_sp", "sp")]:
            m = models.get(target)
            out[key] = m.predict(x) if m is not None else np.full(len(df), np.nan)
        return pd.DataFrame(out, index=df.index)

    def _predict_1d(self, d: float) -> dict[str, float]:
        tab = self.table
        assert tab is not None
        if not np.isfinite(d) or len(tab) == 0:
            return {"base_tau_p": float("nan"), "base_tau_s": float("nan"), "base_delta_sp": float("nan")}
        idx = (tab["dist_center"] - d).abs().idxmin()
        r = tab.loc[idx]
        return {
            "base_tau_p": float(r["tau_p_median"]),
            "base_tau_s": float(r["tau_s_median"]),
            "base_delta_sp": float(r["delta_sp_median"]),
        }

    def _predict_2d(self, d: float, z: float) -> dict[str, float]:
        tab = self.table
        assert tab is not None
        if not np.isfinite(d) or not np.isfinite(z) or len(tab) == 0:
            return self._predict_1d(d) if np.isfinite(d) else {
                "base_tau_p": float("nan"),
                "base_tau_s": float("nan"),
                "base_delta_sp": float("nan"),
            }
        # nearest by normalized (dist, depth)
        dd = (tab["dist_center"] - d) / max(tab["dist_center"].std(), 1.0)
        zz = (tab["depth_center"] - z) / max(tab["depth_center"].std(), 1.0)
        idx = (dd**2 + zz**2).idxmin()
        r = tab.loc[idx]
        return {
            "base_tau_p": float(r["tau_p_median"]),
            "base_tau_s": float(r["tau_s_median"]),
            "base_delta_sp": float(r["delta_sp_median"]),
        }

    def _predict_spline(self, d: float) -> dict[str, float]:
        models = self.models or {}
        if not np.isfinite(d):
            return {"base_tau_p": float("nan"), "base_tau_s": float("nan"), "base_delta_sp": float("nan")}
        out = {}
        for key, target in [("base_tau_p", "p"), ("base_tau_s", "s"), ("base_delta_sp", "sp")]:
            m = models.get(target)
            if m is None:
                out[key] = float("nan")
            else:
                out[key] = float(np.asarray(m(np.array([d]))).ravel()[0])
        return out

    def _predict_mlp(self, d: float, z: float, elev: float) -> dict[str, float]:
        models = self.models or {}
        x = np.array([[d, z, elev if np.isfinite(elev) else 0.0]], dtype=np.float64)
        # fill nan features with train medians
        med = self.meta.get("feature_medians", [0.0, 0.0, 0.0])
        for j in range(3):
            if not np.isfinite(x[0, j]):
                x[0, j] = float(med[j])
        out = {}
        for key, target in [("base_tau_p", "p"), ("base_tau_s", "s"), ("base_delta_sp", "sp")]:
            m = models.get(target)
            out[key] = float(m.predict(x)[0]) if m is not None else float("nan")
        return out


def fit_dist_1d(train: pd.DataFrame, n_bins: int = 20) -> TravelTimeBaseline:
    df = _observed_taus(train).dropna(subset=["distance_km", "obs_tau_p", "obs_tau_s"])
    df = df.assign(dist_bin=pd.qcut(df["distance_km"], q=min(n_bins, max(df["distance_km"].nunique(), 1)), duplicates="drop"))
    g = df.groupby("dist_bin", observed=True).agg(
        dist_center=("distance_km", "median"),
        tau_p_median=("obs_tau_p", "median"),
        tau_s_median=("obs_tau_s", "median"),
        delta_sp_median=("obs_delta_sp", "median"),
        n=("distance_km", "size"),
    ).reset_index(drop=True)
    return TravelTimeBaseline(kind="dist_1d", meta={"n_bins": n_bins, "n_train": len(df)}, table=g)


def fit_dist_depth_2d(train: pd.DataFrame, n_dist: int = 12, n_depth: int = 5) -> TravelTimeBaseline:
    df = _observed_taus(train).dropna(subset=["distance_km", "source_depth_km", "obs_tau_p", "obs_tau_s"])
    df = df.assign(
        dist_bin=pd.qcut(df["distance_km"], q=min(n_dist, max(df["distance_km"].nunique(), 1)), duplicates="drop"),
        depth_bin=pd.qcut(df["source_depth_km"], q=min(n_depth, max(df["source_depth_km"].nunique(), 1)), duplicates="drop"),
    )
    g = df.groupby(["dist_bin", "depth_bin"], observed=True).agg(
        dist_center=("distance_km", "median"),
        depth_center=("source_depth_km", "median"),
        tau_p_median=("obs_tau_p", "median"),
        tau_s_median=("obs_tau_s", "median"),
        delta_sp_median=("obs_delta_sp", "median"),
        n=("distance_km", "size"),
    ).reset_index(drop=True)
    return TravelTimeBaseline(kind="dist_depth_2d", meta={"n_dist": n_dist, "n_depth": n_depth, "n_train": len(df)}, table=g)


def fit_quantile_spline(train: pd.DataFrame) -> TravelTimeBaseline:
    from scipy.interpolate import UnivariateSpline

    df = _observed_taus(train).dropna(subset=["distance_km", "obs_tau_p", "obs_tau_s"]).sort_values("distance_km")
    # bin medians then spline for robustness
    bins = pd.qcut(df["distance_km"], q=min(30, max(df["distance_km"].nunique(), 1)), duplicates="drop")
    g = df.groupby(bins, observed=True).agg(
        x=("distance_km", "median"),
        p=("obs_tau_p", "median"),
        s=("obs_tau_s", "median"),
        sp=("obs_delta_sp", "median"),
    ).dropna()
    models = {}
    for name, col in [("p", "p"), ("s", "s"), ("sp", "sp")]:
        x = g["x"].to_numpy()
        y = g[col].to_numpy()
        if len(x) < 4:
            models[name] = lambda xx, y0=float(np.median(y)): np.full(len(np.atleast_1d(xx)), y0)
        else:
            spl = UnivariateSpline(x, y, k=3, s=max(len(x) * 0.5, 1.0))
            models[name] = spl
    return TravelTimeBaseline(kind="quantile_spline", meta={"n_train": len(df)}, models=models)


def fit_mlp(train: pd.DataFrame, seed: int = 0) -> TravelTimeBaseline:
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    df = _observed_taus(train).dropna(subset=["distance_km", "source_depth_km", "obs_tau_p", "obs_tau_s"])
    elev = df["station_elevation_m"].astype(float) if "station_elevation_m" in df.columns else pd.Series(0.0, index=df.index)
    elev = elev.fillna(elev.median() if elev.notna().any() else 0.0)
    X = np.column_stack(
        [
            df["distance_km"].astype(float).to_numpy(),
            df["source_depth_km"].astype(float).to_numpy(),
            elev.to_numpy(),
        ]
    )
    med = np.nanmedian(X, axis=0).tolist()
    models = {}
    for name, col in [("p", "obs_tau_p"), ("s", "obs_tau_s"), ("sp", "obs_delta_sp")]:
        y = df[col].astype(float).to_numpy()
        m = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(32, 16),
                max_iter=400,
                random_state=seed,
                early_stopping=True,
                validation_fraction=0.1,
            ),
        )
        m.fit(X, y)
        models[name] = m
    return TravelTimeBaseline(kind="mlp", meta={"n_train": len(df), "feature_medians": med, "seed": seed}, models=models)


def evaluate_baseline(model: TravelTimeBaseline, df: pd.DataFrame) -> dict[str, float]:
    obs = _observed_taus(df)
    pred = model.predict_frame(obs)
    out = {}
    for phase, ocol, pcol in [
        ("p", "obs_tau_p", "base_tau_p"),
        ("s", "obs_tau_s", "base_tau_s"),
        ("sp", "obs_delta_sp", "base_delta_sp"),
    ]:
        m = np.isfinite(obs[ocol]) & np.isfinite(pred[pcol])
        err = (pred.loc[m, pcol] - obs.loc[m, ocol]).abs()
        out[f"{phase}_mae"] = float(err.mean()) if m.any() else float("nan")
        out[f"{phase}_median_ae"] = float(err.median()) if m.any() else float("nan")
        out[f"{phase}_n"] = int(m.sum())
    out["score"] = float(np.nanmean([out["p_mae"], out["s_mae"]]))
    return out


def fit_all_and_select(train: pd.DataFrame, val: pd.DataFrame, seed: int = 0) -> tuple[TravelTimeBaseline, pd.DataFrame]:
    candidates = [
        fit_dist_1d(train),
        fit_dist_depth_2d(train),
        fit_quantile_spline(train),
        fit_mlp(train, seed=seed),
    ]
    rows = []
    best = candidates[0]
    best_score = float("inf")
    for m in candidates:
        met = evaluate_baseline(m, val)
        met["kind"] = m.kind
        rows.append(met)
        if np.isfinite(met["score"]) and met["score"] < best_score:
            best_score = met["score"]
            best = m
    return best, pd.DataFrame(rows)
