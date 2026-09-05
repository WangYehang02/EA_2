#!/usr/bin/env python
"""Resume pre-registered waveform pairwise pilot after IO/cache fix.

Preserves PAIRWISE.PILOT.FAILED provenance. Does not read confirm/full-dev.
Does not rebuild SPLIT.LOCK. Held-out once via PAIRWISE.WAVEFORM_HELD_OUT.DONE.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, save_json
from earthquake.metrics import match_picks
from earthquake.pairwise.cache import get_from_unified, load_unified_crop_pack, read_cache_lock
from earthquake.pairwise.model import (
    PairwiseScorer,
    PairwiseWindowConfig,
    assert_swap_antisymmetry,
)
from earthquake.utils import ensure_dir

OUT = artifacts_dir() / "results" / "pairwise_pilot"
CACHE = artifacts_dir() / "cache" / "pairwise_waveform"
REPORT = ROOT / "reports" / "pairwise"
SEED = 42
TAU_GRID = [0.50, 0.60, 0.70, 0.80, 0.90]
WIN = PairwiseWindowConfig()

SCALAR_KEYS_C = [
    "prob",
    "stead_prob",
    "ida_prob",
    "fixed_score",
    "resid_s",
    "resid_sp",
    "delta_sp",
    "src_stead",
    "src_ida",
    "src_both",
]

# Ablation masks over SCALAR_KEYS_C indices
ABLATION_MASKS = {
    "scalar_full": None,
    "scalar_without_resid_s": [4],  # zero resid_s
    "scalar_without_source": [7, 8, 9],
    "scalar_without_resid_s_and_source": [4, 7, 8, 9],
}


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _pick_free_gpu(min_free_gb: float = 4.0, max_util: int = 5) -> int:
    import os

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    smi = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
    )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.strip() != "":
        phys_list = [int(x) for x in visible.split(",") if x.strip() != ""]
    else:
        phys_list = list(range(torch.cuda.device_count()))
    phys_stats = {}
    for line in smi.strip().splitlines():
        idx_s, free_mib_s, util_s = [x.strip() for x in line.split(",")]
        phys_stats[int(idx_s)] = (float(free_mib_s) / 1024.0, int(float(util_s)))
    cands = []
    for local_i, phys in enumerate(phys_list):
        if phys not in phys_stats:
            continue
        free_gb, util = phys_stats[phys]
        if free_gb >= min_free_gb and util <= max_util:
            cands.append((free_gb, local_i, phys))
    if not cands:
        raise RuntimeError(f"no idle GPU among visible={phys_list}")
    cands.sort(reverse=True)
    return int(cands[0][1])


def scalar_vec(row: pd.Series, which: str, zero_idx: list[int] | None = None) -> np.ndarray:
    p = which
    src = str(row[f"{p}_source"])
    v = np.asarray(
        [
            float(row[f"{p}_prob"]),
            float(row[f"{p}_stead_prob"]),
            float(row[f"{p}_ida_prob"]),
            float(row[f"{p}_fixed_score"]),
            float(row[f"{p}_resid_s"]) if np.isfinite(row[f"{p}_resid_s"]) else 0.0,
            float(row[f"{p}_resid_sp"]) if np.isfinite(row[f"{p}_resid_sp"]) else 0.0,
            float(row[f"{p}_delta_sp"]) if np.isfinite(row[f"{p}_delta_sp"]) else 0.0,
            1.0 if src == "stead" else 0.0,
            1.0 if src == "ida" else 0.0,
            1.0 if src == "stead+ida" else 0.0,
        ],
        dtype=np.float32,
    )
    if zero_idx:
        for i in zero_idx:
            v[i] = 0.0
    return v


class CachePairDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        crop_pack: dict,
        *,
        decisive_only: bool,
        augment_swap: bool,
        load_waveform: bool = True,
        zero_scalar_idx: list[int] | None = None,
    ):
        self.df = df.reset_index(drop=True)
        if decisive_only:
            self.df = self.df[self.df["y_choose_c2"].isin([0, 1])].reset_index(drop=True)
        self.crop_pack = crop_pack
        self.augment_swap = bool(augment_swap)
        self.load_waveform = bool(load_waveform)
        self.zero_scalar_idx = zero_scalar_idx
        T = WIN.n_samples
        self._dummy_wave = torch.zeros(3, T)
        self._dummy_mask = torch.ones(T)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        tn = str(row.trace_name)
        s1 = scalar_vec(row, "c1", self.zero_scalar_idx)
        s2 = scalar_vec(row, "c2", self.zero_scalar_idx)
        y = row.y_choose_c2
        y = int(y) if y in (0, 1) else -1
        if self.load_waveform:
            x1, x2, m1, m2 = get_from_unified(self.crop_pack, tn)
            wave1, mask1 = torch.from_numpy(np.asarray(x1)), torch.from_numpy(np.asarray(m1))
            wave2, mask2 = torch.from_numpy(np.asarray(x2)), torch.from_numpy(np.asarray(m2))
        else:
            wave1 = self._dummy_wave.clone()
            mask1 = self._dummy_mask.clone()
            wave2 = self._dummy_wave.clone()
            mask2 = self._dummy_mask.clone()
        if self.augment_swap and (hash((tn, idx)) % 2 == 1):
            wave1, wave2 = wave2, wave1
            mask1, mask2 = mask2, mask1
            s1, s2 = s2, s1
            if y in (0, 1):
                y = 1 - y
        return {
            "wave1": wave1,
            "mask1": mask1,
            "scalar1": torch.from_numpy(s1),
            "wave2": wave2,
            "mask2": mask2,
            "scalar2": torch.from_numpy(s2),
            "y": torch.tensor(y, dtype=torch.float32),
            "trace_name": tn,
        }


def collate(batch):
    out = {}
    for k in ["wave1", "mask1", "scalar1", "wave2", "mask2", "scalar2", "y"]:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    out["trace_name"] = [b["trace_name"] for b in batch]
    return out


def train_loop(model, loader, opt, device, pos_weight: float = 1.0):
    model.train()
    total, n = 0.0, 0
    grad_wave = 0.0
    bce = nn.BCELoss(reduction="none")
    for batch in loader:
        y = batch["y"].to(device)
        mask = y >= 0
        if mask.sum() == 0:
            continue
        w1 = batch["wave1"].to(device) if model.use_waveform else None
        m1 = batch["mask1"].to(device) if model.use_waveform else None
        w2 = batch["wave2"].to(device) if model.use_waveform else None
        m2 = batch["mask2"].to(device) if model.use_waveform else None
        s1 = batch["scalar1"].to(device) if model.use_scalar else None
        s2 = batch["scalar2"].to(device) if model.use_scalar else None
        p = model(w1, m1, s1, w2, m2, s2)
        ypos = y[mask]
        w = torch.where(ypos > 0.5, torch.full_like(ypos, float(pos_weight)), torch.ones_like(ypos))
        loss = (w * bce(p[mask], ypos)).mean()
        opt.zero_grad()
        loss.backward()
        if model.encoder is not None:
            grad_wave = 0.0
            for p_ in model.encoder.parameters():
                if p_.grad is not None:
                    grad_wave += float(p_.grad.abs().sum().item())
        opt.step()
        total += float(loss.item()) * int(mask.sum().item())
        n += int(mask.sum().item())
    return {"loss": total / max(n, 1), "n": n, "grad_wave": grad_wave if model.encoder else None}


@torch.no_grad()
def predict_probs(model, df, crop_pack, device, batch_size=128, zero_scalar_idx=None) -> pd.DataFrame:
    load_wave = bool(getattr(model, "use_waveform", True))
    ds = CachePairDataset(
        df,
        crop_pack,
        decisive_only=False,
        augment_swap=False,
        load_waveform=load_wave,
        zero_scalar_idx=zero_scalar_idx,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    model.eval()
    rows = []
    for batch in loader:
        w1 = batch["wave1"].to(device) if model.use_waveform else None
        m1 = batch["mask1"].to(device) if model.use_waveform else None
        w2 = batch["wave2"].to(device) if model.use_waveform else None
        m2 = batch["mask2"].to(device) if model.use_waveform else None
        s1 = batch["scalar1"].to(device) if model.use_scalar else None
        s2 = batch["scalar2"].to(device) if model.use_scalar else None
        p = model(w1, m1, s1, w2, m2, s2).cpu().numpy()
        for tn, pi in zip(batch["trace_name"], p):
            rows.append({"trace_name": tn, "p_c2_gt_c1": float(pi)})
    return pd.DataFrame(rows)


def apply_policy(pairs: pd.DataFrame, probs: pd.DataFrame, tau: float):
    m = pairs.merge(probs, on="trace_name", how="left")
    pred = m["c1_sample"].to_numpy(float).copy()
    ge2 = m["n_candidates"].to_numpy(int) >= 2
    p = m["p_c2_gt_c1"].to_numpy(float)
    switch = ge2 & np.isfinite(p) & (p > tau)
    pred[switch] = m["c2_sample"].to_numpy(float)[switch]
    return pred, switch


def metrics_from_pred(pred, true, sr):
    m = match_picks(pred, true, sr, windows_s=(0.1, 0.5))
    return {
        "f1@0.1": float(m["f1@0.1s"]),
        "f1@0.5": float(m["f1@0.5s"]),
        "precision@0.1": float(m["precision@0.1s"]),
        "recall@0.1": float(m["recall@0.1s"]),
        "precision@0.5": float(m["precision@0.5s"]),
        "recall@0.5": float(m["recall@0.5s"]),
        "miss_rate": float(m["miss_rate"]),
        "detected_ae_median": float(m["detected_ae_median"]),
        "detected_ae_p95": float(m["detected_ae_p95"]),
        "detected_ae_mae": float(m.get("detected_ae_mae", np.nan)),
    }


def resid_control_pred(pairs: pd.DataFrame) -> np.ndarray:
    lam, sig = 1.0, 0.5
    out = []
    for _, row in pairs.iterrows():
        if int(row.n_candidates) < 2 or not np.isfinite(row.c2_sample):
            out.append(float(row.c1_sample))
            continue
        r1 = float(row.c1_resid_s) if np.isfinite(row.c1_resid_s) else 99.0
        r2 = float(row.c2_resid_s) if np.isfinite(row.c2_resid_s) else 99.0
        s1 = float(row.c1_fixed_score) + lam * float(np.exp(-0.5 * (r1 / sig) ** 2))
        s2 = float(row.c2_fixed_score) + lam * float(np.exp(-0.5 * (r2 / sig) ** 2))
        out.append(float(row.c2_sample) if s2 > s1 else float(row.c1_sample))
    return np.asarray(out, float)


def choose_tau(cal_pairs, cal_probs) -> tuple[float, dict]:
    best = None
    rows = []
    for tau in TAU_GRID:
        pred, switch = apply_policy(cal_pairs, cal_probs, tau)
        base_ok = cal_pairs.c1_ok.to_numpy(bool)
        c2_ok = cal_pairs.c2_ok.to_numpy(bool)
        fixes = int(np.sum(switch & (~base_ok) & c2_ok))
        breaks = int(np.sum(switch & base_ok & (~c2_ok)))
        prec = fixes / max(fixes + breaks, 1)
        net = fixes - breaks
        rows.append(
            {
                "tau": tau,
                "fixes": fixes,
                "breaks": breaks,
                "net": net,
                "precision": prec,
                "n_switch": int(switch.sum()),
                "switch_rate": float(switch.mean()),
            }
        )
        if prec >= 0.80:
            cand = (net, -tau, tau)
            if best is None or cand > best[0]:
                best = (cand, tau, rows[-1])
    if best is None:
        return 1.0, {"tau": 1.0, "note": "no tau met precision>=0.80", "grid": rows}
    return float(best[1]), {"selected": best[2], "grid": rows}


def pairwise_switch_stats(pairs, pred, switch, true, sr, base_ok):
    new_ok = np.abs(pred - true) / sr <= 0.5
    fixes = int(np.sum(switch & (~base_ok) & new_ok))
    breaks = int(np.sum(switch & base_ok & (~new_ok)))
    q1 = base_ok & (~pairs.c2_ok.to_numpy(bool))
    q2 = (~base_ok) & pairs.c2_ok.to_numpy(bool)
    q1_breaks = int(np.sum(switch & q1 & (~new_ok)))
    q2_fixes = int(np.sum(switch & q2 & new_ok))
    return {
        "fixes": fixes,
        "breaks": breaks,
        "net_fixes": fixes - breaks,
        "switch_count": int(switch.sum()),
        "switch_rate": float(switch.mean()),
        "switch_precision": fixes / max(fixes + breaks, 1),
        "q1_n": int(q1.sum()),
        "q1_break_rate": q1_breaks / max(int(q1.sum()), 1),
        "q2_n": int(q2.sum()),
        "q2_recovery": q2_fixes / max(int(q2.sum()), 1),
    }


def train_one(
    name,
    kw,
    train_df,
    cal_df,
    crop_pack,
    device,
    pos_weight,
    zero_scalar_idx=None,
):
    n_scalar = len(SCALAR_KEYS_C)
    model = PairwiseScorer(n_scalar, **kw).to(device)
    if model.n_parameters() > 300_000:
        raise RuntimeError(f"too many params: {model.n_parameters()}")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ds = CachePairDataset(
        train_df,
        crop_pack,
        decisive_only=True,
        augment_swap=True,
        load_waveform=bool(kw.get("use_waveform", True)),
        zero_scalar_idx=zero_scalar_idx,
    )
    loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0, collate_fn=collate)
    best_state, best_net, patience = None, -(10**9), 0
    cal_ge2 = cal_df[cal_df.n_candidates >= 2]
    hist = []
    for ep in range(20):
        st = train_loop(model, loader, opt, device, pos_weight=pos_weight)
        pr = predict_probs(model, cal_ge2, crop_pack, device, zero_scalar_idx=zero_scalar_idx)
        _, switch = apply_policy(cal_df, pr, 0.5)
        base_ok = cal_df.c1_ok.to_numpy(bool)
        c2_ok = cal_df.c2_ok.to_numpy(bool)
        net = int(np.sum(switch & (~base_ok) & c2_ok) - np.sum(switch & base_ok & (~c2_ok)))
        hist.append({"ep": ep, "loss": st["loss"], "cal_net@0.5": net, "grad_wave": st.get("grad_wave")})
        print(f"  {name} ep{ep} loss={st['loss']:.4f} cal_net@0.5={net}", flush=True)
        if net > best_net:
            best_net = net
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 4:
                break
    model.load_state_dict(best_state)
    ckpt = OUT / f"ckpt_{name}_seed42.pt"
    torch.save({"state_dict": best_state, "n_scalar": n_scalar, "zero_scalar_idx": zero_scalar_idx, **kw}, ckpt)
    return model, {"best_cal_net_at_tau0.5": best_net, "ckpt": str(ckpt), "ckpt_sha256": _sha_file(ckpt), "hist": hist, "n_parameters": model.n_parameters()}


def event_bootstrap_delta(pred_a, pred_b, true, sr, events, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    idx_by = {}
    for i, e in enumerate(events):
        idx_by.setdefault(str(e), []).append(i)
    lists = [np.asarray(v, int) for v in idx_by.values()]
    d05 = np.empty(n_boot)
    d01 = np.empty(n_boot)
    dp95 = np.empty(n_boot)
    for b in range(n_boot):
        draw = rng.integers(0, len(lists), size=len(lists))
        ix = np.concatenate([lists[j] for j in draw])
        ma = metrics_from_pred(pred_a[ix], true[ix], sr[ix])
        mb = metrics_from_pred(pred_b[ix], true[ix], sr[ix])
        d05[b] = ma["f1@0.5"] - mb["f1@0.5"]
        d01[b] = ma["f1@0.1"] - mb["f1@0.1"]
        dp95[b] = ma["detected_ae_p95"] - mb["detected_ae_p95"]
    def pack(arr):
        return {"mean": float(arr.mean()), "ci95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]}
    return {"n_boot": n_boot, "seed": seed, "delta_f1@0.5": pack(d05), "delta_f1@0.1": pack(d01), "delta_p95": pack(dp95)}


def hard_fail(reason: str, payload: dict) -> None:
    """Write NEW fail marker without deleting original PAIRWISE.PILOT.FAILED."""
    save_json(payload, OUT / "PAIRWISE.PILOT.FAILED_AFTER_RESUME.json")
    (OUT / "PAIRWISE.PILOT.FAILED_AFTER_RESUME").write_text(reason + "\n", encoding="utf-8")
    print("PAIRWISE.PILOT.FAILED_AFTER_RESUME:", reason, flush=True)
    raise SystemExit(1)


def main() -> None:
    t0 = time.time()
    ensure_dir(OUT)
    ensure_dir(REPORT)

    # Provenance checks
    if not (OUT / "PAIRWISE.PILOT.FAILED").exists():
        raise SystemExit("original PAIRWISE.PILOT.FAILED missing — refuse to rewrite history")
    if not (OUT / "SPLIT.LOCK.json").exists():
        raise SystemExit("SPLIT.LOCK missing")
    if not (CACHE / "CACHE.LOCK.json").exists():
        raise SystemExit("waveform cache not built")
    eq_path = OUT / "waveform_cache_equivalence.json"
    if not eq_path.exists():
        raise SystemExit("run audit_pairwise_waveform_cache.py first")
    eq = json.loads(eq_path.read_text())
    if not eq.get("passed"):
        hard_fail("cache_equivalence_failed", {"eq": eq})

    if (OUT / "PAIRWISE.WAVEFORM_HELD_OUT.DONE").exists():
        raise SystemExit("waveform held-out already executed once")

    split_lock = json.loads((OUT / "SPLIT.LOCK.json").read_text())
    cache_lock = read_cache_lock(CACHE)

    # RESUMED marker (after cache audit pass)
    resumed = {
        "reason": "waveform_cache_built_and_equivalence_passed",
        "original_failure": "waveform_pairwise_no_go_incomplete_io",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cache_lock_hash": cache_lock.get("cache_lock_sha256"),
        "split_lock_event_sha256": split_lock["event_sha256"],
        "git_commit": _git_commit(),
        "model_config_hash": hashlib.sha256(
            json.dumps(
                {"lr": 1e-3, "wd": 1e-4, "max_epoch": 20, "patience": 4, "window": [1.5, 2.5], "seed": SEED},
                sort_keys=True,
            ).encode()
        ).hexdigest(),
        "original_FAILED_preserved": True,
    }
    save_json(resumed, OUT / "PAIRWISE.PILOT.RESUMED_AFTER_IO_FIX.json")
    (OUT / "PAIRWISE.PILOT.RESUMED_AFTER_IO_FIX").write_text(
        json.dumps(resumed, indent=2) + "\n", encoding="utf-8"
    )
    print("RESUMED_AFTER_IO_FIX written", flush=True)

    gpu_i = _pick_free_gpu()
    torch.cuda.set_device(gpu_i)
    device = torch.device(f"cuda:{gpu_i}")
    print(f"using GPU local={gpu_i}", flush=True)

    pairs = pd.read_parquet(OUT / "pairs_ranker_train.parquet")
    # confirm guard
    conf_path = artifacts_dir() / "results/stage6/splits_full/stage6_internal_confirm_events.txt"
    confirm_events = set(conf_path.read_text().split()) if conf_path.exists() else set()
    if set(pairs.event_id.astype(str)) & confirm_events:
        hard_fail("confirm_events_in_pairs", {})

    train_df = pairs[pairs.split == "train"].copy()
    cal_df = pairs[pairs.split == "calibration"].copy()
    eval_df = pairs[pairs.split == "heldout_eval"].copy()
    crop_pack = load_unified_crop_pack(CACHE)

    # Verify split hashes unchanged
    def _sha_lines(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    for sp, key in [("train", "train"), ("calibration", "calibration"), ("heldout_eval", "heldout_eval")]:
        got = _sha_lines(OUT / f"events_{sp if sp != 'heldout_eval' else 'heldout_eval'}.txt")
        # files named events_heldout_eval.txt
        exp = split_lock["event_sha256"][key if key != "heldout_eval" else "heldout_eval"]
        # map
    assert _sha_lines(OUT / "events_train.txt") == split_lock["event_sha256"]["train"]
    assert _sha_lines(OUT / "events_calibration.txt") == split_lock["event_sha256"]["calibration"]
    assert _sha_lines(OUT / "events_heldout_eval.txt") == split_lock["event_sha256"]["heldout_eval"]

    dec_all = train_df[train_df.y_choose_c2.isin([0, 1])]
    n_pos = int((dec_all.y_choose_c2 == 1).sum())
    n_neg = int((dec_all.y_choose_c2 == 0).sum())
    pos_weight = float(n_neg / max(n_pos, 1))
    print(f"pos_weight={pos_weight:.3f} n_pos={n_pos} n_neg={n_neg}", flush=True)

    # ---- overfit / smoke on cache ----
    print("overfit 64...", flush=True)
    pos = dec_all[dec_all.y_choose_c2 == 1].head(32)
    neg = dec_all[dec_all.y_choose_c2 == 0].head(32)
    tiny = pd.concat([pos, neg]).sample(frac=1.0, random_state=SEED)
    model = PairwiseScorer(10, use_waveform=True, use_scalar=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ds = CachePairDataset(tiny, crop_pack, decisive_only=True, augment_swap=True)
    loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=0, collate_fn=collate)
    losses = []
    for ep in range(30):
        losses.append(train_loop(model, loader, opt, device)["loss"])
    probs = predict_probs(model, tiny, crop_pack, device)
    merged = tiny.merge(probs, on="trace_name")
    acc = float((((merged.p_c2_gt_c1 > 0.5).astype(int)) == merged.y_choose_c2.astype(int)).mean())
    batch = collate([ds[i] for i in range(min(8, len(ds)))])
    for k in batch:
        if torch.is_tensor(batch[k]):
            batch[k] = batch[k].to(device)
    swap_err = assert_swap_antisymmetry(model, batch)
    if not ((losses[0] - losses[-1] > 0.05) and (acc > 0.65) and (swap_err < 1e-5)):
        hard_fail("overfit_failed", {"loss0": losses[0], "lossN": losses[-1], "acc": acc, "swap": swap_err})
    print("overfit OK", acc, swap_err, flush=True)

    print("smoke 512...", flush=True)
    pos = dec_all[dec_all.y_choose_c2 == 1].head(256)
    neg = dec_all[dec_all.y_choose_c2 == 0].head(256)
    smoke = pd.concat([pos, neg]).sample(frac=1.0, random_state=SEED)
    model = PairwiseScorer(10, use_waveform=True, use_scalar=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ds = CachePairDataset(smoke, crop_pack, decisive_only=True, augment_swap=True)
    loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0, collate_fn=collate)
    for ep in range(15):
        st = train_loop(model, loader, opt, device)
        if st.get("grad_wave", 1) == 0:
            hard_fail("waveform_grad_zero", {})
    probs = predict_probs(model, smoke, crop_pack, device)
    merged = smoke.merge(probs, on="trace_name")
    acc = float((((merged.p_c2_gt_c1 > 0.5).astype(int)) == merged.y_choose_c2.astype(int)).mean())
    frac = float((merged.p_c2_gt_c1 > 0.5).mean())
    if acc < 0.60 or frac < 0.1 or frac > 0.9:
        hard_fail("smoke_failed", {"acc": acc, "frac": frac})
    print("smoke OK", acc, frac, flush=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ---- train waveform_only + waveform_plus_scalar ----
    train_summary = {}
    models = {}
    for name, kw in [
        ("waveform_only", dict(use_waveform=True, use_scalar=False)),
        ("waveform_plus_scalar", dict(use_waveform=True, use_scalar=True)),
    ]:
        print("training", name, flush=True)
        m, meta = train_one(name, kw, train_df, cal_df, crop_pack, device, pos_weight)
        models[name] = m
        train_summary[name] = meta

    # scalar ablations (lightweight)
    abl_models = {}
    abl_rows = []
    for abl_name, zidx in ABLATION_MASKS.items():
        print("training ablation", abl_name, flush=True)
        m, meta = train_one(
            abl_name,
            dict(use_waveform=False, use_scalar=True),
            train_df,
            cal_df,
            crop_pack,
            device,
            pos_weight,
            zero_scalar_idx=zidx,
        )
        abl_models[abl_name] = (m, zidx)
        train_summary[abl_name] = meta

    # optional waveform+ablated scalar
    for abl_name, zidx in [
        ("waveform_plus_scalar_without_resid_s", [4]),
        ("waveform_plus_scalar_without_source", [7, 8, 9]),
    ]:
        print("training", abl_name, flush=True)
        m, meta = train_one(
            abl_name,
            dict(use_waveform=True, use_scalar=True),
            train_df,
            cal_df,
            crop_pack,
            device,
            pos_weight,
            zero_scalar_idx=zidx,
        )
        abl_models[abl_name] = (m, zidx)
        train_summary[abl_name] = meta

    # load prior scalar_pairwise ckpt for protocol continuity
    scalar_ckpt = OUT / "ckpt_scalar_pairwise_seed42.pt"
    if not scalar_ckpt.exists():
        hard_fail("missing_scalar_ckpt", {})
    blob = torch.load(scalar_ckpt, map_location="cpu", weights_only=False)
    scalar_model = PairwiseScorer(10, use_waveform=False, use_scalar=True).to(device)
    scalar_model.load_state_dict(blob["state_dict"])
    models["scalar_pairwise"] = scalar_model
    train_summary["scalar_pairwise"] = {"ckpt": str(scalar_ckpt), "ckpt_sha256": _sha_file(scalar_ckpt), "reused": True}

    # calibration tau for primary waveform_plus_scalar
    primary = "waveform_plus_scalar"
    cal_probs = predict_probs(models[primary], cal_df[cal_df.n_candidates >= 2], crop_pack, device)
    tau, tau_info = choose_tau(cal_df, cal_probs)
    print("selected tau", tau, tau_info.get("selected"), flush=True)

    # also report tau for waveform_only / scalar with same rule
    tau_by = {"waveform_plus_scalar": (tau, tau_info)}
    for name in ["waveform_only", "scalar_pairwise"]:
        z = None
        pr = predict_probs(models[name], cal_df[cal_df.n_candidates >= 2], crop_pack, device, zero_scalar_idx=z)
        t, info = choose_tau(cal_df, pr)
        tau_by[name] = (t, info)
        print(f"tau[{name}]={t}", info.get("selected"), flush=True)

    method_lock = {
        "method": primary,
        "seed": SEED,
        "tau": tau,
        "tau_selection": tau_info,
        "tau_by_model": {k: v[1].get("selected", v[1]) for k, v in tau_by.items()},
        "window": {"pre_s": WIN.pre_s, "post_s": WIN.post_s, "n_samples": WIN.n_samples},
        "scalar_features": SCALAR_KEYS_C,
        "n_parameters": models[primary].n_parameters(),
        "ckpt_sha256": train_summary[primary]["ckpt_sha256"],
        "cache_lock_sha256": cache_lock.get("cache_lock_sha256"),
        "split_lock_event_sha256": split_lock["event_sha256"],
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "max_epoch": 20,
        "patience": 4,
        "resumed_after_io_fix": True,
        "original_FAILED_preserved": True,
        "code_note": "pairwise waveform resume; not Stage-6 method lock",
    }
    save_json(method_lock, OUT / "PAIRWISE.METHOD.LOCK.json")
    save_json(train_summary, OUT / "waveform_train_summary.json")

    # ---- held-out once ----
    print("held-out evaluation (WAVEFORM ONCE)...", flush=True)
    (OUT / "PAIRWISE.WAVEFORM_HELD_OUT.DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ") + "\n")

    true = eval_df.true_s_sample.to_numpy(float)
    sr = eval_df.sampling_rate_hz.to_numpy(float)
    base_ok = eval_df.c1_ok.to_numpy(bool)
    fixed_pred = eval_df.c1_sample.to_numpy(float)
    resid_pred = resid_control_pred(eval_df)

    ablation = {}
    ablation["fixed_rescore_UNION"] = {
        **metrics_from_pred(fixed_pred, true, sr),
        **{"fixes": 0, "breaks": 0, "net_fixes": 0, "switch_precision": float("nan"), "switch_rate": 0.0},
    }
    rmet = metrics_from_pred(resid_pred, true, sr)
    switched_r = np.abs(resid_pred - fixed_pred) > 0.5
    new_ok_r = np.abs(resid_pred - true) / sr <= 0.5
    fr = int(np.sum(switched_r & (~base_ok) & new_ok_r))
    br = int(np.sum(switched_r & base_ok & (~new_ok_r)))
    ablation["resid_s_control"] = {
        **rmet,
        "delta_f1@0.5": rmet["f1@0.5"] - ablation["fixed_rescore_UNION"]["f1@0.5"],
        "fixes": fr,
        "breaks": br,
        "net_fixes": fr - br,
        "switch_precision": fr / max(fr + br, 1),
        "switch_rate": float(switched_r.mean()),
    }

    preds = {}
    for name in ["scalar_pairwise", "waveform_only", "waveform_plus_scalar"]:
        t_use = tau_by[name][0]
        z = None
        pr = predict_probs(models[name], eval_df[eval_df.n_candidates >= 2], crop_pack, device, zero_scalar_idx=z)
        pred, switch = apply_policy(eval_df, pr, t_use)
        preds[name] = (pred, switch, pr, t_use)
        met = metrics_from_pred(pred, true, sr)
        st = pairwise_switch_stats(eval_df, pred, switch, true, sr, base_ok)
        ablation[name] = {
            **met,
            **st,
            "delta_f1@0.5": met["f1@0.5"] - ablation["fixed_rescore_UNION"]["f1@0.5"],
            "delta_f1@0.1": met["f1@0.1"] - ablation["fixed_rescore_UNION"]["f1@0.1"],
            "delta_p95": met["detected_ae_p95"] - ablation["fixed_rescore_UNION"]["detected_ae_p95"],
            "tau": t_use,
        }
        expected = st["net_fixes"] / len(eval_df)
        if abs(ablation[name]["delta_f1@0.5"] - expected) > 0.002:
            print("WARN deltaF1 mismatch", name, ablation[name]["delta_f1@0.5"], expected)

    # scalar feature ablations on held-out (each with own cal tau)
    abl_metrics = []
    for abl_name, (m, zidx) in abl_models.items():
        pr = predict_probs(m, cal_df[cal_df.n_candidates >= 2], crop_pack, device, zero_scalar_idx=zidx)
        t_abl, info = choose_tau(cal_df, pr)
        pr_e = predict_probs(m, eval_df[eval_df.n_candidates >= 2], crop_pack, device, zero_scalar_idx=zidx)
        pred, switch = apply_policy(eval_df, pr_e, t_abl)
        met = metrics_from_pred(pred, true, sr)
        st = pairwise_switch_stats(eval_df, pred, switch, true, sr, base_ok)
        row = {
            "model": abl_name,
            "tau": t_abl,
            "f1@0.5": met["f1@0.5"],
            "delta_f1@0.5_vs_fixed": met["f1@0.5"] - ablation["fixed_rescore_UNION"]["f1@0.5"],
            **{k: st[k] for k in ("fixes", "breaks", "net_fixes", "switch_precision", "q2_recovery")},
        }
        abl_metrics.append(row)
        ablation[abl_name] = {**met, **st, "tau": t_abl, "delta_f1@0.5": row["delta_f1@0.5_vs_fixed"]}
    pd.DataFrame(abl_metrics).to_csv(OUT / "waveform_scalar_ablation.csv", index=False)

    # waveform vs scalar case analysis
    pred_s, sw_s, _, _ = preds["scalar_pairwise"]
    pred_w, sw_w, _, _ = preds["waveform_plus_scalar"]
    ok_s = np.abs(pred_s - true) / sr <= 0.5
    ok_w = np.abs(pred_w - true) / sr <= 0.5
    cases = eval_df.copy()
    cases["ok_scalar"] = ok_s
    cases["ok_wps"] = ok_w
    cases["sw_scalar"] = sw_s
    cases["sw_wps"] = sw_w
    cases["pred_scalar"] = pred_s
    cases["pred_wps"] = pred_w
    wf_fix_s = (~ok_s) & ok_w
    wf_break_s = ok_s & (~ok_w)
    both_ok = ok_s & ok_w
    both_bad = (~ok_s) & (~ok_w)
    case_summary = {
        "n_waveform_fixes_over_scalar": int(wf_fix_s.sum()),
        "n_waveform_breaks_over_scalar": int(wf_break_s.sum()),
        "net_waveform_vs_scalar": int(wf_fix_s.sum() - wf_break_s.sum()),
        "both_correct": int(both_ok.sum()),
        "both_wrong": int(both_bad.sum()),
        "scalar_only_correct": int((ok_s & ~ok_w).sum()),
        "waveform_only_correct": int((~ok_s & ok_w).sum()),
    }
    # dump interesting cases
    focus = cases[wf_fix_s | wf_break_s].copy()
    focus["case_type"] = np.where(wf_fix_s[wf_fix_s | wf_break_s], "wf_fixes_scalar", "wf_breaks_scalar")
    # Q type
    focus["quadrant"] = np.where(
        focus.c1_ok & ~focus.c2_ok,
        "Q1",
        np.where(~focus.c1_ok & focus.c2_ok, "Q2", np.where(focus.c1_ok & focus.c2_ok, "both_ok_cand", "both_bad_cand")),
    )
    focus["cand_sep_s"] = (focus.c2_sample - focus.c1_sample).abs() / focus.sampling_rate_hz
    focus["margin_fixed"] = focus.margin_fixed
    cols = [
        "trace_name",
        "event_id",
        "case_type",
        "quadrant",
        "c1_source",
        "c2_source",
        "c1_resid_s",
        "c2_resid_s",
        "margin_fixed",
        "cand_sep_s",
        "snr_db",
        "ok_scalar",
        "ok_wps",
    ]
    focus[[c for c in cols if c in focus.columns]].to_csv(OUT / "waveform_vs_scalar_cases.csv", index=False)
    save_json(case_summary, OUT / "waveform_vs_scalar_case_summary.json")

    # bootstraps
    boot_ws = event_bootstrap_delta(pred_w, pred_s, true, sr, eval_df.event_id.to_numpy(), n_boot=2000, seed=SEED)
    boot_sf = event_bootstrap_delta(pred_s, fixed_pred, true, sr, eval_df.event_id.to_numpy(), n_boot=2000, seed=SEED)
    boot = {
        "waveform_plus_scalar_vs_scalar_pairwise": boot_ws,
        "scalar_pairwise_vs_fixed": boot_sf,
        "delta_waveform_point": float(ablation["waveform_plus_scalar"]["f1@0.5"] - ablation["scalar_pairwise"]["f1@0.5"]),
    }
    save_json(boot, OUT / "waveform_bootstrap.json")
    save_json(boot, OUT / "bootstrap.json")  # update shared name carefully; keep old? overwrite with richer

    # per-model metrics files
    save_json(ablation["waveform_only"], OUT / "waveform_only_metrics.json")
    save_json(ablation["waveform_plus_scalar"], OUT / "waveform_plus_scalar_metrics.json")
    save_json(ablation, OUT / "ablation_metrics.json")

    delta_wf = boot["delta_waveform_point"]
    gate_a = {
        "best_pairwise_delta_f1_ge_0.010": max(
            ablation["scalar_pairwise"]["delta_f1@0.5"],
            ablation["waveform_plus_scalar"]["delta_f1@0.5"],
            ablation["waveform_only"]["delta_f1@0.5"],
        )
        >= 0.010,
        "p95_not_materially_worse_best": True,  # filled below
    }
    best_name = max(
        ["scalar_pairwise", "waveform_plus_scalar", "waveform_only"],
        key=lambda n: ablation[n]["f1@0.5"],
    )
    gate_a["p95_not_materially_worse_best"] = ablation[best_name]["delta_p95"] <= 0.05
    gate_a["passed"] = all(gate_a[k] for k in gate_a if k != "passed")

    ci_lo = boot_ws["delta_f1@0.5"]["ci95"][0]
    gate_b = {
        "delta_f1_ge_0.003": delta_wf >= 0.003,
        "bootstrap_ci_excludes_0": ci_lo > 0,
        "p95_not_worse_0.05": (ablation["waveform_plus_scalar"]["detected_ae_p95"] - ablation["scalar_pairwise"]["detected_ae_p95"])
        <= 0.05,
        "or_q2_and_precision": (
            ablation["waveform_plus_scalar"]["q2_recovery"] > ablation["scalar_pairwise"]["q2_recovery"] + 0.02
            and ablation["waveform_plus_scalar"]["switch_precision"] >= 0.80
        ),
        "not_material_lt_0.001": delta_wf < 0.001,
        "worse_than_scalar": delta_wf < 0,
    }
    if gate_b["worse_than_scalar"]:
        gate_b["verdict"] = "NO_GO_waveform_branch"
        gate_b["passed"] = False
    elif gate_b["not_material_lt_0.001"]:
        gate_b["verdict"] = "waveform_increment_not_material"
        gate_b["passed"] = False
    elif (gate_b["delta_f1_ge_0.003"] and gate_b["bootstrap_ci_excludes_0"] and gate_b["p95_not_worse_0.05"]) or gate_b[
        "or_q2_and_precision"
    ]:
        gate_b["verdict"] = "waveform_has_independent_value"
        gate_b["passed"] = True
    else:
        gate_b["verdict"] = "waveform_increment_inconclusive_or_weak"
        gate_b["passed"] = False

    # overall pilot: require waveform path completed + gate A; gate B informational for PASSED mark
    # User: Gate A + Gate B for this phase. PASSED only if both meaningful.
    overall_passed = bool(gate_a["passed"] and gate_b["passed"])
    # Also keep original 11-gate style for waveform_plus_scalar vs fixed
    gates_legacy = {
        "1_delta_f1_vs_fixed_ge_0.010": ablation["waveform_plus_scalar"]["delta_f1@0.5"] >= 0.010,
        "2_delta_f1_vs_resid_ge_0.004": (
            ablation["waveform_plus_scalar"]["f1@0.5"] - ablation["resid_s_control"]["f1@0.5"]
        )
        >= 0.004,
        "waveform_vs_scalar_delta": delta_wf,
        "gate_a": gate_a,
        "gate_b": gate_b,
    }

    final = {
        "gates_A": gate_a,
        "gates_B": gate_b,
        "gates_legacy_primary": gates_legacy,
        "passed": overall_passed,
        "ablation": {k: ablation[k] for k in ablation},
        "case_summary": case_summary,
        "bootstrap": boot,
        "tau": tau,
        "tau_by_model": {k: v[0] for k, v in tau_by.items()},
        "n_eval": len(eval_df),
        "n_parameters": {
            "waveform_only": models["waveform_only"].n_parameters(),
            "waveform_plus_scalar": models["waveform_plus_scalar"].n_parameters(),
            "scalar_pairwise": models["scalar_pairwise"].n_parameters(),
        },
        "full_dev_read": False,
        "confirm_read": False,
        "original_FAILED_preserved": True,
        "elapsed_s": time.time() - t0,
        "recommend_full_dev": False,
    }
    save_json(final, OUT / "waveform_pilot_final.json")

    # markdown report
    lines = [
        "# Waveform pairwise pilot report (IO-fix resume)",
        "",
        f"Original FAILED preserved: `PAIRWISE.PILOT.FAILED` (incomplete_io).",
        f"Resumed: `PAIRWISE.PILOT.RESUMED_AFTER_IO_FIX`.",
        "",
        f"Primary: `{primary}`  tau={tau}",
        f"Params waveform_only={models['waveform_only'].n_parameters()}  "
        f"waveform_plus_scalar={models['waveform_plus_scalar'].n_parameters()}",
        "",
        "## Held-out F1@0.5",
        json.dumps(
            {
                k: {
                    kk: ablation[k].get(kk)
                    for kk in (
                        "f1@0.5",
                        "delta_f1@0.5",
                        "fixes",
                        "breaks",
                        "net_fixes",
                        "switch_precision",
                        "q2_recovery",
                        "detected_ae_p95",
                        "tau",
                    )
                }
                for k in [
                    "fixed_rescore_UNION",
                    "resid_s_control",
                    "scalar_pairwise",
                    "waveform_only",
                    "waveform_plus_scalar",
                ]
            },
            indent=2,
        ),
        "",
        f"## delta_waveform (wps - scalar) = {delta_wf:.6f}",
        f"bootstrap CI: {boot_ws['delta_f1@0.5']}",
        "",
        "## Case summary vs scalar",
        json.dumps(case_summary, indent=2),
        "",
        "## Gate A / Gate B",
        json.dumps({"A": gate_a, "B": gate_b}, indent=2),
        "",
        f"Verdict: {'PAIRWISE.PILOT.PASSED' if overall_passed else 'PAIRWISE.PILOT.FAILED_AFTER_RESUME'}",
        "",
        "Hard stop. No full-dev. No confirm. recommend_full_dev=false.",
        "",
    ]
    (REPORT / "waveform_pairwise_pilot_report.md").write_text("\n".join(lines), encoding="utf-8")

    if overall_passed:
        (OUT / "PAIRWISE.PILOT.PASSED").write_text("passed_after_io_fix\n", encoding="utf-8")
        save_json(final, OUT / "PAIRWISE.PILOT.PASSED.json")
        print("PAIRWISE.PILOT.PASSED", flush=True)
    else:
        hard_fail(gate_b.get("verdict", "waveform_pilot_failed"), final)


if __name__ == "__main__":
    main()
