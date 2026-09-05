#!/usr/bin/env python
"""Strictly controlled waveform top1-vs-top2 pairwise pilot (seed=42 only).

Hard-stops after PASS/FAIL. Never reads confirm or phaseB full-dev.
"""

from __future__ import annotations

import hashlib
import json
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

from earthquake.config import artifacts_dir, resolve_instance_root, save_json
from earthquake.data.hdf5_reader import InstanceHDF5Reader
from earthquake.metrics import match_picks
from earthquake.pairwise.model import (
    PairwiseScorer,
    PairwiseWindowConfig,
    assert_swap_antisymmetry,
    crop_candidate_window,
)
from earthquake.utils import ensure_dir

OUT = artifacts_dir() / "results" / "pairwise_pilot"
SEED = 42
TAU_GRID = [0.50, 0.60, 0.70, 0.80, 0.90]
DEVICE = None  # chosen at runtime among truly free GPUs
WIN = PairwiseWindowConfig()


def _pick_free_gpu(min_free_gb: float = 8.0, max_util: int = 5) -> int:
    """Return a *local* CUDA device index that is idle.

    Respects CUDA_VISIBLE_DEVICES: returned index is for torch.cuda (0..n_visible-1).
    """
    import os
    import subprocess

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    smi = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
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

    candidates = []
    for local_i, phys in enumerate(phys_list):
        if phys not in phys_stats:
            continue
        free_gb, util = phys_stats[phys]
        if free_gb >= min_free_gb and util <= max_util:
            candidates.append((free_gb, local_i, phys))
    if not candidates:
        raise RuntimeError(
            f"no idle GPU among visible={phys_list} "
            f"(need free>={min_free_gb}GiB and util<={max_util}%); refuse to share busy devices"
        )
    candidates.sort(reverse=True)
    return int(candidates[0][1])

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


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _block(msg: str) -> None:
    p = OUT / "PAIRWISE.TRAIN_BLOCKED"
    p.write_text(msg + "\n", encoding="utf-8")
    raise SystemExit(f"PAIRWISE.TRAIN_BLOCKED: {msg}")


def _fail(msg: str, payload: dict) -> None:
    save_json(payload, OUT / "PAIRWISE.PILOT.FAILED.json")
    (OUT / "PAIRWISE.PILOT.FAILED").write_text(msg + "\n", encoding="utf-8")
    print("PAIRWISE.PILOT.FAILED:", msg)
    raise SystemExit(1)


def _pass(payload: dict) -> None:
    save_json(payload, OUT / "PAIRWISE.PILOT.PASSED.json")
    (OUT / "PAIRWISE.PILOT.PASSED").write_text("passed\n", encoding="utf-8")
    print("PAIRWISE.PILOT.PASSED")


def scalar_vec(row: pd.Series, which: str) -> np.ndarray:
    p = which  # 'c1' or 'c2'
    src = str(row[f"{p}_source"])
    return np.asarray(
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


class PairDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        reader: InstanceHDF5Reader | None,
        *,
        decisive_only: bool,
        augment_swap: bool,
        load_waveform: bool = True,
        crop_cache: dict | None = None,
    ):
        self.df = df.reset_index(drop=True)
        if decisive_only:
            self.df = self.df[self.df["y_choose_c2"].isin([0, 1])].reset_index(drop=True)
        self.reader = reader
        self.augment_swap = bool(augment_swap)
        self.load_waveform = bool(load_waveform)
        self.crop_cache = crop_cache  # {trace_name: (wave1,wave2,mask1,mask2)} or memmap pack
        self.wave_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        T = WIN.n_samples
        self._dummy_wave = torch.zeros(3, T)
        self._dummy_mask = torch.ones(T)

    def __len__(self) -> int:
        return len(self.df)

    def _from_cache(self, tn: str):
        pack = self.crop_cache
        assert pack is not None
        idx = pack["index"][tn]
        x1 = np.asarray(pack["waves"][idx, 0])
        x2 = np.asarray(pack["waves"][idx, 1])
        m1 = np.asarray(pack["masks"][idx, 0])
        m2 = np.asarray(pack["masks"][idx, 1])
        return x1, x2, m1, m2

    def _wave_stats(self, tn: str):
        assert self.reader is not None
        if tn not in self.wave_cache:
            w = self.reader.read_waveform(tn)
            zne = np.stack([w[2], w[1], w[0]], axis=0).astype(np.float32)
            mean = zne.mean(axis=1)
            std = np.maximum(zne.std(axis=1), 1e-6)
            self.wave_cache[tn] = (zne, mean, std)
        return self.wave_cache[tn]

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        tn = str(row.trace_name)
        s1 = scalar_vec(row, "c1")
        s2 = scalar_vec(row, "c2")
        y = row.y_choose_c2
        y = int(y) if y in (0, 1) else -1
        if self.load_waveform:
            if self.crop_cache is not None and tn in self.crop_cache["index"]:
                x1, x2, m1, m2 = self._from_cache(tn)
            else:
                zne, mean, std = self._wave_stats(tn)
                x1, m1 = crop_candidate_window(
                    zne, float(row.c1_sample), cfg=WIN, full_trace_mean=mean, full_trace_std=std
                )
                x2, m2 = crop_candidate_window(
                    zne, float(row.c2_sample), cfg=WIN, full_trace_mean=mean, full_trace_std=std
                )
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


def train_loop(model, loader, opt, device, max_batches=None, pos_weight: float = 1.0):
    model.train()
    total, n = 0.0, 0
    grad_wave = 0.0
    bce = nn.BCELoss(reduction="none")
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
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
def predict_probs(model, df: pd.DataFrame, reader, device, batch_size=64, crop_cache=None) -> pd.DataFrame:
    load_wave = bool(getattr(model, "use_waveform", True))
    ds = PairDataset(
        df,
        reader,
        decisive_only=False,
        augment_swap=False,
        load_waveform=load_wave,
        crop_cache=crop_cache,
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


def apply_policy(pairs: pd.DataFrame, probs: pd.DataFrame, tau: float) -> np.ndarray:
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
        "precision@0.5": float(m["precision@0.5s"]),
        "recall@0.5": float(m["recall@0.5s"]),
        "miss_rate": float(m["miss_rate"]),
        "detected_ae_median": float(m["detected_ae_median"]),
        "detected_ae_p95": float(m["detected_ae_p95"]),
        "detected_ae_mae": float(m.get("detected_ae_mae", m.get("mae", np.nan))),
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
        # fixes/breaks on natural cal
        base_ok = cal_pairs.c1_ok.to_numpy(bool)
        c2_ok = cal_pairs.c2_ok.to_numpy(bool)
        switched = switch
        # after switch correctness approx: if switched use c2_ok else c1_ok
        new_ok = np.where(switched, c2_ok, base_ok)
        fixes = int(np.sum(switched & (~base_ok) & c2_ok))
        breaks = int(np.sum(switched & base_ok & (~c2_ok)))
        prec = fixes / max(fixes + breaks, 1)
        net = fixes - breaks
        rows.append({"tau": tau, "fixes": fixes, "breaks": breaks, "net": net, "precision": prec, "n_switch": int(switched.sum())})
        if prec >= 0.80:
            cand = (net, -tau, tau, prec, fixes, breaks)  # higher net, then higher tau
            if best is None or cand > best[0]:
                best = (cand, tau, rows[-1])
    if best is None:
        return 1.0, {"tau": 1.0, "note": "no tau met precision>=0.80; never switch", "grid": rows}
    return float(best[1]), {"selected": best[2], "grid": rows}


def main() -> None:
    t0 = time.time()
    out = ensure_dir(OUT)
    if not (out / "SPLIT.LOCK.json").exists():
        raise SystemExit("Run freeze_pairwise_split_and_pairs.py first")
    if (out / "PAIRWISE.EVAL.DONE").exists():
        raise SystemExit("held-out evaluation already executed once")

    # GPU check — only truly free GPUs
    try:
        gpu_i = _pick_free_gpu(min_free_gb=2.5, max_util=5)
    except Exception as e:
        _block(str(e))
    torch.cuda.set_device(gpu_i)
    device = torch.device(f"cuda:{gpu_i}")
    print(f"using GPU {gpu_i}, free={torch.cuda.mem_get_info(gpu_i)[0]/1e9:.1f}GB", flush=True)

    pairs = pd.read_parquet(out / "pairs_ranker_train.parquet")
    confirm_events = set(
        Path(artifacts_dir() / "results/stage6/splits_full/stage6_internal_confirm_events.txt").read_text().split()
    )
    if set(pairs.event_id.astype(str)) & confirm_events:
        _block("confirm events found in pairwise pairs table")
    train_df = pairs[pairs.split == "train"].copy()
    cal_df = pairs[pairs.split == "calibration"].copy()
    eval_df = pairs[pairs.split == "heldout_eval"].copy()
    h5 = resolve_instance_root() / "events" / "Instance_events_counts.hdf5"
    reader = InstanceHDF5Reader(h5).open()
    n_scalar = len(SCALAR_KEYS_C)

    # optional crop cache
    crop_cache = None
    cache_dir = out / "crop_cache"
    if (cache_dir / "waves.npy").exists() and (cache_dir / "trace_index.json").exists():
        crop_cache = {
            "waves": np.load(cache_dir / "waves.npy", mmap_mode="r"),
            "masks": np.load(cache_dir / "masks.npy", mmap_mode="r"),
            "index": json.loads((cache_dir / "trace_index.json").read_text()),
        }
        print(f"loaded crop cache n={len(crop_cache['index'])}", flush=True)
    else:
        print("WARNING: no crop cache; waveform training will be slow (HDF5 random reads)", flush=True)

    def make_model(use_waveform=True, use_scalar=True):
        m = PairwiseScorer(n_scalar, use_waveform=use_waveform, use_scalar=use_scalar)
        if m.n_parameters() > 300_000:
            _block(f"too many params: {m.n_parameters()}")
        return m.to(device)

    # class weight from natural train decisive counts (calibration/eval remain unweighted)
    dec_all = train_df[train_df.y_choose_c2.isin([0, 1])]
    n_pos = int((dec_all.y_choose_c2 == 1).sum())
    n_neg = int((dec_all.y_choose_c2 == 0).sum())
    pos_weight = float(n_neg / max(n_pos, 1))
    print(f"class balance train decisive: n_pos={n_pos} n_neg={n_neg} pos_weight={pos_weight:.3f}", flush=True)

    # ---------- overfit 64 ----------
    print("overfit 64...", flush=True)
    dec = train_df[train_df.y_choose_c2.isin([0, 1])]
    # balance a tiny set
    pos = dec[dec.y_choose_c2 == 1].head(32)
    neg = dec[dec.y_choose_c2 == 0].head(32)
    tiny = pd.concat([pos, neg]).sample(frac=1.0, random_state=SEED)
    if len(tiny) < 32:
        _block(f"not enough decisive pairs for overfit: {len(tiny)}")
    model = make_model(True, True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ds = PairDataset(tiny, reader, decisive_only=True, augment_swap=True)
    loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=0, collate_fn=collate)
    losses = []
    for ep in range(30):
        st = train_loop(model, loader, opt, device, pos_weight=1.0)
        losses.append(st["loss"])
    # accuracy on tiny
    probs = predict_probs(model, tiny, reader, device)
    merged = tiny.merge(probs, on="trace_name")
    acc = float((((merged.p_c2_gt_c1 > 0.5).astype(int)) == merged.y_choose_c2.astype(int)).mean())
    # swap test
    batch = collate([ds[i] for i in range(min(8, len(ds)))])
    for k in batch:
        if torch.is_tensor(batch[k]):
            batch[k] = batch[k].to(device)
    swap_err = assert_swap_antisymmetry(model, batch)
    overfit_ok = (losses[0] - losses[-1] > 0.05) and (acc > 0.65) and (swap_err < 1e-5)
    print("overfit", {"loss0": losses[0], "lossN": losses[-1], "acc": acc, "swap_err": swap_err, "ok": overfit_ok}, flush=True)
    if not overfit_ok:
        _block(f"overfit failed loss0={losses[0]:.3f} lossN={losses[-1]:.3f} acc={acc:.3f}")

    # ---------- smoke 512 ----------
    print("smoke 512...", flush=True)
    pos = dec[dec.y_choose_c2 == 1].head(256)
    neg = dec[dec.y_choose_c2 == 0].head(256)
    smoke = pd.concat([pos, neg]).sample(frac=1.0, random_state=SEED)
    model = make_model(True, True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ds = PairDataset(smoke, reader, decisive_only=True, augment_swap=True)
    loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0, collate_fn=collate)
    losses = []
    for ep in range(15):
        st = train_loop(model, loader, opt, device, pos_weight=1.0)
        losses.append(st["loss"])
        if st.get("grad_wave", 1) == 0:
            _block("waveform gradient is zero")
    probs = predict_probs(model, smoke, reader, device)
    merged = smoke.merge(probs, on="trace_name")
    acc = float((((merged.p_c2_gt_c1 > 0.5).astype(int)) == merged.y_choose_c2.astype(int)).mean())
    frac_pred_c2 = float((merged.p_c2_gt_c1 > 0.5).mean())
    if acc < 0.60 or frac_pred_c2 < 0.1 or frac_pred_c2 > 0.9:
        _block(f"smoke collapse/acc fail acc={acc} frac_c2={frac_pred_c2}")
    print("smoke ok", acc, frac_pred_c2, flush=True)

    # ---------- full train pilot (waveform+scalar) ----------
    print("full train pilot...", flush=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    configs = {
        "scalar_pairwise": dict(use_waveform=False, use_scalar=True),
        "waveform_pairwise": dict(use_waveform=True, use_scalar=False),
        "waveform_plus_scalar_pairwise": dict(use_waveform=True, use_scalar=True),
    }
    cal_metrics = {}
    models = {}
    for name, kw in configs.items():
        print("training", name, flush=True)
        model = make_model(**kw)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        ds = PairDataset(
            train_df,
            reader,
            decisive_only=True,
            augment_swap=True,
            load_waveform=bool(kw.get("use_waveform", True)),
            crop_cache=crop_cache,
        )
        loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0, collate_fn=collate)
        best_state, best_net, patience = None, -10**9, 0
        cal_ge2 = cal_df[cal_df.n_candidates >= 2]
        for ep in range(20):
            st = train_loop(model, loader, opt, device, pos_weight=pos_weight)
            # natural calibration net @ tau=0.5 (checkpoint metric)
            pr_all = predict_probs(model, cal_ge2, reader, device, batch_size=128, crop_cache=crop_cache)
            pred, switch = apply_policy(cal_df, pr_all, 0.5)
            base_ok = cal_df.c1_ok.to_numpy(bool)
            c2_ok = cal_df.c2_ok.to_numpy(bool)
            fixes_n = int(np.sum(switch & (~base_ok) & c2_ok))
            breaks_n = int(np.sum(switch & base_ok & (~c2_ok)))
            net = fixes_n - breaks_n
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
        models[name] = model
        ckpt = out / f"ckpt_{name}_seed42.pt"
        torch.save({"state_dict": best_state, "n_scalar": n_scalar, **kw}, ckpt)
        cal_metrics[name] = {"best_cal_net_at_tau0.5": best_net, "ckpt": str(ckpt), "ckpt_sha256": _sha_file(ckpt)}

    # primary model
    primary = "waveform_plus_scalar_pairwise"
    model = models[primary]
    cal_probs = predict_probs(model, cal_df[cal_df.n_candidates >= 2], reader, device)
    # for traces <2 cand, no prob needed
    tau, tau_info = choose_tau(cal_df, cal_probs)
    print("selected tau", tau, tau_info.get("selected"), flush=True)

    lock = {
        "method": primary,
        "seed": SEED,
        "tau": tau,
        "tau_selection": tau_info,
        "window": {"pre_s": WIN.pre_s, "post_s": WIN.post_s, "n_samples": WIN.n_samples},
        "scalar_features": SCALAR_KEYS_C,
        "n_parameters": models[primary].n_parameters(),
        "ckpt_sha256": cal_metrics[primary]["ckpt_sha256"],
        "split_lock": str(out / "SPLIT.LOCK.json"),
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "max_epoch": 20,
        "patience": 4,
        "code_note": "pairwise pilot; not Stage-6 method lock",
    }
    save_json(lock, out / "PAIRWISE.METHOD.LOCK.json")

    # ---------- held-out once ----------
    print("held-out evaluation (ONCE)...", flush=True)
    (out / "PAIRWISE.EVAL.DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ") + "\n")

    true = eval_df.true_s_sample.to_numpy(float)
    sr = eval_df.sampling_rate_hz.to_numpy(float)
    fixed_pred = eval_df.c1_sample.to_numpy(float)
    resid_pred = resid_control_pred(eval_df)

    ablation = {}
    ablation["fixed_rescore_UNION"] = {
        **metrics_from_pred(fixed_pred, true, sr),
        "fixes": 0,
        "breaks": 0,
        "net_fixes": 0,
        "switch_precision": float("nan"),
        "switch_coverage": 0.0,
    }
    rmet = metrics_from_pred(resid_pred, true, sr)
    # resid fixes/breaks vs fixed
    base_ok = eval_df.c1_ok.to_numpy(bool)
    new_ok = np.abs(resid_pred - true) / sr <= 0.5
    switched_r = np.abs(resid_pred - fixed_pred) > 0.5
    fixes_r = int(np.sum(switched_r & (~base_ok) & new_ok))
    breaks_r = int(np.sum(switched_r & base_ok & (~new_ok)))
    ablation["resid_s_control"] = {
        **rmet,
        "delta_f1@0.5": rmet["f1@0.5"] - ablation["fixed_rescore_UNION"]["f1@0.5"],
        "fixes": fixes_r,
        "breaks": breaks_r,
        "net_fixes": fixes_r - breaks_r,
        "switch_precision": fixes_r / max(fixes_r + breaks_r, 1),
        "switch_coverage": float(switched_r.mean()),
    }

    for name, model in models.items():
        pr = predict_probs(model, eval_df[eval_df.n_candidates >= 2], reader, device)
        # use primary tau for final; for ablations also report at same tau
        pred, switch = apply_policy(eval_df, pr, tau if name == primary else tau)
        met = metrics_from_pred(pred, true, sr)
        new_ok = np.abs(pred - true) / sr <= 0.5
        fixes = int(np.sum(switch & (~base_ok) & new_ok))
        breaks = int(np.sum(switch & base_ok & (~new_ok)))
        # Q2 recovery
        q2 = (~eval_df.c1_ok) & (eval_df.c2_ok)
        q2_rec = float(np.sum(switch & q2.to_numpy(bool) & new_ok) / max(int(q2.sum()), 1))
        ablation[name] = {
            **met,
            "delta_f1@0.5": met["f1@0.5"] - ablation["fixed_rescore_UNION"]["f1@0.5"],
            "delta_f1@0.1": met["f1@0.1"] - ablation["fixed_rescore_UNION"]["f1@0.1"],
            "delta_p95": met["detected_ae_p95"] - ablation["fixed_rescore_UNION"]["detected_ae_p95"],
            "fixes": fixes,
            "breaks": breaks,
            "net_fixes": fixes - breaks,
            "switch_precision": fixes / max(fixes + breaks, 1),
            "switch_coverage": float(switch.mean()),
            "q2_n": int(q2.sum()),
            "q2_recovery": q2_rec,
            "tau": tau,
        }
        # consistency check
        n_eval = len(eval_df)
        expected_df1 = (fixes - breaks) / n_eval
        if abs(ablation[name]["delta_f1@0.5"] - expected_df1) > 0.002:
            print("WARN deltaF1 vs net_fixes/n mismatch", ablation[name]["delta_f1@0.5"], expected_df1)

    # bootstrap on primary
    prim = ablation[primary]
    prim_pred, prim_switch = apply_policy(
        eval_df,
        predict_probs(models[primary], eval_df[eval_df.n_candidates >= 2], reader, device),
        tau,
    )
    rng = np.random.default_rng(SEED)
    events = eval_df.event_id.astype(str).to_numpy()
    idx_by = {}
    for i, e in enumerate(events):
        idx_by.setdefault(e, []).append(i)
    lists = [np.asarray(v, int) for v in idx_by.values()]
    deltas = np.empty(2000)
    for b in range(2000):
        draw = rng.integers(0, len(lists), size=len(lists))
        ix = np.concatenate([lists[j] for j in draw])
        ma = metrics_from_pred(prim_pred[ix], true[ix], sr[ix])
        mb = metrics_from_pred(fixed_pred[ix], true[ix], sr[ix])
        deltas[b] = ma["f1@0.5"] - mb["f1@0.5"]
    boot = {
        "n_boot": 2000,
        "seed": SEED,
        "delta_f1@0.5": {
            "mean": float(deltas.mean()),
            "ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
        },
    }
    save_json(boot, out / "bootstrap.json")
    save_json(ablation, out / "ablation_metrics.json")
    save_json(cal_metrics, out / "calibration_train_summary.json")

    # GO/NO-GO gates
    n_eval = len(eval_df)
    gates = {
        "1_delta_f1_vs_fixed_ge_0.010": prim["delta_f1@0.5"] >= 0.010,
        "2_delta_f1_vs_resid_ge_0.004": (prim["f1@0.5"] - ablation["resid_s_control"]["f1@0.5"]) >= 0.004,
        "3_bootstrap_ci_low_gt_0": boot["delta_f1@0.5"]["ci95"][0] > 0,
        "4_net_fixes_ge_1pct": prim["net_fixes"] >= int(np.ceil(0.01 * n_eval)),
        "5_switch_precision_ge_0.80": prim["switch_precision"] >= 0.80,
        "6_q2_recovery_ge_0.50": prim.get("q2_recovery", 0) >= 0.50,
        "7_f1_0.1_not_worse_than_0.001": prim["delta_f1@0.1"] >= -0.001,
        "8_p95_not_worse_0.05": prim["delta_p95"] <= 0.05,
        "9_miss_rate_zero": prim["miss_rate"] == 0.0 and ablation["fixed_rescore_UNION"]["miss_rate"] == 0.0,
        "10_outputs_from_c1_c2_only": True,  # by construction
        "11_waveform_not_identical_to_scalar": abs(
            ablation["waveform_pairwise"]["delta_f1@0.5"] - ablation["scalar_pairwise"]["delta_f1@0.5"]
        )
        > 1e-4
        or ablation["waveform_pairwise"]["f1@0.5"] != ablation["scalar_pairwise"]["f1@0.5"],
    }
    passed = all(gates.values())
    report = {
        "gates": gates,
        "passed": passed,
        "primary": prim,
        "ablation": ablation,
        "bootstrap": boot,
        "tau": tau,
        "n_eval": n_eval,
        "n_parameters": models[primary].n_parameters(),
        "input_shape": [3, WIN.n_samples],
        "full_dev_read": False,
        "confirm_read": False,
        "elapsed_s": time.time() - t0,
    }
    # markdown report
    lines = [
        "# Pairwise pilot report",
        "",
        f"Primary: `{primary}`  tau={tau}",
        f"Params: {models[primary].n_parameters()}  input: (3, {WIN.n_samples})",
        "",
        "## Ablation F1@0.5",
        json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ('f1@0.5','delta_f1@0.5','fixes','breaks','net_fixes','switch_precision')} for k,v in ablation.items()}, indent=2),
        "",
        "## Gates",
        json.dumps(gates, indent=2),
        "",
        f"Verdict: {'PASSED' if passed else 'FAILED'}",
        "",
        "Hard stop. No full-dev. No confirm.",
    ]
    (ROOT / "reports" / "pairwise" / "pairwise_pilot_report.md").write_text("\n".join(lines) + "\n")
    save_json(report, out / "pilot_final_report.json")

    if passed:
        _pass(report)
    else:
        _fail("waveform_pairwise_no_go", report)


if __name__ == "__main__":
    main()
