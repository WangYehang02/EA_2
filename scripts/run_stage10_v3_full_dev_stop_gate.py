#!/usr/bin/env python
"""Stage-6 full-dev stop gate after v3 train. Never reads confirm. Never starts confirm/other seeds."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, resolve_instance_root, save_json
from earthquake.stage6.full_splits import load_full_event_ids, load_full_trace_names
from earthquake.stage6.lazy_dataset import WorkerHDF5WaveformSource
from earthquake.stage6.phaseC1.metrics import comprehensive_pick_metrics, correct_at
from earthquake.stage10.candidates import event_bootstrap_delta
from earthquake.stage10.dataset_v2 import DKPNPartialCropDataset, annotate_online_catalog
from earthquake.stage10.dkpn_clean import build_dkpn_random, dkpn_logits
from earthquake.stage10.dkpn_picks import extract_picks
from earthquake.utils import ensure_dir

OUT_DEFAULT = Path("/data/mnt_data/yehang/PSdetec/Earthquake_stage10/dkpn_clean_v3_seed42_mixedcrop")
HEIGHT = 0.2
N_BOOT = 5000
NOISE_N = 4000


def collate(batch):
    keys_t = ["x", "p_pos", "s_pos", "n_pos", "not_p", "not_s", "pad_mask", "vis_p", "vis_s", "p_c", "s_c"]
    out = {k: torch.stack([b[k] for b in batch]) for k in keys_t}
    out["crop_kind"] = [b["crop_kind"] for b in batch]
    out["trace_name"] = [b["trace_name"] for b in batch]
    out["event_id"] = [b["event_id"] for b in batch]
    out["is_noise"] = torch.tensor([b["is_noise"] for b in batch])
    return out


def _load_model(path: Path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build_dkpn_random()
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model, ck if isinstance(ck, dict) else {}


@torch.no_grad()
def infer_s_centered_abs(model, cat: pd.DataFrame, wave, device, batch: int = 32) -> pd.DataFrame:
    cat = cat.copy()
    cat["crop_kind"] = "s_centered"
    cat["is_noise"] = False
    ds = DKPNPartialCropDataset(cat, lambda n, is_noise=False: wave.read(n, is_noise=is_noise), augment=False, seed=42)
    ld = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0, collate_fn=collate)
    rows = []
    n_by = cat.set_index("trace_name")
    for batch_ in ld:
        pr = torch.softmax(dkpn_logits(model, batch_["x"].to(device)).float(), dim=1).cpu().numpy()
        for i in range(pr.shape[0]):
            tn = str(batch_["trace_name"][i])
            peaks, _, ampl, _ = extract_picks(pr[i, 1], thr=HEIGHT, min_distance=50)
            win_peak = int(peaks[int(np.argmax(ampl))]) if len(peaks) else None
            row = n_by.loc[tn]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            s_abs = float(row["s_arrival_sample"])
            s_c = float(batch_["s_c"][i])
            pred_abs = float("nan")
            if win_peak is not None and s_c >= 0:
                pred_abs = float(s_abs - s_c + win_peak)
            rows.append(
                {
                    "trace_name": tn,
                    "event_id": str(batch_["event_id"][i]),
                    "pred_s_sample": pred_abs,
                    "true_s_sample": s_abs,
                    "s_c": s_c,
                    "win_peak": win_peak,
                    "s_max": float(pr[i, 1].max()),
                }
            )
    return pd.DataFrame(rows)


def metrics_from_aligned(pred, true, sr, event_ids) -> dict:
    m = comprehensive_pick_metrics(pred, true, sr)
    ok05 = correct_at(pred, true, sr, 0.5)
    ok01 = correct_at(pred, true, sr, 0.1)
    return {
        **{k: m[k] for k in m if not str(k).startswith("p95_")},
        "correct@0.5": ok05.astype(float),
        "correct@0.1": ok01.astype(float),
        "event_id": event_ids,
    }


def k_oracles() -> dict:
    p = artifacts_dir() / "results" / "stage6" / "phaseB_candidate_oracle.json"
    if not p.is_file():
        return {}
    doc = json.loads(p.read_text())
    out = {}
    for key in ("STEAD_K5", "STEAD_K10", "UNION_K5", "UNION_K10", "K5", "K10"):
        if key in doc:
            out[key] = doc[key]
    # nested oracles
    ora = doc.get("oracles") or doc
    for k in ("UNION_K5", "UNION_K10", "STEAD_K5", "STEAD_K10"):
        if k in ora and k not in out:
            out[k] = ora[k]
    return out


def noise_fpr(model, wave, device, seed=42) -> dict:
    noise = pd.read_parquet(artifacts_dir() / "index_full" / "noise.parquet")
    confirm = set(load_full_event_ids("stage6_internal_confirm"))
    if "event_id" in noise.columns:
        noise = noise[~noise["event_id"].astype(str).isin(confirm)]
    nsel = noise.sample(n=min(NOISE_N, len(noise)), random_state=seed).reset_index(drop=True)
    nsel["is_noise"] = True
    nsel["crop_kind"] = "noise"
    ds = DKPNPartialCropDataset(nsel, lambda n, is_noise=True: wave.read(n, is_noise=True), augment=False, seed=seed)
    ld = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate)
    n_pick = 0
    n_tot = 0
    with torch.no_grad():
        for batch in ld:
            pr = torch.softmax(dkpn_logits(model, batch["x"].to(device)).float(), dim=1).cpu().numpy()
            for i in range(pr.shape[0]):
                n_tot += 1
                peaks, _, _, _ = extract_picks(pr[i, 1], thr=HEIGHT, min_distance=50)
                n_pick += int(len(peaks) > 0)
    return {"n": n_tot, "n_with_s_peak": n_pick, "fpr": n_pick / max(n_tot, 1), "confirm_read": False}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    args = ap.parse_args()
    out = Path(args.out_dir)
    gate_dir = ensure_dir(out / "full_dev_stop_gate")
    if any("confirm" in p.lower() and "internal_confirm" in p for p in sys.argv):
        raise SystemExit("confirm path in argv")
    confirm_ids = set(load_full_event_ids("stage6_internal_confirm"))
    meta = pd.read_csv(artifacts_dir() / "results" / "stage6" / "phaseB_eval_manifest.csv")
    if set(meta["event_id"].astype(str)) & confirm_ids:
        raise SystemExit("LEAK stop-gate manifest ∩ confirm")
    pred_dir = artifacts_dir() / "results" / "stage6" / "phaseC" / "baseline_preds"
    names = meta["trace_name"].astype(str).to_numpy()
    true = meta["s_arrival_sample"].to_numpy(float)
    sr = meta["sampling_rate_hz"].to_numpy(float)
    events = meta["event_id"].astype(str).to_numpy()
    union = np.load(pred_dir / "fixed_rescore_UNION.npy")
    stead = np.load(pred_dir / "STEAD_top1.npy")
    if len(union) != len(meta) or len(stead) != len(meta):
        raise SystemExit(f"pred length mismatch union={len(union)} stead={len(stead)} meta={len(meta)}")
    union_m = comprehensive_pick_metrics(union, true, sr)
    stead_m = comprehensive_pick_metrics(stead, true, sr)
    kora = k_oracles()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inst = resolve_instance_root()
    wave = WorkerHDF5WaveformSource(inst / "events" / "Instance_events_counts.hdf5", inst / "noise" / "Instance_noise.hdf5")
    events_idx = pd.read_parquet(artifacts_dir() / "index_full" / "events.parquet")
    dev_tr = set(load_full_trace_names("stage6_dev"))
    ev = events_idx[events_idx["trace_name"].astype(str).isin(dev_tr)].copy()
    if set(ev["event_id"].astype(str)) & confirm_ids:
        raise SystemExit("LEAK full-dev events ∩ confirm")
    s = pd.to_numeric(ev["s_arrival_sample"], errors="coerce")
    ev = ev[s.notna()].copy()
    # align to manifest order
    ev = ev.drop_duplicates("trace_name").set_index("trace_name").reindex(names).reset_index()
    ev["is_noise"] = False
    cat = annotate_online_catalog(ev)
    ckpts = {
        "best_metric": out / "checkpoints" / "best_metric.pt",
        "best_loss": out / "checkpoints" / "best_loss.pt",
        "last": out / "checkpoints" / "last.pt",
    }
    nvis = torch.cuda.device_count() if device.type == "cuda" else 1
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "confirm_read": False,
        "auto_confirm": False,
        "auto_other_seeds": False,
        "official_height": HEIGHT,
        "n_boot": N_BOOT,
        "UNION": {k: union_m[k] for k in union_m},
        "STEAD_top1_waveform_only": {k: stead_m[k] for k in stead_m},
        "K_oracles_phaseB": kora,
        "dkpn": {},
        "n_full_dev": int(len(meta)),
    }
    union_ok = correct_at(union, true, sr, 0.5)
    for name, path in ckpts.items():
        if not path.is_file():
            report["dkpn"][name] = {"missing": True}
            continue
        model, meta_ck = _load_model(path, device)
        if nvis >= 2:
            model = torch.nn.DataParallel(model)
        df = infer_s_centered_abs(model, cat, wave, device, batch=32)
        df = df.set_index("trace_name").reindex(names).reset_index()
        pred = df["pred_s_sample"].to_numpy(float)
        dm = comprehensive_pick_metrics(pred, true, sr)
        dkpn_ok = correct_at(pred, true, sr, 0.5)
        closer = np.where(
            np.isfinite(pred) & np.isfinite(union),
            np.where(np.abs(pred - true) <= np.abs(union - true), pred, union),
            np.where(np.isfinite(pred), pred, union),
        )
        oracle_ud = np.where(dkpn_ok | union_ok, true, np.nan)  # hit-or-miss oracle for coverage
        # prefer closer candidate as the UNION+DKPN pick
        combo = closer
        combo_m = comprehensive_pick_metrics(combo, true, sr)
        only = dkpn_ok & ~union_ok
        f1_hit_d = dkpn_ok.astype(float)
        f1_hit_u = union_ok.astype(float)
        boot = event_bootstrap_delta(events, f1_hit_u, f1_hit_d, n_boot=N_BOOT, seed=42)
        nf = noise_fpr(model, wave, device)
        report["dkpn"][name] = {
            "path": str(path),
            "epoch": meta_ck.get("epoch") or meta_ck.get("virtual_epoch"),
            "valid_best": meta_ck.get("valid_best"),
            "metrics": {k: dm[k] for k in dm},
            "UNION_plus_DKPN": {k: combo_m[k] for k in combo_m},
            "dkpn_only_correct@0.5": int(only.sum()),
            "dkpn_only_correct_rate@0.5": float(only.mean()),
            "event_bootstrap_5000_dkpn_minus_UNION_hit@0.5": boot,
            "noise_fpr_height0p2": nf,
            "oracle_union_or_dkpn_hit_rate@0.5": float((dkpn_ok | union_ok).mean()),
        }
        print(json.dumps({"ckpt": name, "f1@0.5": dm.get("f1@0.5"), "dkpn_only": int(only.sum())}), flush=True)
    # stop-gate pass/fail vs strongest waveform-only (STEAD) and vs UNION
    best_name = "best_metric"
    dkm = report["dkpn"].get(best_name, {}).get("metrics") or {}
    f1_d = float(dkm.get("f1@0.5") or 0)
    f1_s = float(stead_m.get("f1@0.5") or 0)
    f1_u = float(union_m.get("f1@0.5") or 0)
    p95_d = float(dkm.get("detected_ae_p95") or 1e9)
    p95_s = float(stead_m.get("detected_ae_p95") or 0)
    boot_bm = (report["dkpn"].get(best_name) or {}).get("event_bootstrap_5000_dkpn_minus_UNION_hit@0.5") or {}
    gates = {
        "A_vs_waveform_only_delta_f1_0.5_ge_0.01": bool(f1_d - f1_s >= 0.01),
        "B_f1_not_down_and_p95_ge_0.15s": bool(f1_d >= f1_s - 1e-12 and (p95_s - p95_d) >= 0.15),
        "C_union_oracle_or_hit_bootstrap": bool(float(boot_bm.get("ci95_lo") or -1) > 0),
    }
    passed = bool(gates["A_vs_waveform_only_delta_f1_0.5_ge_0.01"] or gates["B_f1_not_down_and_p95_ge_0.15s"] or gates["C_union_oracle_or_hit_bootstrap"])
    report["gates"] = gates
    report["passed"] = passed
    report["discuss_confirm_or_other_seeds"] = bool(passed)
    report["auto_executed_confirm"] = False
    save_json(report, gate_dir / "stop_gate_report.json")
    save_json(report, artifacts_dir() / "results" / "stage10" / "dkpn_v3_full_dev_stop_gate.json")
    md = [
        "# DKPN v3 full-dev stop gate",
        "",
        f"passed: **{passed}**",
        "confirm_read: false",
        "auto_confirm: false",
        "",
        f"- STEAD_top1 F1@0.5 = {f1_s:.4f}",
        f"- UNION F1@0.5 = {f1_u:.4f}",
        f"- DKPN best_metric F1@0.5 = {f1_d:.4f}",
        "",
        "Do not start confirm or other seeds unless this gate passed and the user explicitly approves.",
    ]
    (gate_dir / "stop_gate_report.md").write_text("\n".join(md) + "\n")
    (out / ("STOP_GATE.PASSED" if passed else "STOP_GATE.FAILED")).write_text(datetime.now(timezone.utc).isoformat())
    print(json.dumps({"passed": passed, "gates": gates, "confirm_read": False}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
