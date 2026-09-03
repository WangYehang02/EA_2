#!/usr/bin/env python
"""v4 frozen-checkpoint Stage-6 full-dev stop gate.

Requires METHOD.LOCK.json written first. Does not train, does not create v5,
does not reread confirm waveforms/metrics, does not overwrite PILOT failures.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthquake.config import artifacts_dir, load_json, resolve_instance_root, save_json
from earthquake.stage10.dkpn_clean import build_dkpn_random
from earthquake.stage10.fp32_guard import assert_no_autocast, assert_params_fp32, state_dict_tensor_sha256
from earthquake.stage10.full_dev_annotate import (
    cf_from_enz,
    picks_from_s_proba,
    s_windows_fp32,
    windows_from_cf,
    stack_avg,
)
from earthquake.stage10.full_dev_stop_gate import (
    BOOT_SEED,
    HEIGHT,
    MARK,
    N_BOOT,
    NOISE_N,
    STRONGEST_WAVEFORM_ONLY,
    V4,
    dkpn_k5_frame,
    event_bootstrap_f1,
    grouping_table,
    jsonable,
    merge_union_dkpn5,
    metrics_pack,
    oracle_pred_from_candidates,
    per_event_table,
    sha256_file,
    stop_gates,
)
from earthquake.stage10.gpu_policy import idle_gpu_indices
from earthquake.stage6.keyed_align import keyed_align_predictions
from earthquake.stage6.lazy_dataset import WorkerHDF5WaveformSource
from earthquake.stage6.phaseB import oracle_metrics_for_set
from earthquake.stage6.phaseC1.metrics import correct_at
from earthquake.utils import ensure_dir

CONFIRM_PRED = ROOT / "artifacts/results/stage6/final_confirm/confirm_predictions.parquet"
CONFIRM_METRICS = ROOT / "artifacts/results/stage6/final_confirm/confirm_method_metrics.json"
PILOT_FAILED = V4 / "PILOT.FAILED"
PILOT_ADJ = V4 / "PILOT.ADJUDICATION.FAILED"
AMENDMENT = ROOT / "configs/stage10/protocol_amendment_20260902_noninferiority.json"
FAILED_SHA = "b544aba50775be366bd0d49f23c5e3c3f6f00120f1d54f0c4871f9f0f7994e67"
LOCK_PATH = V4 / "full_dev_stop_gate" / "METHOD.LOCK.json"


def _atime(p: Path) -> float | None:
    return p.stat().st_atime if p.is_file() else None


class FullTraceDS(Dataset):
    def __init__(self, meta: pd.DataFrame, wave: WorkerHDF5WaveformSource, *, is_noise: bool = False):
        self.meta = meta.reset_index(drop=True)
        self.wave = wave
        self.is_noise = bool(is_noise)

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, i: int) -> dict:
        row = self.meta.iloc[i]
        enz = np.asarray(self.wave.read(str(row["trace_name"]), is_noise=self.is_noise), dtype=np.float32)
        cf = cf_from_enz(enz)
        xs, offsets, n = windows_from_cf(cf)
        if not np.isfinite(xs).all():
            raise RuntimeError(f"non-finite CF windows {row['trace_name']}")
        return {
            "xs": torch.from_numpy(xs),
            "offsets": torch.from_numpy(offsets.astype(np.int64)),
            "n_samples": int(n),
            "trace_name": str(row["trace_name"]),
            "event_id": str(row.get("event_id", "")),
            "index": int(i),
        }


def collate(batch):
    return batch


def load_model(path: Path, device) -> torch.nn.Module:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build_dkpn_random()
    model.load_state_dict(ck["model"])
    model.to(device)
    model.eval()
    assert_params_fp32(model)
    return model


def infer_roles(models: dict[str, torch.nn.Module], loader, device, n: int) -> dict[str, dict]:
    """One CF pass; all unique-weight models. Returns dict role -> arrays."""
    roles = list(models)
    acc = {r: {"pred": [None] * n, "n_peaks": [None] * n, "s_max": [None] * n, "peaks_k5": [None] * n, "peaks_k10": [None] * n, "name": [None] * n, "eid": [None] * n} for r in roles}
    seen = 0
    assert_no_autocast()
    for batch in loader:
        for item in batch:
            xs = item["xs"].numpy()
            offsets = item["offsets"].numpy()
            n_s = int(item["n_samples"])
            i = int(item["index"])
            for r, model in models.items():
                win = s_windows_fp32(model, xs, device)
                s_pr = stack_avg(win, offsets, n_s)
                if s_pr.size and not np.isfinite(s_pr).all():
                    raise RuntimeError("non-finite stitched S")
                pk = picks_from_s_proba(s_pr)
                acc[r]["pred"][i] = pk["pred_s_sample"]
                acc[r]["n_peaks"][i] = pk["n_peaks"]
                acc[r]["s_max"][i] = pk["s_max"]
                acc[r]["peaks_k5"][i] = pk["peaks_k5"]
                acc[r]["peaks_k10"][i] = pk["peaks_k10"]
                acc[r]["name"][i] = item["trace_name"]
                acc[r]["eid"][i] = item["event_id"]
            seen += 1
            if seen % 200 == 0:
                print(json.dumps({"inferred": seen, "n": n}), flush=True)
    out = {}
    for r in roles:
        if any(x is None for x in acc[r]["pred"]):
            raise RuntimeError(f"incomplete preds {r}")
        out[r] = {
            "pred": np.asarray(acc[r]["pred"], dtype=float),
            "n_peaks": np.asarray(acc[r]["n_peaks"], dtype=int),
            "s_max": np.asarray(acc[r]["s_max"], dtype=float),
            "peaks_k5": acc[r]["peaks_k5"],
            "peaks_k10": acc[r]["peaks_k10"],
            "trace_name": np.asarray(acc[r]["name"]),
            "event_id": np.asarray(acc[r]["eid"]),
        }
    return out


def noise_fpr(model, wave, device, seed=42) -> dict:
    noise = pd.read_parquet(artifacts_dir() / "index_full" / "noise.parquet")
    nsel = noise.sample(n=min(NOISE_N, len(noise)), random_state=seed).reset_index(drop=True)
    nsel["trace_name"] = nsel["trace_name"].astype(str)
    ds = FullTraceDS(nsel, wave, is_noise=True)
    ld = DataLoader(ds, batch_size=8, shuffle=False, num_workers=4, collate_fn=collate, persistent_workers=False)
    n_pick = 0
    n_tot = 0
    for batch in ld:
        for item in batch:
            xs = item["xs"].numpy()
            offsets = item["offsets"].numpy()
            n_s = int(item["n_samples"])
            win = s_windows_fp32(model, xs, device)
            s_pr = stack_avg(win, offsets, n_s)
            pk = picks_from_s_proba(s_pr)
            n_tot += 1
            n_pick += int(pk["n_peaks"] > 0)
    return {"n": n_tot, "n_with_s_peak": n_pick, "fpr": n_pick / max(n_tot, 1), "confirm_read": False, "height": HEIGHT}


def main() -> int:
    if any("confirm" in a.lower() and "internal_confirm" in a for a in sys.argv):
        raise SystemExit("confirm path in argv")
    if not LOCK_PATH.is_file():
        raise SystemExit("METHOD.LOCK.json missing; run lock_stage10_v4_fulldev_stop_gate.py first")
    lock = load_json(LOCK_PATH)
    if lock.get("marker") != MARK:
        raise SystemExit("bad lock marker")
    if sha256_file(PILOT_FAILED) != FAILED_SHA:
        raise SystemExit("PILOT.FAILED hash changed")
    if os.access(PILOT_FAILED, os.W_OK):
        raise SystemExit("PILOT.FAILED became writable")
    if not PILOT_ADJ.is_file() or not AMENDMENT.is_file():
        raise SystemExit("pilot failure markers missing")

    confirm_atime_before = {"predictions": _atime(CONFIRM_PRED), "metrics": _atime(CONFIRM_METRICS)}

    art = artifacts_dir()
    man_csv = art / "results" / "stage6" / "phaseB_eval_manifest.csv"
    if sha256_file(man_csv) != lock["full_dev"]["expected_csv_sha256"]:
        raise SystemExit("manifest hash drifted after lock")
    if sha256_file(ROOT / "src/earthquake/stage10/dkpn_picks.py") != lock["extract_picks"]["source_sha256"]:
        raise SystemExit("extract_picks changed after lock")
    if sha256_file(ROOT / "src/earthquake/stage10/full_dev_annotate.py") != lock["metric_impl"]["annotate_sha256"]:
        raise SystemExit("annotate impl changed after lock")
    if sha256_file(ROOT / "src/earthquake/stage10/full_dev_stop_gate.py") != lock["metric_impl"]["stop_gate_sha256"]:
        raise SystemExit("stop_gate impl changed after lock")
    if sha256_file(ROOT / "scripts/run_stage10_v4_full_dev_stop_gate.py") != lock["metric_impl"]["run_script_sha256"]:
        raise SystemExit("run script changed after lock")

    meta = pd.read_csv(man_csv)
    if len(meta) != 87293 or int(meta["event_id"].nunique()) != 5341:
        raise SystemExit(f"unexpected full-dev size {len(meta)} {meta['event_id'].nunique()}")

    pred_dir = art / "results" / "stage6" / "phaseC" / "baseline_preds"
    if sha256_file(pred_dir / "STEAD_top1.npy") != lock["baseline"]["stead_top1_npy_sha256"]:
        raise SystemExit("STEAD_top1.npy hash drifted")
    stead = np.load(pred_dir / "STEAD_top1.npy")
    union_top1 = np.load(pred_dir / "fixed_rescore_UNION.npy")
    if len(stead) != len(meta) or len(union_top1) != len(meta):
        raise SystemExit("baseline pred length mismatch")

    union_pq = pd.read_parquet(art / "cache" / "stage6" / "phaseC" / "dev_union.parquet")
    names = meta["trace_name"].astype(str).to_numpy()
    true = meta["s_arrival_sample"].to_numpy(float)
    sr = meta["sampling_rate_hz"].to_numpy(float)
    events = meta["event_id"].astype(str).to_numpy()

    idle = idle_gpu_indices()
    idle = [i for i in idle if i != 2]
    if not idle:
        raise SystemExit("no idle GPU (GPU2 vLLM excluded)")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(idle[0])
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    device = torch.device("cuda:0")

    ck = V4 / "checkpoints"
    primary_path = Path(lock["checkpoints"]["primary"]["path"])
    if sha256_file(primary_path) != lock["checkpoints"]["primary"]["file_sha256"]:
        raise SystemExit("primary checkpoint hash drifted")
    if primary_path.name != "epoch_1.pt":
        raise SystemExit("primary path is not epoch_1.pt")

    role_files = {
        "primary_epoch1": ck / "epoch_1.pt",
        "epoch_0": ck / "epoch_0.pt",
        "epoch_2": ck / "epoch_2.pt",
        "best_loss": ck / "best_loss.pt",
        "last": ck / "last.pt",
        "best_metric_pt_file": ck / "best_metric.pt",
    }
    param = {k: state_dict_tensor_sha256(torch.load(p, map_location="cpu", weights_only=False)["model"]) for k, p in role_files.items()}
    unique: dict[str, Path] = {}
    alias: dict[str, str] = {}
    for k, p in role_files.items():
        hit = None
        for u, up in unique.items():
            if param[k] == param[u]:
                hit = u
                break
        if hit is None:
            unique[k] = p
            alias[k] = k
        else:
            alias[k] = hit

    inst = resolve_instance_root()
    wave = WorkerHDF5WaveformSource(inst / "events" / "Instance_events_counts.hdf5", inst / "noise" / "Instance_noise.hdf5")
    ds = FullTraceDS(meta, wave, is_noise=False)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=8, collate_fn=collate, persistent_workers=False)

    models = {k: load_model(p, device) for k, p in unique.items()}
    print(json.dumps({"unique_models": list(unique), "alias": alias, "gpu": idle[0], "primary": str(primary_path)}, indent=2), flush=True)
    raw = infer_roles(models, loader, device, n=len(meta))
    collected = {role: raw[alias[role]] for role in role_files}

    for r in collected:
        if not np.array_equal(collected[r]["trace_name"], names):
            # keyed realign if worker order preserved by index
            df = pd.DataFrame({"trace_name": collected[r]["trace_name"], "pred_s_sample": collected[r]["pred"]})
            aligned = keyed_align_predictions(meta, df)
            if not np.array_equal(aligned["trace_name"].astype(str).to_numpy(), names):
                raise SystemExit(f"trace order mismatch {r}")
            collected[r]["pred"] = aligned["pred_s_sample"].to_numpy(float)

    out_dir = ensure_dir(V4 / "full_dev_stop_gate")
    reports = {}
    for role, pack in collected.items():
        pred = pack["pred"]
        m = metrics_pack(pred, true, sr, pack["n_peaks"])

        def closest(peaks_list):
            out = np.full(len(true), np.nan)
            for i, pk in enumerate(peaks_list):
                if not pk:
                    continue
                arr = np.asarray(pk, float)
                out[i] = arr[int(np.argmin(np.abs((arr - true[i]) / sr[i])))]
            return out

        ora5 = metrics_pack(closest(pack["peaks_k5"]), true, sr)
        ora10 = metrics_pack(closest(pack["peaks_k10"]), true, sr)
        dkpn_ok = correct_at(pred, true, sr, 0.5)
        stead_ok = correct_at(stead, true, sr, 0.5)
        union_ok = correct_at(union_top1, true, sr, 0.5)
        pe = per_event_table(meta, pred)
        groups, st_df = grouping_table(meta, pred, pack["n_peaks"])
        pe.to_csv(out_dir / f"per_event_{role}.csv", index=False)
        st_df.to_csv(out_dir / f"per_station_{role}.csv", index=False)
        pd.DataFrame(
            {
                "trace_name": names,
                "event_id": events,
                "pred_s_sample": pred,
                "n_peaks": pack["n_peaks"],
                "s_max": pack["s_max"],
            }
        ).to_parquet(out_dir / f"preds_{role}.parquet", index=False)
        reports[role] = {
            "role": role,
            "alias_of": alias[role],
            "param_sha256": param[role],
            "metrics": m,
            "oracle_k5": ora5,
            "oracle_k10": ora10,
            "dkpn_only_correct@0.5_vs_STEAD": int((dkpn_ok & ~stead_ok).sum()),
            "baseline_only_correct@0.5_vs_STEAD": int((stead_ok & ~dkpn_ok).sum()),
            "dkpn_only_correct@0.5_vs_UNION_top1": int((dkpn_ok & ~union_ok).sum()),
            "union_only_correct@0.5": int((union_ok & ~dkpn_ok).sum()),
            "grouping": groups,
            "is_primary": role == "primary_epoch1",
        }
        print(json.dumps({"role": role, "f1@0.5": m["f1@0.5"], "primary": role == "primary_epoch1"}), flush=True)

    primary = collected["primary_epoch1"]
    union_ora = oracle_metrics_for_set(union_pq, meta, sample_col="candidate_sample")
    union_ora_pred = union_ora["oracle_predictions"]
    dkpn5 = dkpn_k5_frame(names, events, sr, primary["peaks_k5"])
    union_plus = merge_union_dkpn5(union_pq, dkpn5)
    ud_pred, ud_ora = oracle_pred_from_candidates(union_plus, meta)

    stead_m = metrics_pack(stead, true, sr)
    union_top1_m = metrics_pack(union_top1, true, sr)
    union_ora_m = metrics_pack(union_ora_pred, true, sr)
    ud_m = metrics_pack(ud_pred, true, sr)

    boot_vs_stead = event_bootstrap_f1(events, stead, primary["pred"], true, sr, n_boot=N_BOOT, seed=BOOT_SEED)
    boot_ud = event_bootstrap_f1(events, union_ora_pred, ud_pred, true, sr, n_boot=N_BOOT, seed=BOOT_SEED)

    nf = noise_fpr(models[alias["primary_epoch1"]], wave, device)
    gate = stop_gates(
        dkpn=reports["primary_epoch1"]["metrics"],
        baseline=stead_m,
        union_oracle={"f1@0.5": union_ora_m["f1@0.5"]},
        union_plus_dkpn_oracle={"f1@0.5": ud_m["f1@0.5"]},
        boot_dkpn_minus_baseline=boot_vs_stead,
        boot_union_plus_minus_union=boot_ud,
    )

    confirm_atime_after = {"predictions": _atime(CONFIRM_PRED), "metrics": _atime(CONFIRM_METRICS)}
    confirm_unread = confirm_atime_before == confirm_atime_after
    if not confirm_unread:
        gate["passed"] = False
        gate["verdict"] = "rejected_candidate_source"
        gate["gates"]["confirm_atime_changed"] = True

    passed = bool(gate["passed"]) and confirm_unread
    if not confirm_unread:
        passed = False
        gate["verdict"] = "rejected_candidate_source"

    report = {
        "marker": MARK,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "not_PILOT.PASSED": True,
        "pilot_failed_retained": sha256_file(PILOT_FAILED) == FAILED_SHA,
        "pilot_adjudication_failed_retained": PILOT_ADJ.is_file(),
        "protocol_amendment_retained": AMENDMENT.is_file(),
        "primary": lock["checkpoints"]["primary"],
        "full_dev": {"n_traces": int(len(meta)), "n_events": int(meta["event_id"].nunique()), "csv_sha256": sha256_file(man_csv)},
        "strongest_waveform_only": STRONGEST_WAVEFORM_ONLY,
        "STEAD_top1": stead_m,
        "UNION_STEAD5_IDA5_fixed_rescore": union_top1_m,
        "UNION_oracle": {k: union_ora[k] for k in union_ora if k not in {"oracle_predictions", "true_samples", "sampling_rates", "event_ids"}},
        "UNION_oracle_metrics": union_ora_m,
        "UNION_plus_DKPN5_oracle": ud_m,
        "dkpn": reports,
        "stop_gate": gate,
        "noise_fpr_primary": nf,
        "confirm_waveforms_read": False,
        "confirm_metrics_read": False,
        "confirm_atime_unchanged": confirm_unread,
        "confirm_atime_before": confirm_atime_before,
        "confirm_atime_after": confirm_atime_after,
        "continued_training": False,
        "created_v5": False,
        "other_seeds": False,
        "height": HEIGHT,
    }
    save_json(jsonable(report), out_dir / "REPORT.json")
    save_json(jsonable(gate), out_dir / "stop_gate_verdict.json")
    save_json(jsonable(report), artifacts_dir() / "results" / "stage10" / "dkpn_v4_full_dev_stop_gate.json")

    flag = V4 / ("FULLDEV.STOP_GATE.PASSED" if passed else "FULLDEV.STOP_GATE.FAILED")
    flag.write_text(
        json.dumps(
            {
                "passed": passed,
                "verdict": gate["verdict"],
                "utc": datetime.now(timezone.utc).isoformat(),
                "not_PILOT.PASSED": True,
                "confirm_read": False,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"passed": passed, "verdict": gate["verdict"], "gates": gate["gates"], "confirm_unread": confirm_unread}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
