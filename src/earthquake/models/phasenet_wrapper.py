from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from earthquake.models.seisbench_reference import (
    SeisBenchPhaseNetReference,
    peak_utc_from_trace,
    sample_index_on_waveform,
    stream_from_row,
)

FORBIDDEN_WEIGHT_TOKENS = ("instance",)


def list_pretrained_weights() -> list[str]:
    import seisbench.models as sbm

    return list(sbm.PhaseNet.list_pretrained())


def select_external_weight(preferred: list[str] | None = None) -> str:
    available = list_pretrained_weights()
    preferred = preferred or ["stead", "ethz", "ncedc", "scedc", "geofon", "neic"]
    safe = [w for w in available if not any(tok in w.lower() for tok in FORBIDDEN_WEIGHT_TOKENS)]
    for name in preferred:
        for w in safe:
            if name.lower() in w.lower():
                return w
    if not safe:
        raise RuntimeError(
            f"No safe PhaseNet weights found. Available={available}. "
            "Refusing INSTANCE-trained weights due to leakage risk."
        )
    return safe[0]


class PhaseNetWrapper:
    """PhaseNet wrapper aligned to SeisBench official annotate() via UTC remapping.

    Primary inference path (predict_proba / predict_row) uses ObsPy Stream + annotate(),
    then remaps probability traces onto the original waveform sample grid by absolute time.
    This avoids silent offsets from padding/overlap/output starttime differences.

    Low-level forward_window() remains for finetuning on fixed in_samples crops.
    """

    def __init__(
        self,
        weight: str | None = None,
        device: str | None = None,
        allow_instance: bool = False,
    ):
        import seisbench.models as sbm

        self.weight = weight or select_external_weight()
        self.allow_instance = bool(allow_instance)
        if (not self.allow_instance) and any(tok in self.weight.lower() for tok in FORBIDDEN_WEIGHT_TOKENS):
            raise ValueError(f"Refusing potentially leaky PhaseNet weight: {self.weight}")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = sbm.PhaseNet.from_pretrained(self.weight)
        self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        # Shared reference helper using the same loaded weight name
        self._ref = SeisBenchPhaseNetReference(
            weight=self.weight, device=self.device, allow_instance=self.allow_instance
        )
        # Ensure both share the same model instance for exact agreement
        self._ref.model = self.model

    def info(self) -> dict[str, Any]:
        return {
            "weight": self.weight,
            "device": self.device,
            "forbidden_tokens": list(FORBIDDEN_WEIGHT_TOKENS),
            "allow_instance": self.allow_instance,
            "leakage_risk": (
                "diagnostic_only_data_leakage"
                if self.allow_instance and "instance" in self.weight.lower()
                else "low_external_pretrain"
            ),
            "inference_path": "seisbench.annotate + UTC remap onto waveform grid",
            "component_handling": "ObsPy channels from ENZ array with correct E/N/Z suffixes",
            "source": "seisbench.models.PhaseNet.from_pretrained",
        }

    @torch.no_grad()
    def forward_window(self, waveform_zne: np.ndarray | torch.Tensor) -> np.ndarray:
        """Low-level forward on a single ZNE window of length ~= in_samples. Returns (3, L)."""
        if isinstance(waveform_zne, np.ndarray):
            x = torch.from_numpy(waveform_zne.astype(np.float32))
        else:
            x = waveform_zne.float()
        if x.ndim != 2 or x.shape[0] != 3:
            raise ValueError(f"Expected ZNE (3, T), got {tuple(x.shape)}")
        win = int(getattr(self.model, "in_samples", 3001) or 3001)
        t = x.shape[-1]
        if t < win:
            pad = torch.zeros(3, win, dtype=x.dtype)
            pad[:, :t] = x
            xw = pad
            cut = t
        else:
            xw = x[:, :win]
            cut = min(t, win)
        xb = xw.unsqueeze(0).to(self.device)
        xb = xb - xb.mean(dim=-1, keepdim=True)
        xb = xb / xb.std(dim=-1, keepdim=True).clamp_min(1e-6)
        y = self.model(xb).squeeze(0).detach().cpu().numpy()
        return y[:, :cut].astype(np.float32)

    def predict_row(self, waveform_enz: np.ndarray, row: pd.Series, **annotate_kwargs) -> dict[str, Any]:
        """Official-aligned prediction for one metadata row. waveform is ENZ (3, T)."""
        out = self._ref.predict_row(waveform_enz, row, remap_to_waveform=True, **annotate_kwargs)
        out["wrapper"] = "PhaseNetWrapper"
        return out

    @torch.no_grad()
    def predict_proba(
        self,
        waveform: np.ndarray | torch.Tensor,
        row: pd.Series | None = None,
        **annotate_kwargs,
    ) -> dict[str, np.ndarray]:
        """Return noise/p/s probs on the original waveform grid.

        If `row` metadata is provided, uses annotate()+UTC remap (recommended).
        Without metadata, falls back to a deprecated sliding-window path that assumes
        sample0 alignment and should not be used for evaluation.
        """
        if isinstance(waveform, torch.Tensor):
            wave = waveform.detach().cpu().numpy().astype(np.float32)
        else:
            wave = np.asarray(waveform, dtype=np.float32)
        if row is not None:
            pred = self.predict_row(wave, row, **annotate_kwargs)
            return {"noise": pred["noise"], "p": pred["p"], "s": pred["s"]}
        # Fallback only for debugging — not evaluation-safe
        return self._legacy_sliding_window_proba(wave)

    def _legacy_sliding_window_proba(self, waveform_enz: np.ndarray) -> dict[str, np.ndarray]:
        """Legacy path kept for ablation; may disagree with annotate timing."""
        x = torch.from_numpy(waveform_enz.astype(np.float32))
        # ENZ -> ZNE
        x = torch.stack([x[2], x[1], x[0]], dim=0)
        win = int(getattr(self.model, "in_samples", 3001) or 3001)
        overlap = 1500
        step = max(win - overlap, 1)
        t = int(x.shape[-1])
        acc = np.zeros((3, t), dtype=np.float64)
        wsum = np.zeros(t, dtype=np.float64)
        starts = list(range(0, max(t - win, 0) + 1, step))
        if not starts or starts[-1] != max(t - win, 0):
            starts.append(max(t - win, 0))
        for s0 in starts:
            y = self.forward_window(x[:, s0 : s0 + win])
            L = y.shape[-1]
            acc[:, s0 : s0 + L] += y
            wsum[s0 : s0 + L] += 1.0
        wsum = np.maximum(wsum, 1e-6)
        y = (acc / wsum[None, :]).astype(np.float32)
        return {"noise": y[0], "p": y[1], "s": y[2]}

    def pick_samples_utc(self, waveform_enz: np.ndarray, row: pd.Series) -> dict[str, float]:
        pred = self.predict_row(waveform_enz, row)
        return {
            "p_sample": float(pred["p_pred_sample_on_waveform"]),
            "s_sample": float(pred["s_pred_sample_on_waveform"]),
            "p_prob": float(pred["p_peak_probability"]),
            "s_prob": float(pred["s_peak_probability"]),
            "p_utc": pred["p_peak_utc"],
            "s_utc": pred["s_peak_utc"],
            "output_starttime": pred["p_output_starttime"],
            "output_npts": pred["p_output_npts"],
        }

    def save_info(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.info(), f, indent=2)
