from __future__ import annotations

import numpy as np


def pick_from_prob(prob: np.ndarray, threshold: float = 0.3) -> dict[str, float]:
    prob = np.asarray(prob, dtype=np.float64)
    peak = int(np.argmax(prob))
    peak_prob = float(prob[peak])
    detected = peak_prob >= threshold
    return {
        "peak_sample": float(peak) if detected else float("nan"),
        "peak_probability": peak_prob,
        "detected": float(detected),
    }


def extract_pick_features(prob: np.ndarray) -> dict[str, float]:
    prob = np.asarray(prob, dtype=np.float64)
    peak = int(np.argmax(prob))
    peak_prob = float(prob[peak])
    # entropy
    p = prob / (prob.sum() + 1e-12)
    entropy = float(-(p * np.log(p + 1e-12)).sum())
    # peak width at half maximum
    half = peak_prob * 0.5
    left = peak
    while left > 0 and prob[left] >= half:
        left -= 1
    right = peak
    n = len(prob)
    while right < n - 1 and prob[right] >= half:
        right += 1
    width = float(right - left)
    return {
        "peak_sample": float(peak),
        "peak_probability": peak_prob,
        "entropy": entropy,
        "peak_width": width,
    }


def sample_to_seconds(sample: float, sampling_rate: float) -> float:
    return float(sample) / float(sampling_rate)
