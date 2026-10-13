from __future__ import annotations

import numpy as np


def participant_cluster_interval(participant_rates: np.ndarray, iterations: int, seed: int,
                                 one_sided_confidence: float = 0.95) -> dict[str, float]:
    values = np.asarray(participant_rates, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("participant rates must be a finite one-dimensional sample of size at least two")
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(iterations, len(values)))].mean(axis=1)
    return {"point_estimate": float(values.mean()),
            "ci95_low": float(np.quantile(draws, 0.025)), "ci95_high": float(np.quantile(draws, 0.975)),
            "one_sided_upper": float(np.quantile(draws, one_sided_confidence))}
