from __future__ import annotations

import numpy as np

from cucu_advanced_study.algorithms import calibrate_pocock_boundaries, first_crossings
from cucu_detection.evaluation import empirical_threshold, holm_adjust


METHODS = ("hann", "rectangular_ls", "hilbert")


def sinusoidal_ls_coefficients(values: np.ndarray, sampling_rate: int, frequency_hz: float,
                               start_sample: int = 0, *, hann: bool = False) -> np.ndarray:
    """Exact-frequency complex sinusoidal least-squares coefficients with an intercept."""
    data = np.asarray(values, dtype=np.float64)
    if data.ndim < 1 or data.shape[-1] < 3:
        raise ValueError("values must contain at least three time samples")
    time = (start_sample + np.arange(data.shape[-1], dtype=float)) / sampling_rate
    design = np.column_stack((np.ones(len(time)), np.cos(2 * np.pi * frequency_hz * time),
                              np.sin(2 * np.pi * frequency_hz * time)))
    flat = data.reshape(-1, data.shape[-1])
    if hann:
        root = np.sqrt(np.hanning(data.shape[-1]))
        beta = np.linalg.pinv(design * root[:, None]) @ (flat * root).T
    else:
        beta = np.linalg.pinv(design) @ flat.T
    return (beta[1] - 1j * beta[2]).reshape(data.shape[:-1])


def hilbert_interval_coefficients(analytic: np.ndarray, sampling_rate: int, frequency_hz: float,
                                  start_sample: int) -> np.ndarray:
    data = np.asarray(analytic)
    if not np.iscomplexobj(data):
        raise ValueError("analytic input must be complex")
    sample = start_sample + np.arange(data.shape[-1], dtype=float)
    demodulator = np.exp(-2j * np.pi * frequency_hz * sample / sampling_rate)
    return np.mean(data * demodulator, axis=-1)


def phase_rotations(surrogates: int, trials: int, seed: int | np.random.SeedSequence) -> np.ndarray:
    if surrogates < 1 or trials < 1:
        raise ValueError("surrogates and trials must be positive")
    rng = np.random.default_rng(seed)
    return np.exp(1j * rng.uniform(-np.pi, np.pi, size=(surrogates, trials))).astype(np.complex64)


def _unit(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values)
    return np.divide(data, np.abs(data), out=np.zeros_like(data), where=np.abs(data) > 0)


def score_trajectories(coefficients: np.ndarray, k_values: tuple[int, ...],
                       rotations: np.ndarray | None = None) -> np.ndarray:
    """Return 2K times equal-channel ITPC-squared for block x trial x channel data."""
    values = np.asarray(coefficients)
    if values.ndim != 3:
        raise ValueError("coefficients must be block x trial x channel")
    if not k_values or min(k_values) < 1 or max(k_values) > values.shape[1]:
        raise ValueError("invalid K values")
    unit = _unit(values)
    indices = np.asarray(k_values) - 1
    scale = 2.0 * np.asarray(k_values, dtype=float)
    if rotations is None:
        selected = np.cumsum(unit, axis=1)[:, indices]
        r2 = np.abs(selected / np.asarray(k_values)[None, :, None]) ** 2
        return (scale[None, :] * r2.mean(axis=-1)).astype(np.float32)
    rotate = np.asarray(rotations)
    if rotate.ndim == 2:
        rotate = np.broadcast_to(rotate, (values.shape[0], *rotate.shape))
    if rotate.ndim != 3 or rotate.shape[0] != values.shape[0] or rotate.shape[2] != values.shape[1]:
        raise ValueError("rotations must be block x surrogate x trial")
    output = np.empty((values.shape[0], rotate.shape[1], len(k_values)), dtype=np.float32)
    for block in range(values.shape[0]):
        randomized = unit[block][None, :, :] * rotate[block, :, :, None]
        selected = np.cumsum(randomized, axis=1)[:, indices]
        r2 = np.abs(selected / np.asarray(k_values)[None, :, None]) ** 2
        output[block] = scale[None, :] * r2.mean(axis=-1)
    return output


def heldout_static(positive: np.ndarray, null: np.ndarray, participants: tuple[str, ...],
                   target_fpr: float) -> list[dict]:
    """Participant-held-out K=10 calibration for balanced participant arrays."""
    pos, nul = np.asarray(positive), np.asarray(null)
    if pos.ndim != 2 or nul.ndim != 3 or pos.shape[:2] != nul.shape[:2] or pos.shape[0] != len(participants):
        raise ValueError("static arrays must be participant x block [x surrogate]")
    rows = []
    for held, participant in enumerate(participants):
        training = np.arange(len(participants)) != held
        threshold = empirical_threshold(nul[training].reshape(-1), target_fpr)
        held_null = nul[held].reshape(-1)
        fpr = float(np.mean(held_null > threshold))
        for block, score in enumerate(pos[held], 1):
            rows.append({"participant": participant, "block": block, "score": float(score),
                         "threshold": threshold, "detected": bool(score > threshold), "heldout_fpr": fpr})
    return rows


def heldout_sequential(positive: np.ndarray, null: np.ndarray, participants: tuple[str, ...],
                       k_values: tuple[int, ...], target_fpr: float) -> tuple[list[dict], list[dict]]:
    pos, nul = np.asarray(positive), np.asarray(null)
    if pos.ndim != 3 or nul.ndim != 4 or pos.shape[:2] != nul.shape[:2] or pos.shape[0] != len(participants):
        raise ValueError("trajectory arrays must be participant x block [x surrogate] x look")
    rows, summaries = [], []
    for held, participant in enumerate(participants):
        training = np.arange(len(participants)) != held
        calibration = nul[training].reshape(-1, len(k_values))
        fitted = calibrate_pocock_boundaries(calibration, target_fpr)
        first = first_crossings(pos[held], fitted.boundaries, k_values)
        null_first = first_crossings(nul[held].reshape(-1, len(k_values)), fitted.boundaries, k_values)
        crossed = np.maximum.accumulate(pos[held] > fitted.boundaries[None, :], axis=1)
        summaries.append({"participant": participant, "detection": float(np.mean(first <= k_values[-1])),
                          "rmst_k": float(np.mean(first)), "familywise_fpr": float(np.mean(null_first <= k_values[-1])),
                          "marginal_tail_probability": fitted.marginal_tail_probability})
        for block in range(pos.shape[1]):
            row = {"participant": participant, "block": block + 1,
                   "first_detection_k": int(first[block]), "detected": bool(first[block] <= k_values[-1])}
            row.update({f"score_k{k}": float(pos[held, block, look]) for look, k in enumerate(k_values)})
            row.update({f"boundary_k{k}": float(fitted.boundaries[look]) for look, k in enumerate(k_values)})
            row.update({f"detected_by_k{k}": bool(crossed[block, look]) for look, k in enumerate(k_values)})
            rows.append(row)
    return rows, summaries


def participant_bootstrap(values: np.ndarray, iterations: int, seed: int) -> dict[str, float]:
    data = np.asarray(values, dtype=float).reshape(-1)
    if len(data) < 2 or not np.all(np.isfinite(data)):
        raise ValueError("participant bootstrap needs at least two finite values")
    rng = np.random.default_rng(seed)
    draws = data[rng.integers(0, len(data), size=(iterations, len(data)))].mean(axis=1)
    return {"estimate": float(data.mean()), "ci95_low": float(np.quantile(draws, 0.025)),
            "ci95_high": float(np.quantile(draws, 0.975))}


def simultaneous_participant_band(values: np.ndarray, iterations: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(values, dtype=float)
    if data.ndim != 2 or data.shape[0] < 2:
        raise ValueError("values must be participant x look")
    estimate = data.mean(axis=0)
    rng = np.random.default_rng(seed)
    draws = data[rng.integers(0, data.shape[0], size=(iterations, data.shape[0]))].mean(axis=1)
    radius = float(np.quantile(np.max(np.abs(draws - estimate), axis=1), 0.95))
    return estimate - radius, estimate + radius


def paired_sign_flip(values: np.ndarray, iterations: int, seed: int) -> float:
    data = np.asarray(values, dtype=float).reshape(-1)
    observed = abs(float(data.mean()))
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(iterations):
        exceed += abs(float(np.mean(data * rng.choice((-1.0, 1.0), len(data))))) >= observed
    return float((exceed + 1) / (iterations + 1))


__all__ = ["METHODS", "sinusoidal_ls_coefficients", "hilbert_interval_coefficients", "phase_rotations",
           "score_trajectories", "heldout_static", "heldout_sequential", "participant_bootstrap",
           "simultaneous_participant_band", "paired_sign_flip", "holm_adjust"]
