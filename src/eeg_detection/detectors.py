from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np


DETECTOR_NAMES = {
    "D1": "squared ITPC",
    "D2": "debiased PPC",
    "D3": "Rayleigh statistic",
    "D4": "phase-directed V-test",
    "D5": "complex Hotelling T2",
    "D6": "magnitude-squared coherence",
    "D7": "DPSS harmonic F-test",
    "D8": "equal-weight joint-harmonic statistic",
}


def deterministic_subsets(n_trials: int, k: int, cap: int, seed: int) -> np.ndarray:
    """Enumerate small subset spaces and deterministically sample larger ones."""
    if not 0 < k <= n_trials:
        raise ValueError("k must be in 1..n_trials")
    total = comb(n_trials, k)
    if total <= cap:
        return np.asarray(list(combinations(range(n_trials), k)), dtype=np.int16)
    all_subsets = np.asarray(list(combinations(range(n_trials), k)), dtype=np.int16)
    rng = np.random.default_rng(np.random.SeedSequence([seed, n_trials, k]))
    selected = np.sort(rng.choice(total, size=cap, replace=False))
    return all_subsets[selected]


def _unit_phase(coefficients: np.ndarray) -> np.ndarray:
    values = np.asarray(coefficients, dtype=np.complex128)
    magnitude = np.abs(values)
    return np.divide(values, magnitude, out=np.zeros_like(values), where=magnitude > 0)


def squared_itpc(coefficients: np.ndarray, trial_axis: int = 0) -> np.ndarray:
    return np.abs(np.mean(_unit_phase(coefficients), axis=trial_axis)) ** 2


def debiased_ppc(coefficients: np.ndarray, trial_axis: int = 0) -> np.ndarray:
    k = np.asarray(coefficients).shape[trial_axis]
    if k < 2: raise ValueError("PPC requires at least two trials")
    return (k * squared_itpc(coefficients, trial_axis) - 1.0) / (k - 1.0)


def rayleigh_statistic(coefficients: np.ndarray, trial_axis: int = 0) -> np.ndarray:
    return np.asarray(coefficients).shape[trial_axis] * squared_itpc(coefficients, trial_axis)


def v_test(coefficients: np.ndarray, expected_phase: np.ndarray | float = 0.0) -> np.ndarray:
    """Signed phase alignment; larger values indicate locking to the trained direction."""
    phases = np.angle(coefficients) - np.asarray(expected_phase)
    return np.mean(np.cos(phases), axis=0)


def estimate_neural_delay(
    coefficients: np.ndarray, frequencies_hz: np.ndarray, acoustic_phase: np.ndarray,
) -> float:
    """Estimate a single delay from training-only circular phase differences."""
    values = np.asarray(coefficients)
    mean_phase = np.angle(np.mean(_unit_phase(values), axis=tuple(range(values.ndim - 1))))
    differences = np.unwrap(mean_phase - np.asarray(acoustic_phase))
    slope = np.polyfit(2.0 * np.pi * np.asarray(frequencies_hz), differences, 1)[0]
    return float(-slope)


def hotelling_t2(coefficients: np.ndarray, regularization: float = 1e-8) -> np.ndarray:
    values = np.asarray(coefficients)
    k = values.shape[0]
    flattened = values.reshape(k, -1)
    scores = np.empty(flattened.shape[1], dtype=float)
    for index, column in enumerate(flattened.T):
        xy = np.column_stack((column.real, column.imag))
        mean = xy.mean(axis=0)
        covariance = np.cov(xy, rowvar=False, ddof=1)
        scale = max(float(np.trace(covariance)) / 2.0, 1.0)
        covariance = covariance + regularization * scale * np.eye(2)
        scores[index] = k * mean @ np.linalg.pinv(covariance) @ mean
    return scores.reshape(values.shape[1:])


def magnitude_squared_coherence(coefficients: np.ndarray) -> np.ndarray:
    values = np.asarray(coefficients)
    numerator = np.abs(np.sum(values, axis=0)) ** 2
    denominator = values.shape[0] * np.sum(np.abs(values) ** 2, axis=0)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator > 0)


def dpss_harmonic_f(taper_coefficients: np.ndarray) -> np.ndarray:
    """Line-to-residual ratio for trial x channel x harmonic x taper coefficients."""
    values = np.asarray(taper_coefficients)
    if values.ndim < 4: raise ValueError("D7 requires a final taper dimension")
    line = np.mean(values, axis=(0, -1))
    residual = values - line[None, ..., None]
    numerator = values.shape[0] * values.shape[-1] * np.abs(line) ** 2
    denominator = np.mean(np.abs(residual) ** 2, axis=(0, -1))
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator > 0)


def detector_score(
    detector: str,
    coefficients: np.ndarray,
    *,
    expected_phase: np.ndarray | float = 0.0,
    taper_coefficients: np.ndarray | None = None,
) -> float:
    """Apply published S0: compute channelwise statistics, then average channels."""
    values = np.asarray(coefficients)
    k = values.shape[0]
    if detector == "D1": statistic = squared_itpc(values)
    elif detector == "D2": statistic = debiased_ppc(values)
    elif detector == "D3": statistic = rayleigh_statistic(values)
    elif detector == "D4": statistic = v_test(values, expected_phase)
    elif detector == "D5": statistic = hotelling_t2(values)
    elif detector == "D6": statistic = magnitude_squared_coherence(values)
    elif detector == "D7":
        if taper_coefficients is None: raise ValueError("D7 requires taper_coefficients")
        statistic = dpss_harmonic_f(taper_coefficients)
    elif detector == "D8":
        per_harmonic = squared_itpc(values)
        if per_harmonic.ndim == 0: return float(2 * k * per_harmonic)
        return float(np.sum(2 * k * np.mean(per_harmonic, axis=0)))
    else: raise ValueError(f"Unknown detector {detector!r}")
    return float(np.mean(statistic))


def compatible(estimator: str, detector: str) -> bool:
    if detector == "D7": return estimator == "F3"
    if estimator == "F3": return True
    return detector != "D7"
