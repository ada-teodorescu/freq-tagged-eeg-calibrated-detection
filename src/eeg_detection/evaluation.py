from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import stats


@dataclass(frozen=True)
class CalibrationResult:
    threshold: float
    false_positive_rate: float
    ci_low: float
    ci_high: float
    n_null: int


def empirical_threshold(null_scores: np.ndarray, target_fpr: float = 0.05) -> float:
    values = np.sort(np.asarray(null_scores, dtype=float).reshape(-1))
    if values.size == 0: raise ValueError("No calibration scores")
    # Conservative finite-sample upper-tail threshold.
    index = int(np.ceil((1.0 - target_fpr) * (values.size + 1))) - 1
    return float(values[np.clip(index, 0, values.size - 1)])


def binomial_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    if trials <= 0: return float("nan"), float("nan")
    alpha = 1.0 - confidence
    low = 0.0 if successes == 0 else stats.beta.ppf(alpha / 2, successes, trials - successes + 1)
    high = 1.0 if successes == trials else stats.beta.ppf(1 - alpha / 2, successes + 1, trials - successes)
    return float(low), float(high)


def calibrate(train_null: np.ndarray, heldout_null: np.ndarray, target_fpr: float = 0.05) -> CalibrationResult:
    threshold = empirical_threshold(train_null, target_fpr)
    heldout = np.asarray(heldout_null).reshape(-1)
    positives = int(np.count_nonzero(heldout > threshold))
    low, high = binomial_interval(positives, heldout.size)
    return CalibrationResult(threshold, positives / heldout.size, low, high, heldout.size)


def auc_from_scores(signal_scores: np.ndarray, null_scores: np.ndarray) -> float:
    signal = np.asarray(signal_scores).reshape(-1)
    null = np.asarray(null_scores).reshape(-1)
    ranks = stats.rankdata(np.concatenate((signal, null)))
    rank_sum = ranks[: signal.size].sum()
    return float((rank_sum - signal.size * (signal.size + 1) / 2) / (signal.size * null.size))


def trials_to_target(k_values: np.ndarray, detection_rates: np.ndarray, target: float = 0.8) -> int | None:
    for k, rate in zip(k_values, detection_rates, strict=True):
        if rate >= target: return int(k)
    return None


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    m = values.size
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def sign_flip_test(differences: np.ndarray, iterations: int, seed: int) -> float:
    values = np.asarray(differences, dtype=float)
    observed = float(np.mean(values))
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(iterations):
        statistic = np.mean(values * rng.choice((-1.0, 1.0), size=values.size))
        exceed += abs(statistic) >= abs(observed)
    return float((exceed + 1) / (iterations + 1))


def crossed_bootstrap_gain(
    gains: np.ndarray, iterations: int, seed: int,
) -> tuple[float, float]:
    """Crossed bootstrap for participant x condition gain matrices."""
    values = np.asarray(gains, dtype=float)
    if values.ndim != 2: raise ValueError("gains must be participant x condition")
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations)
    for index in range(iterations):
        participants = rng.integers(0, values.shape[0], values.shape[0])
        conditions = rng.integers(0, values.shape[1], values.shape[1])
        draws[index] = values[np.ix_(participants, conditions)].mean()
    return tuple(float(value) for value in np.quantile(draws, (0.025, 0.975)))
