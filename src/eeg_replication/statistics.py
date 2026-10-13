from __future__ import annotations

from dataclasses import dataclass
import itertools

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class PCAResult:
    mean: np.ndarray
    loadings: np.ndarray
    scores: np.ndarray
    explained_variance_ratio: np.ndarray


def covariance_pca(values: np.ndarray, orient_first_index: int = 0) -> PCAResult:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("PCA input must be observations x features")
    mean = values.mean(axis=0)
    centered = values - mean
    u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    loadings = vt.T
    scores = u * singular
    for component in range(loadings.shape[1]):
        anchor = orient_first_index if component == 0 else int(np.argmax(np.abs(loadings[:, component])))
        if loadings[anchor, component] < 0:
            loadings[:, component] *= -1
            scores[:, component] *= -1
    variance = singular ** 2
    return PCAResult(mean, loadings, scores, variance / variance.sum())


def project_pca(values: np.ndarray, pca: PCAResult) -> np.ndarray:
    return (np.asarray(values, dtype=float) - pca.mean) @ pca.loadings


def rm_anova_oneway(values: np.ndarray) -> dict[str, float]:
    """One-factor repeated-measures ANOVA with Greenhouse-Geisser correction."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("values must be subject x level with at least two levels")
    if not np.all(np.isfinite(values)):
        raise ValueError("repeated-measures ANOVA requires finite values")
    n_subjects, n_levels = values.shape
    grand = values.mean()
    condition_means = values.mean(axis=0)
    subject_means = values.mean(axis=1)
    ss_condition = n_subjects * np.sum((condition_means - grand) ** 2)
    ss_subject = n_levels * np.sum((subject_means - grand) ** 2)
    ss_total = np.sum((values - grand) ** 2)
    ss_error = max(0.0, ss_total - ss_condition - ss_subject)
    df1 = n_levels - 1
    df2 = (n_subjects - 1) * df1
    ms_condition = ss_condition / df1
    ms_error = ss_error / df2
    f_value = ms_condition / ms_error if ms_error > 0 else np.inf
    p_uncorrected = float(stats.f.sf(f_value, df1, df2))

    covariance = np.cov(values, rowvar=False, ddof=1)
    centering = np.eye(n_levels) - np.ones((n_levels, n_levels)) / n_levels
    transformed = centering @ covariance
    denominator = df1 * np.trace(transformed @ transformed)
    epsilon = float(np.trace(transformed) ** 2 / denominator) if denominator > 0 else 1.0
    epsilon = min(1.0, max(1.0 / df1, epsilon))
    corrected_df1 = epsilon * df1
    corrected_df2 = epsilon * df2
    p_gg = float(stats.f.sf(f_value, corrected_df1, corrected_df2))
    return {
        "f": float(f_value),
        "df1": float(df1),
        "df2": float(df2),
        "p_uncorrected": p_uncorrected,
        "gg_epsilon": epsilon,
        "gg_df1": corrected_df1,
        "gg_df2": corrected_df2,
        "p_gg": p_gg,
    }


def paired_comparisons(values: np.ndarray, labels: tuple[str, ...]) -> list[dict[str, float | str]]:
    values = np.asarray(values, dtype=float)
    pairs = list(itertools.combinations(range(values.shape[1]), 2))
    rows = []
    for left, right in pairs:
        result = stats.ttest_rel(values[:, left], values[:, right])
        rows.append({
            "left": labels[left],
            "right": labels[right],
            "mean_difference": float(np.mean(values[:, left] - values[:, right])),
            "t": float(result.statistic),
            "p_uncorrected": float(result.pvalue),
            "p_bonferroni": float(min(1.0, result.pvalue * len(pairs))),
        })
    return rows


def bootstrap_ci(
    values: np.ndarray,
    statistic,
    iterations: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=float)
    for index in range(iterations):
        sample = rng.integers(0, values.shape[0], values.shape[0])
        estimates[index] = statistic(values[sample])
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(estimates, [tail, 1.0 - tail])
    return float(low), float(high)


def permutation_correlation(
    x: np.ndarray,
    y: np.ndarray,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    observed = float(stats.pearsonr(x, y).statistic)
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(iterations):
        permuted = rng.permutation(y)
        if abs(float(stats.pearsonr(x, permuted).statistic)) >= abs(observed):
            extreme += 1
    return {"r": observed, "p_permutation_two_sided": (extreme + 1) / (iterations + 1)}


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    z = float(stats.norm.ppf(0.5 + confidence / 2.0))
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = z * np.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)) / denominator
    return float(center - half), float(center + half)
