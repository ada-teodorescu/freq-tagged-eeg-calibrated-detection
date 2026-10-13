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
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("PCA input must be observations x features")
    mean = values.mean(axis=0)
    u, singular, vt = np.linalg.svd(values - mean, full_matrices=False)
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
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2 or not np.all(np.isfinite(values)):
        raise ValueError("values must be finite subject x level data")
    subjects, levels = values.shape
    grand = values.mean()
    ss_level = subjects * np.sum((values.mean(0) - grand) ** 2)
    ss_subject = levels * np.sum((values.mean(1) - grand) ** 2)
    ss_error = max(0.0, np.sum((values - grand) ** 2) - ss_level - ss_subject)
    df1 = levels - 1
    df2 = (subjects - 1) * df1
    f_value = (ss_level / df1) / (ss_error / df2) if ss_error else np.inf
    covariance = np.cov(values, rowvar=False, ddof=1)
    centering = np.eye(levels) - np.ones((levels, levels)) / levels
    transformed = centering @ covariance
    denominator = df1 * np.trace(transformed @ transformed)
    epsilon = float(np.trace(transformed) ** 2 / denominator) if denominator else 1.0
    epsilon = min(1.0, max(1.0 / df1, epsilon))
    return {
        "f": float(f_value), "df1": float(df1), "df2": float(df2),
        "p_uncorrected": float(stats.f.sf(f_value, df1, df2)),
        "gg_epsilon": epsilon, "gg_df1": epsilon * df1, "gg_df2": epsilon * df2,
        "p_gg": float(stats.f.sf(f_value, epsilon * df1, epsilon * df2)),
    }


def paired_comparisons(values: np.ndarray, labels: tuple[str, ...]) -> list[dict[str, float | str]]:
    values = np.asarray(values, dtype=float)
    pairs = list(itertools.combinations(range(values.shape[1]), 2))
    rows = []
    for left, right in pairs:
        test = stats.ttest_rel(values[:, left], values[:, right])
        rows.append({
            "left": labels[left], "right": labels[right],
            "mean_difference": float(np.mean(values[:, left] - values[:, right])),
            "t": float(test.statistic), "p_uncorrected": float(test.pvalue),
            "p_bonferroni": float(min(1.0, test.pvalue * len(pairs))),
        })
    return rows


def agreement_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    left = np.asarray(candidate, dtype=np.float64).ravel()
    right = np.asarray(reference, dtype=np.float64).ravel()
    if left.shape != right.shape:
        raise ValueError(f"Agreement shapes differ: {left.shape} != {right.shape}")
    difference = left - right
    reference_rms = float(np.sqrt(np.mean(np.square(right))))
    candidate_rms = float(np.sqrt(np.mean(np.square(left))))
    return {
        "pearson_r": float(np.corrcoef(left, right)[0, 1]),
        "normalized_rmse": float(np.sqrt(np.mean(np.square(difference))) / reference_rms) if reference_rms else float("nan"),
        "amplitude_ratio": candidate_rms / reference_rms if reference_rms else float("nan"),
        "maximum_absolute_error": float(np.max(np.abs(difference))),
    }


def classify_agreement(
    metrics: dict[str, float],
    match_min_r: float,
    match_max_nrmse: float,
    approximate_min_r: float,
    approximate_max_nrmse: float,
) -> str:
    correlation = metrics.get("pearson_r", float("nan"))
    nrmse = metrics.get("normalized_rmse", float("nan"))
    if not np.isfinite(correlation) or not np.isfinite(nrmse):
        return "cannot_be_determined"
    if correlation >= match_min_r and nrmse <= match_max_nrmse:
        return "matches"
    if correlation >= approximate_min_r and nrmse <= approximate_max_nrmse:
        return "approximately_agrees"
    return "does_not_agree"
