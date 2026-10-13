from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import optimize, special, stats


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=float) - self.mean) / self.scale


@dataclass(frozen=True)
class LogisticModel:
    coefficients: np.ndarray
    intercept: float
    standardizer: Standardizer
    c_value: float
    converged: bool

    def decision_function(self, values: np.ndarray) -> np.ndarray:
        return self.standardizer.transform(values) @ self.coefficients + self.intercept

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        return special.expit(self.decision_function(values))


def fit_standardizer(values: np.ndarray) -> Standardizer:
    array = np.asarray(values, dtype=float)
    mean = array.mean(axis=0)
    scale = array.std(axis=0, ddof=0)
    scale = np.where(scale > np.finfo(float).eps, scale, 1.0)
    return Standardizer(mean, scale)


def fit_logistic_l2(
    values: np.ndarray,
    labels: np.ndarray,
    c_value: float,
    *,
    maximum_iterations: int = 100,
    tolerance: float = 1e-8,
) -> LogisticModel:
    """Fit balanced L2 logistic regression without an external ML dependency."""
    x = np.asarray(values, dtype=float)
    y = np.asarray(labels, dtype=float).reshape(-1)
    if x.ndim != 2 or x.shape[0] != y.size or set(np.unique(y)) != {0.0, 1.0}:
        raise ValueError("values must be 2-D and labels must contain both 0 and 1")
    if c_value <= 0:
        raise ValueError("c_value must be positive")
    standardizer = fit_standardizer(x)
    z = standardizer.transform(x)
    counts = np.bincount(y.astype(int), minlength=2)
    weights = np.where(y == 1, y.size / (2 * counts[1]), y.size / (2 * counts[0]))
    weight_sum = weights.sum()

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        coefficients, intercept = parameters[:-1], parameters[-1]
        logits = z @ coefficients + intercept
        loss = np.sum(weights * (np.logaddexp(0.0, logits) - y * logits)) / weight_sum
        loss += 0.5 * np.dot(coefficients, coefficients) / c_value
        residual = weights * (special.expit(logits) - y) / weight_sum
        gradient = np.concatenate((z.T @ residual + coefficients / c_value, [residual.sum()]))
        return float(loss), gradient

    result = optimize.minimize(
        objective, np.zeros(x.shape[1] + 1), jac=True, method="L-BFGS-B",
        options={"maxiter": maximum_iterations, "ftol": tolerance, "gtol": tolerance},
    )
    return LogisticModel(
        coefficients=np.asarray(result.x[:-1]), intercept=float(result.x[-1]),
        standardizer=standardizer, c_value=float(c_value), converged=bool(result.success),
    )


def grouped_folds(groups: np.ndarray, folds: int, seed: int) -> list[np.ndarray]:
    unique = np.unique(np.asarray(groups))
    if not 2 <= folds <= unique.size:
        raise ValueError("fold count must lie in 2..number of groups")
    shuffled = unique.copy()
    np.random.default_rng(seed).shuffle(shuffled)
    return [np.asarray(part) for part in np.array_split(shuffled, folds)]


def select_regularization(
    values: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    c_grid: tuple[float, ...],
    folds: int,
    seed: int,
    maximum_iterations: int,
    tolerance: float,
) -> tuple[float, dict[float, float]]:
    """Grouped inner CV; equal scores select the strongest regularization (smallest C)."""
    x, y, group_values = np.asarray(values), np.asarray(labels), np.asarray(groups)
    fold_groups = grouped_folds(group_values, folds, seed)
    results: dict[float, float] = {}
    for c_value in sorted(c_grid):
        aucs = []
        for validation_groups in fold_groups:
            validation = np.isin(group_values, validation_groups)
            training = ~validation
            model = fit_logistic_l2(x[training], y[training], c_value,
                                    maximum_iterations=maximum_iterations, tolerance=tolerance)
            scores = model.decision_function(x[validation])
            if np.unique(y[validation]).size != 2:
                raise ValueError("every grouped validation fold must contain both classes")
            aucs.append(float(stats.rankdata(scores)[y[validation] == 1].sum()))
            n_pos = int(np.count_nonzero(y[validation] == 1)); n_neg = int(np.count_nonzero(y[validation] == 0))
            aucs[-1] = (aucs[-1] - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        results[float(c_value)] = float(np.mean(aucs))
    best_auc = max(results.values())
    best_c = min(c for c, auc in results.items() if np.isclose(auc, best_auc, rtol=0, atol=1e-12))
    return float(best_c), results


def make_training_observations(
    positive_features: np.ndarray,
    null_features: np.ndarray,
    participant_ids: np.ndarray,
    streams_per_condition: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average correlated subset/stream draws into participant-condition observations."""
    positive = np.asarray(positive_features, dtype=float)
    null = np.asarray(null_features, dtype=float)
    # participant x stream x subset x feature; null adds neighbor before feature
    if positive.ndim != 4 or null.ndim != 5 or positive.shape[:3] != null.shape[:3]:
        raise ValueError("unexpected feature tensor shapes")
    participants, streams = positive.shape[:2]
    if streams % streams_per_condition:
        raise ValueError("streams must divide evenly into conditions")
    conditions = streams // streams_per_condition
    pos = positive.reshape(participants, conditions, streams_per_condition, *positive.shape[2:]).mean(axis=(2, 3))
    neg = null.reshape(participants, conditions, streams_per_condition, *null.shape[2:]).mean(axis=(2, 3))
    pos_values = pos.reshape(-1, pos.shape[-1])
    neg_values = neg.reshape(-1, neg.shape[-1])
    values = np.concatenate((pos_values, neg_values), axis=0)
    labels = np.concatenate((np.ones(len(pos_values), dtype=np.int8), np.zeros(len(neg_values), dtype=np.int8)))
    group_base = np.repeat(np.asarray(participant_ids), conditions)
    groups = np.concatenate((group_base, np.repeat(group_base, null.shape[-2])))
    return values, labels, groups
