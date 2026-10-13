from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def unit_phasors(coefficients: np.ndarray) -> np.ndarray:
    """Return unit-magnitude coefficients, mapping exact zeros to zero."""
    values = np.asarray(coefficients)
    return np.divide(values, np.abs(values), out=np.zeros_like(values), where=np.abs(values) > 0)


def temporal_shift_multipliers(
    trials: int,
    harmonic_orders: tuple[int, ...],
    surrogates: int,
    seed: int,
    *,
    independent_harmonics: bool = False,
) -> np.ndarray:
    """Trial-wise phase rotations for a circular temporal-shift surrogate.

    The locked form draws one base phase for each trial and surrogate and applies
    ``exp(i * h * phi)`` to harmonic order ``h``.  It is intentionally shared
    over channels.  The sensitivity form draws a separate phase per harmonic.
    """
    if trials < 1 or surrogates < 1 or not harmonic_orders:
        raise ValueError("trials, surrogates, and harmonic orders must be non-empty")
    orders = np.asarray(harmonic_orders, dtype=float)
    if np.any(orders <= 0) or np.any(orders != np.floor(orders)):
        raise ValueError("harmonic orders must be positive integers")
    rng = np.random.default_rng(seed)
    if independent_harmonics:
        phase = rng.uniform(-np.pi, np.pi, size=(surrogates, trials, len(orders)))
        angles = phase
    else:
        phase = rng.uniform(-np.pi, np.pi, size=(surrogates, trials, 1))
        angles = phase * orders[None, None, :]
    return np.exp(1j * angles).astype(np.complex64)


def phase_trajectories(
    coefficients: np.ndarray,
    k_values: tuple[int, ...],
    region_indices: tuple[np.ndarray, ...] | None = None,
    rotations: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Compute global and optional regional ITPC-squared prefix trajectories.

    Coefficients are ``stream x trial x channel x harmonic``.  With rotations,
    the result gains a surrogate dimension after stream.
    """
    values = np.asarray(coefficients)
    if values.ndim != 4:
        raise ValueError("coefficients must be stream x trial x channel x harmonic")
    if not k_values or min(k_values) < 1 or max(k_values) > values.shape[1]:
        raise ValueError("invalid trial prefixes")
    units = unit_phasors(values)
    if rotations is None:
        cumulative = np.cumsum(units, axis=1)
        selected = cumulative[:, np.asarray(k_values) - 1]
        r2_channel = np.abs(selected / np.asarray(k_values)[None, :, None, None]) ** 2
    else:
        rotate = np.asarray(rotations)
        if rotate.ndim != 3 or rotate.shape[1:] != (values.shape[1], values.shape[3]):
            raise ValueError("rotations must be surrogate x trial x harmonic")
        randomized = units[:, None, :, :, :] * rotate[None, :, :, None, :]
        cumulative = np.cumsum(randomized, axis=2)
        selected = cumulative[:, :, np.asarray(k_values) - 1]
        r2_channel = np.abs(selected / np.asarray(k_values)[None, None, :, None, None]) ** 2
    global_scores = r2_channel.mean(axis=-2).astype(np.float32)
    if region_indices is None:
        return global_scores, None
    regional = np.stack(
        [r2_channel[..., np.asarray(indices, dtype=int), :].mean(axis=-2) for indices in region_indices],
        axis=-2,
    ).astype(np.float32)
    return global_scores, regional


def fixed_stream_scores(global_r2: np.ndarray, k_values: tuple[int, ...]) -> np.ndarray:
    """Create the locked S1 and S2 statistics from ITPC-squared trajectories."""
    values = np.asarray(global_r2, dtype=float)
    if values.shape[-1] < 2 or values.shape[-2] != len(k_values):
        raise ValueError("expected look x harmonic ITPC trajectories")
    scale_shape = (1,) * (values.ndim - 2) + (len(k_values),)
    scale = 2.0 * np.asarray(k_values, dtype=float).reshape(scale_shape)
    s1 = scale * values[..., 0]
    s2 = scale * (values[..., 0] + values[..., 1])
    return np.stack((s1, s2), axis=-1).astype(np.float32)


def regional_stream_features(regional_r2: np.ndarray, k_values: tuple[int, ...]) -> np.ndarray:
    """Flatten scaled 4/8-Hz regional ITPC into ten monotone features."""
    values = np.asarray(regional_r2, dtype=float)
    if values.shape[-1] < 2 or values.shape[-3] != len(k_values):
        raise ValueError("expected look x region x harmonic ITPC trajectories")
    scale_shape = (1,) * (values.ndim - 3) + (len(k_values), 1, 1)
    scaled = 2.0 * np.asarray(k_values, dtype=float).reshape(scale_shape) * values[..., :2]
    return scaled.reshape(*scaled.shape[:-2], scaled.shape[-2] * 2).astype(np.float32)


def ridge_complex_statistic(coefficients: np.ndarray, ridge: float) -> np.ndarray:
    """Channel-averaged ridge Hotelling statistic for complex 4/8-Hz data.

    The trial axis must be third from last; arbitrary leading batch dimensions
    are supported.  Shrinkage is toward the diagonal covariance, followed by a
    scale-aware numerical floor, so K=2 and constant inputs are well defined.
    """
    values = np.asarray(coefficients)
    if values.ndim < 3 or values.shape[-1] != 2 or not 0 <= ridge <= 1:
        raise ValueError("expected ... x trial x channel x 2 coefficients and ridge in [0,1]")
    trials = values.shape[-3]
    if trials < 2:
        raise ValueError("at least two trials are required")
    x = np.stack(
        (values[..., 0].real, values[..., 0].imag, values[..., 1].real, values[..., 1].imag), axis=-1
    ).astype(np.float64)
    mean = x.mean(axis=-3)
    centered = x - mean[..., None, :, :]
    covariance = np.einsum("...tci,...tcj->...cij", centered, centered, optimize=True) / (trials - 1)
    diagonal = np.zeros_like(covariance)
    index = np.arange(4)
    diagonal[..., index, index] = covariance[..., index, index]
    regularized = (1.0 - ridge) * covariance + ridge * diagonal
    scale = np.trace(diagonal, axis1=-2, axis2=-1) / 4.0
    floor = np.maximum(scale, 1.0) * (64.0 * np.finfo(np.float64).eps)
    regularized[..., index, index] += floor[..., None]
    inverse = np.linalg.pinv(regularized, hermitian=True)
    quadratic = np.einsum("...ci,...cij,...cj->...c", mean, inverse, mean, optimize=True)
    return (trials * np.maximum(quadratic, 0.0).mean(axis=-1)).astype(np.float32)


def ridge_complex_trajectories(
    coefficients: np.ndarray,
    k_values: tuple[int, ...],
    ridge: float,
    rotations: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(coefficients)[..., :2]
    if values.ndim != 4:
        raise ValueError("coefficients must be stream x trial x channel x harmonic")
    outputs = []
    for k in k_values:
        prefix = values[:, :k]
        if rotations is not None:
            rotate = np.asarray(rotations)[:, :k, :2]
            prefix = prefix[:, None, :, :, :] * rotate[None, :, :, None, :]
        outputs.append(ridge_complex_statistic(prefix, ridge))
    return np.stack(outputs, axis=-1).astype(np.float32)


@dataclass(frozen=True)
class PocockCalibration:
    boundaries: np.ndarray
    marginal_tail_probability: float
    familywise_fpr: float


def _higher_quantiles(values: np.ndarray, tail_probability: float) -> np.ndarray:
    try:
        return np.quantile(values, 1.0 - tail_probability, axis=0, method="higher")
    except TypeError:  # NumPy < 1.22
        return np.quantile(values, 1.0 - tail_probability, axis=0, interpolation="higher")


def crossing_rate(trajectories: np.ndarray, boundaries: np.ndarray) -> float:
    values = np.asarray(trajectories, dtype=float)
    limits = np.asarray(boundaries, dtype=float)
    if values.ndim != 2 or limits.shape != (values.shape[1],):
        raise ValueError("trajectories must be observation x look and boundaries one per look")
    return float(np.mean(np.any(values > limits[None, :], axis=1)))


def calibrate_pocock_boundaries(
    null_trajectories: np.ndarray, target_fpr: float = 0.05, iterations: int = 24
) -> PocockCalibration:
    """Calibrate equal-rank marginal boundaries using complete null paths.

    The exact rank formulation is equivalent to a Pocock-style search over a
    common marginal tail probability, but it needs one sort rather than many
    full-matrix crossing passes. ``iterations`` is retained for API stability.
    """
    del iterations
    values = np.asarray(null_trajectories, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1 or not 0 < target_fpr < 1:
        raise ValueError("invalid null trajectories or target FPR")
    if not np.all(np.isfinite(values)):
        raise ValueError("null trajectories must be finite")
    ordered = np.sort(values, axis=0)
    # A row crosses the common-rank boundary r iff at least one value has a
    # lower-bound rank greater than r. The (1-alpha) quantile of those row-wise
    # maxima is therefore the least conservative rank controlling the full path.
    lower_ranks = np.empty(values.shape, dtype=np.int64)
    for look in range(values.shape[1]):
        lower_ranks[:, look] = np.searchsorted(ordered[:, look], values[:, look], side="left")
    maximum_rank = lower_ranks.max(axis=1)
    index = int(np.ceil((1.0 - target_fpr) * values.shape[0])) - 1
    rank = int(np.sort(maximum_rank)[np.clip(index, 0, values.shape[0] - 1)])
    rank = int(np.clip(rank, 0, values.shape[0] - 1))
    boundaries = ordered[rank]
    fpr = crossing_rate(values, boundaries)
    tail = float((values.shape[0] - 1 - rank) / max(values.shape[0] - 1, 1))
    return PocockCalibration(np.asarray(boundaries, dtype=float), tail, fpr)


def first_crossings(
    trajectories: np.ndarray, boundaries: np.ndarray, k_values: tuple[int, ...], censor_at: int = 11
) -> np.ndarray:
    values = np.asarray(trajectories, dtype=float)
    if values.shape[-1] != len(k_values):
        raise ValueError("trajectory looks do not match K values")
    crossed = values > np.asarray(boundaries, dtype=float)
    any_crossed = crossed.any(axis=-1)
    first_index = np.argmax(crossed, axis=-1)
    return np.where(any_crossed, np.asarray(k_values)[first_index], censor_at).astype(np.int16)


def blend_boundaries(population: np.ndarray, local: np.ndarray, weight: float) -> np.ndarray:
    if not 0 <= weight <= 1:
        raise ValueError("weight must lie in [0,1]")
    population, local = np.asarray(population, dtype=float), np.asarray(local, dtype=float)
    if population.shape != local.shape:
        raise ValueError("boundary shapes differ")
    if weight == 0:
        return population.copy()
    if weight == 1:
        return local.copy()
    return (1.0 - weight) * population + weight * local


def select_shrinkage_weight(
    population_boundaries: np.ndarray,
    local_boundaries: np.ndarray,
    validation_trajectories: np.ndarray,
    weights: tuple[float, ...],
    target_fpr: float,
) -> tuple[float, dict[float, float]]:
    """Null-only pseudo-LOPO selection; ties choose less participant adaptation.

    ``local_boundaries`` is fold x look and ``validation_trajectories`` is
    fold x observation x look.  Each fold should hold out one local condition.
    """
    local = np.asarray(local_boundaries, dtype=float)
    validation = np.asarray(validation_trajectories, dtype=float)
    if local.ndim != 2 or validation.ndim != 3 or local.shape[0] != validation.shape[0] or local.shape[1] != validation.shape[2]:
        raise ValueError("invalid pseudo-LOPO boundary or validation arrays")
    if not np.all(np.isfinite(population_boundaries)) or not np.all(np.isfinite(validation)):
        raise ValueError("population boundaries and validation trajectories must be finite")
    errors: dict[float, float] = {}
    for weight in sorted(weights):
        if weight > 0 and not np.all(np.isfinite(local)):
            errors[float(weight)] = float("inf")
            continue
        rates = [
            crossing_rate(validation[fold], blend_boundaries(population_boundaries, local[fold], weight))
            for fold in range(local.shape[0])
        ]
        errors[float(weight)] = float(np.sqrt(np.mean((np.asarray(rates) - target_fpr) ** 2)))
    best = min(errors.values())
    selected = min(weight for weight, error in errors.items() if np.isclose(error, best, atol=1e-12, rtol=0))
    return float(selected), errors
