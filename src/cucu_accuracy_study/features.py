from __future__ import annotations

import numpy as np


def exact_harmonic_coefficients(
    trials: np.ndarray,
    sampling_rate: int,
    harmonics_hz: tuple[float, ...],
    start_seconds: float,
    duration_seconds: float,
) -> np.ndarray:
    """Hann-tapered complex coefficients at exact frequencies."""
    values = np.asarray(trials, dtype=np.float64)
    start = int(round(start_seconds * sampling_rate))
    samples = int(round(duration_seconds * sampling_rate))
    stop = start + samples
    if values.ndim != 3 or start < 0 or samples < 2 or stop > values.shape[-1]:
        raise ValueError("invalid trial array or time window")
    time = np.arange(samples, dtype=float) / sampling_rate
    basis = np.hanning(samples)[:, None] * np.exp(
        -2j * np.pi * time[:, None] * np.asarray(harmonics_hz, dtype=float)[None, :]
    )
    return np.einsum("nct,th->nch", values[..., start:stop], basis, optimize=True).astype(np.complex64)


def amplitude_reliability_weights(
    coefficients: np.ndarray,
    harmonic_indices: tuple[int, ...] = (0, 1),
    lower_quantile: float = 0.1,
    upper_quantile: float = 0.9,
) -> np.ndarray:
    """Phase-independent trial weights from robust harmonic amplitude."""
    values = np.asarray(coefficients)
    if values.ndim != 3:
        raise ValueError("coefficients must be trial x channel x harmonic")
    quality = np.sqrt(np.mean(np.abs(values[..., list(harmonic_indices)]) ** 2, axis=(1, 2)))
    low, high = np.quantile(quality, (lower_quantile, upper_quantile))
    clipped = np.clip(quality, low, high)
    mean = float(clipped.mean())
    return np.ones_like(clipped) if mean <= np.finfo(float).eps else clipped / mean


def region_indices(channel_labels: tuple[str, ...], regions: tuple[tuple[str, ...], ...]) -> tuple[np.ndarray, ...]:
    lookup = {label: index for index, label in enumerate(channel_labels)}
    configured = [channel for region in regions for channel in region]
    if set(configured) != set(channel_labels) or len(configured) != len(channel_labels):
        raise ValueError("configured regions must exactly partition the recording channels")
    return tuple(np.asarray([lookup[channel] for channel in region], dtype=int) for region in regions)


def _phase_scores(units: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    if weights is None:
        mean = units.mean(axis=-3)
    else:
        normalized = np.asarray(weights, dtype=float) / np.sum(weights)
        mean = np.einsum("t,...tch->...ch", normalized, units, optimize=True)
    return np.abs(mean) ** 2


def condition_feature_bundle(
    coefficients: np.ndarray,
    regions: tuple[np.ndarray, ...],
    surrogate_seeds: tuple[int, ...],
    surrogates_per_seed: int,
    reliability_harmonic_indices: tuple[int, ...] = (0, 1),
    reliability_quantiles: tuple[float, float] = (0.1, 0.9),
    seed_salt: int = 0,
) -> dict[str, np.ndarray]:
    """Aggregate three streams per condition while preserving channel evidence."""
    values = np.asarray(coefficients)
    if values.ndim != 4 or values.shape[0] % 3:
        raise ValueError("coefficients must be stream x trial x channel x harmonic")
    streams, trials, channels, harmonics = values.shape
    conditions = streams // 3
    units = np.divide(values, np.abs(values), out=np.zeros_like(values), where=np.abs(values) > 0)
    weights = np.stack([
        amplitude_reliability_weights(
            values[stream], reliability_harmonic_indices,
            reliability_quantiles[0], reliability_quantiles[1],
        ) for stream in range(streams)
    ])
    positive_channel = _phase_scores(units)
    positive_weighted_channel = np.stack([
        _phase_scores(units[stream], weights[stream]) for stream in range(streams)
    ])
    positive_channel = positive_channel.reshape(conditions, 3, channels, harmonics).mean(axis=1)
    positive_weighted_channel = positive_weighted_channel.reshape(conditions, 3, channels, harmonics).mean(axis=1)

    total_surrogates = len(surrogate_seeds) * surrogates_per_seed
    null_channel = np.empty((conditions, total_surrogates, channels, harmonics), dtype=np.float32)
    null_weighted_channel = np.empty_like(null_channel)
    offset = 0
    for seed in surrogate_seeds:
        accum = np.zeros((conditions, surrogates_per_seed, channels, harmonics), dtype=np.float64)
        weighted_accum = np.zeros_like(accum)
        for stream in range(streams):
            rng = np.random.default_rng(np.random.SeedSequence([seed, seed_salt, stream]))
            rotations = np.exp(1j * rng.uniform(-np.pi, np.pi, size=(surrogates_per_seed, trials, 1, harmonics)))
            randomized = units[stream][None, ...] * rotations
            condition = stream // 3
            accum[condition] += _phase_scores(randomized)
            weighted_accum[condition] += _phase_scores(randomized, weights[stream])
        block = slice(offset, offset + surrogates_per_seed)
        null_channel[:, block] = (accum / 3).astype(np.float32)
        null_weighted_channel[:, block] = (weighted_accum / 3).astype(np.float32)
        offset += surrogates_per_seed

    def regional(array: np.ndarray) -> np.ndarray:
        return np.stack([array[..., indices, :].mean(axis=-2) for indices in regions], axis=-2)

    return {
        "positive_global": positive_channel.mean(axis=-2).astype(np.float32),
        "null_global": null_channel.mean(axis=-2).astype(np.float32),
        "positive_weighted_global": positive_weighted_channel.mean(axis=-2).astype(np.float32),
        "null_weighted_global": null_weighted_channel.mean(axis=-2).astype(np.float32),
        "positive_regional": regional(positive_channel).astype(np.float32),
        "null_regional": regional(null_channel).astype(np.float32),
    }


def matched_condition_observations(
    positive: np.ndarray,
    null: np.ndarray,
    participant_ids: np.ndarray,
    nulls_per_condition: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create same-unit positive and null condition observations for ML."""
    pos = np.asarray(positive, dtype=float)
    neg = np.asarray(null, dtype=float)
    if pos.ndim != 3 or neg.ndim != 4 or pos.shape[:2] != neg.shape[:2] or pos.shape[-1] != neg.shape[-1]:
        raise ValueError("expected participant x condition x feature tensors")
    if len(participant_ids) != pos.shape[0] or not 1 <= nulls_per_condition <= neg.shape[2]:
        raise ValueError("invalid participants or null sampling count")
    pos_values = pos.reshape(-1, pos.shape[-1])
    selected_null = []
    for participant in range(pos.shape[0]):
        for condition in range(pos.shape[1]):
            rng = np.random.default_rng(np.random.SeedSequence([seed, participant, condition]))
            selected = np.sort(rng.choice(neg.shape[2], nulls_per_condition, replace=False))
            selected_null.append(neg[participant, condition, selected])
    neg_values = np.asarray(selected_null).reshape(-1, neg.shape[-1])
    base_groups = np.repeat(np.asarray(participant_ids), pos.shape[1])
    groups = np.concatenate((base_groups, np.repeat(base_groups, nulls_per_condition)))
    labels = np.concatenate((np.ones(len(pos_values), dtype=np.int8), np.zeros(len(neg_values), dtype=np.int8)))
    return np.concatenate((pos_values, neg_values)), labels, groups
