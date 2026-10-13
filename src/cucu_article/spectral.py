from __future__ import annotations

import numpy as np


def frequency_indices(
    sampling_rate: int,
    fft_samples: int,
    harmonics_hz: tuple[float, ...],
    neighbor_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if neighbor_count <= 0 or neighbor_count % 2:
        raise ValueError("neighbor_count must be a positive even integer")
    targets = np.rint(np.asarray(harmonics_hz) * fft_samples / sampling_rate).astype(int)
    half = neighbor_count // 2
    offsets = np.concatenate((np.arange(-half, 0), np.arange(1, half + 1)))
    neighbors = targets[:, None] + offsets[None, :]
    if np.any(neighbors < 0) or np.any(neighbors > fft_samples // 2):
        raise ValueError("neighbor frequencies fall outside the one-sided spectrum")
    return targets, neighbors, offsets


def hann_coefficients(
    trials: np.ndarray,
    sampling_rate: int,
    duration_seconds: float,
    fft_samples: int,
    harmonics_hz: tuple[float, ...],
    neighbor_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return target and neighbor coefficients, zero-padding shorter Hann windows."""
    values = np.asarray(trials, dtype=np.float64)
    samples = int(round(duration_seconds * sampling_rate))
    if samples <= 0 or samples > values.shape[-1] or fft_samples < samples:
        raise ValueError("invalid duration or FFT length")
    targets, neighbors, _ = frequency_indices(sampling_rate, fft_samples, harmonics_hz, neighbor_count)
    spectrum = np.fft.rfft(values[..., :samples] * np.hanning(samples), n=fft_samples, axis=-1)
    target = np.take(spectrum, targets, axis=-1)
    neighbor = np.take(spectrum, neighbors.reshape(-1), axis=-1).reshape(
        *values.shape[:-1], len(targets), neighbor_count
    )
    return target, neighbor


def phase_features(coefficients: np.ndarray, subsets: np.ndarray, *, null: bool = False) -> np.ndarray:
    values = np.asarray(coefficients)
    selected = values[subsets]
    units = np.divide(selected, np.abs(selected), out=np.zeros_like(selected), where=np.abs(selected) > 0)
    r2 = np.abs(units.mean(axis=1)) ** 2
    channel_average = r2.mean(axis=1)
    return np.moveaxis(channel_average, -1, -2) if null else channel_average


def _hotelling_vectorized(selected: np.ndarray, regularization: float = 1e-8) -> np.ndarray:
    values = np.asarray(selected)
    k = values.shape[1]
    x, y = values.real, values.imag
    mx, my = x.mean(axis=1), y.mean(axis=1)
    dx, dy = x - np.expand_dims(mx, 1), y - np.expand_dims(my, 1)
    a = np.sum(dx * dx, axis=1) / (k - 1)
    b = np.sum(dx * dy, axis=1) / (k - 1)
    c = np.sum(dy * dy, axis=1) / (k - 1)
    scale = np.maximum((a + c) / 2.0, np.finfo(float).eps)
    a, c = a + regularization * scale, c + regularization * scale
    determinant = a * c - b * b
    numerator = c * mx * mx - 2 * b * mx * my + a * my * my
    score = k * np.divide(numerator, determinant, out=np.zeros_like(numerator), where=determinant > 0)
    return score


def hotelling_scores(coefficients: np.ndarray, subsets: np.ndarray, *, null: bool = False) -> np.ndarray:
    """Complex Hotelling T2 at 4 Hz, averaged across channels."""
    values = np.asarray(coefficients)
    if null:
        # trial x channel x neighbor -> subset x trial x channel x neighbor
        return _hotelling_vectorized(values[subsets]).mean(axis=1)
    return _hotelling_vectorized(values[subsets]).mean(axis=1)


def score_bundle(target: np.ndarray, neighbors: np.ndarray, subsets: np.ndarray) -> dict[str, np.ndarray]:
    positive_phase = phase_features(target, subsets)
    null_phase = phase_features(neighbors, subsets, null=True)
    k = subsets.shape[1]
    return {
        "positive_phase": positive_phase.astype(np.float32),
        "null_phase": null_phase.astype(np.float32),
        "positive_hotelling": hotelling_scores(target[..., 0], subsets).astype(np.float32),
        "null_hotelling": hotelling_scores(neighbors[..., 0, :], subsets, null=True).astype(np.float32),
        "positive_joint": np.sum(2 * k * positive_phase, axis=-1).astype(np.float32),
        "null_joint": np.sum(2 * k * null_phase, axis=-1).astype(np.float32),
    }
