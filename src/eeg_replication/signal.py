from __future__ import annotations

import numpy as np
from scipy import signal


def squared_itpc(coefficients: np.ndarray, axis: int = -1) -> np.ndarray:
    values = np.asarray(coefficients)
    magnitude = np.abs(values)
    unit = np.divide(
        values,
        magnitude,
        out=np.full(values.shape, np.nan + 0j, dtype=np.complex128),
        where=magnitude > 0,
    )
    return np.abs(np.nanmean(unit, axis=axis)) ** 2


def published_fft_itpc(
    trials: np.ndarray,
    sampling_rate: int,
    harmonics_hz: tuple[float, ...],
    crop_samples: int = 500,
) -> np.ndarray:
    """Published 4.5-s symmetric-Hann FFT; return channel x harmonic squared ITPC."""
    central = np.asarray(trials, dtype=np.float64)[..., crop_samples:]
    window = np.hanning(central.shape[-1])
    bins = np.asarray([
        int(round(frequency * central.shape[-1] / sampling_rate))
        for frequency in harmonics_hz
    ])
    coefficients = np.fft.rfft(central * window, axis=-1)[..., bins]
    return squared_itpc(coefficients, axis=0)


def hilbert_itpc(
    trials: np.ndarray,
    sampling_rate: int,
    frequency: float,
    half_bandwidth_hz: float,
    order: int,
    edge_crop_seconds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return channel-wise mean R, mean R^2, and channel-mean R(t)."""
    time_itpc = hilbert_time_resolved_itpc(
        trials,
        sampling_rate,
        frequency,
        half_bandwidth_hz,
        order,
        edge_crop_seconds,
    )
    return (
        time_itpc.mean(axis=-1),
        np.square(time_itpc).mean(axis=-1),
        time_itpc.mean(axis=0),
    )


def hilbert_time_resolved_itpc(
    trials: np.ndarray,
    sampling_rate: int,
    frequency: float,
    half_bandwidth_hz: float,
    order: int,
    edge_crop_seconds: float,
) -> np.ndarray:
    """Return ordinary Hilbert ITPC as channel x time."""
    sos = signal.butter(
        order,
        [frequency - half_bandwidth_hz, frequency + half_bandwidth_hz],
        btype="bandpass",
        fs=sampling_rate,
        output="sos",
    )
    filtered = signal.sosfiltfilt(sos, np.asarray(trials, dtype=np.float64), axis=-1, padtype="odd")
    phase = np.angle(signal.hilbert(filtered, axis=-1))
    crop = int(round(edge_crop_seconds * sampling_rate))
    if crop:
        phase = phase[..., crop:-crop]
    return np.abs(np.mean(np.exp(1j * phase), axis=0))


def phase_randomize_trials(trials: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(trials, dtype=np.float64)
    spectrum = np.fft.rfft(values, axis=-1)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=spectrum.shape)
    phases[..., 0] = 0.0
    if values.shape[-1] % 2 == 0:
        phases[..., -1] = 0.0
    return np.fft.irfft(np.abs(spectrum) * np.exp(1j * phases), n=values.shape[-1], axis=-1)


def weighted_sinusoid_coefficients(
    trials: np.ndarray,
    sampling_rate: int,
    frequency: float,
    crop_samples: int = 500,
) -> np.ndarray:
    """Weighted least-squares line coefficient, returned as trial x channel complex values."""
    values = np.asarray(trials, dtype=np.float64)[..., crop_samples:]
    n_times = values.shape[-1]
    time = np.arange(n_times, dtype=float) / sampling_rate
    window = np.hanning(n_times)
    root_window = np.sqrt(window)
    design = np.column_stack([
        np.ones(n_times),
        np.cos(2.0 * np.pi * frequency * time),
        np.sin(2.0 * np.pi * frequency * time),
    ])
    weighted_design = design * root_window[:, None]
    projector = np.linalg.pinv(weighted_design)
    flat = values.reshape(-1, n_times)
    beta = projector @ (flat * root_window).T
    coefficient = beta[1] - 1j * beta[2]
    return coefficient.reshape(values.shape[:-1])


def hann_scalloping_loss(delta_hz: np.ndarray | float, duration_seconds: float) -> np.ndarray:
    """Normalized Hann response power at an offset from the analysed frequency."""
    delta = np.asarray(delta_hz, dtype=float)
    x = delta * duration_seconds
    amplitude = np.abs(
        np.sinc(x) + 0.5 * np.sinc(x - 1.0) + 0.5 * np.sinc(x + 1.0)
    )
    return np.square(amplitude)
