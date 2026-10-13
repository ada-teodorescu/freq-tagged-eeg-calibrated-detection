from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HarmonicResult:
    coefficients: np.ndarray
    channel_itpc: np.ndarray
    mean_itpc: np.ndarray
    frequencies_hz: np.ndarray


def squared_itpc(coefficients: np.ndarray, axis: int) -> np.ndarray:
    values = np.asarray(coefficients)
    magnitudes = np.abs(values)
    unit = np.divide(
        values,
        magnitudes,
        out=np.full(values.shape, np.nan + 0j, dtype=np.complex128),
        where=magnitudes > 0,
    )
    return np.square(np.abs(np.nanmean(unit, axis=axis)))


def harmonic_coefficients(
    trials: np.ndarray,
    sampling_rate: int,
    harmonics_hz: tuple[float, ...],
    crop_start_samples: int,
) -> np.ndarray:
    """Return amplitude-normalized trial x channel x harmonic coefficients."""
    values = np.asarray(trials, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("trials must have shape trial x channel x time")
    central = values[..., crop_start_samples:]
    window = np.hanning(central.shape[-1])
    bins = np.rint(np.asarray(harmonics_hz) * central.shape[-1] / sampling_rate).astype(int)
    realized = bins * sampling_rate / central.shape[-1]
    if not np.allclose(realized, harmonics_hz, atol=1e-12):
        raise ValueError(f"Requested harmonics are not exact FFT bins: {realized.tolist()}")
    fft = np.fft.rfft(central * window, axis=-1)
    return (2.0 / window.sum() * fft[..., bins]).astype(np.complex64)


def analyse_epochs(
    epochs: np.ndarray,
    stream_ids: np.ndarray,
    sampling_rate: int,
    harmonics_hz: tuple[float, ...],
    crop_start_samples: int,
    streams: int = 45,
    trials_per_stream: int = 10,
) -> HarmonicResult:
    values = np.asarray(epochs)
    stream_ids = np.asarray(stream_ids, dtype=int)
    if values.shape[0] != stream_ids.size:
        raise ValueError("Epoch and stream ID counts differ")
    coefficient_rows = []
    for stream in range(1, streams + 1):
        selected = values[stream_ids == stream]
        if selected.shape[0] != trials_per_stream:
            raise ValueError(f"Stream {stream}: expected {trials_per_stream} trials, got {selected.shape[0]}")
        coefficient_rows.append(harmonic_coefficients(
            selected, sampling_rate, harmonics_hz, crop_start_samples,
        ))
    coefficients = np.stack(coefficient_rows).astype(np.complex64)
    channel_itpc = squared_itpc(coefficients, axis=1).astype(np.float32)
    return HarmonicResult(
        coefficients=coefficients,
        channel_itpc=channel_itpc,
        mean_itpc=channel_itpc.mean(axis=1).astype(np.float32),
        frequencies_hz=np.asarray(harmonics_hz, dtype=np.float64),
    )
