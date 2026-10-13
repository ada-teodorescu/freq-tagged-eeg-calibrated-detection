from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile

from .config import ReplicationConfig, condition_for_stream


@dataclass(frozen=True)
class AcousticResult:
    stream_id: int
    condition_id: int
    condition: str
    sampling_rate: int
    duration_seconds: float
    onsets_seconds: np.ndarray
    candidate_indices: np.ndarray
    fallback: bool
    features: dict[str, float]


def float_audio(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 2:
        values = values.mean(axis=1)
    if np.issubdtype(values.dtype, np.integer):
        scale = max(abs(np.iinfo(values.dtype).min), np.iinfo(values.dtype).max)
        return values.astype(np.float64) / float(scale)
    return values.astype(np.float64)


def cochlear_envelope(audio: np.ndarray, sampling_rate: int, lowpass_hz: float = 20.0) -> np.ndarray:
    # A logarithmic auditory-like bank is used instead of scipy's low-frequency
    # IIR gammatone implementation, which is numerically unstable for these
    # 44.1-kHz, five-second streams on some SciPy builds.
    edges = np.geomspace(80.0, min(8000.0, sampling_rate * 0.45), 33)
    envelope = np.zeros(audio.size, dtype=np.float64)
    for low, high in zip(edges[:-1], edges[1:]):
        sos_band = signal.butter(2, [float(low), float(high)], btype="bandpass", fs=sampling_rate, output="sos")
        band = signal.sosfiltfilt(sos_band, audio, padtype="odd")
        envelope += np.abs(signal.hilbert(band))
    sos = signal.butter(4, lowpass_hz, btype="lowpass", fs=sampling_rate, output="sos")
    envelope = signal.sosfiltfilt(sos, envelope)
    envelope = np.maximum(envelope, 0.0)
    scale = float(np.percentile(envelope, 99.0))
    return envelope / max(scale, np.finfo(float).eps)


def select_landmarks(
    peakrate: np.ndarray,
    sampling_rate: int,
    expected: int,
    min_distance_ms: int,
    target_ioi_ms: int,
    soft_scale_ms: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    distance = max(1, int(round(min_distance_ms * sampling_rate / 1000.0)))
    candidates, properties = signal.find_peaks(peakrate, distance=distance, prominence=0.0)
    if candidates.size < expected:
        candidates, properties = signal.find_peaks(peakrate, distance=max(1, distance // 2), prominence=0.0)
    if candidates.size < expected:
        raise ValueError(f"Only {candidates.size} measured acoustic peaks; need {expected}")
    prominence = np.asarray(properties.get("prominences", peakrate[candidates]), dtype=float)
    prominence = (prominence - prominence.mean()) / max(prominence.std(), 1e-12)
    target = target_ioi_ms * sampling_rate / 1000.0
    scale = soft_scale_ms * sampling_rate / 1000.0
    n_candidates = candidates.size
    score = np.full((expected, n_candidates), -np.inf)
    previous = np.full((expected, n_candidates), -1, dtype=int)
    score[0] = prominence
    for count in range(1, expected):
        for current in range(count, n_candidates):
            earlier = np.arange(count - 1, current)
            intervals = candidates[current] - candidates[earlier]
            transition = score[count - 1, earlier] - np.square((intervals - target) / scale)
            best_local = int(np.argmax(transition))
            best_previous = int(earlier[best_local])
            if np.isfinite(transition[best_local]):
                score[count, current] = transition[best_local] + prominence[current]
                previous[count, current] = best_previous
    end = int(np.argmax(score[-1]))
    selected = [end]
    for count in range(expected - 1, 0, -1):
        end = int(previous[count, end])
        if end < 0:
            raise RuntimeError("Landmark dynamic program did not find a complete path")
        selected.append(end)
    selected = np.asarray(selected[::-1], dtype=int)
    indices = candidates[selected]
    fallback = bool(
        np.any(np.diff(indices) < 0.12 * sampling_rate)
        or np.any(np.diff(indices) > 0.38 * sampling_rate)
    )
    return indices, candidates, fallback


def _gini(values: np.ndarray) -> float:
    values = np.sort(np.maximum(np.asarray(values, dtype=float), 0.0))
    total = values.sum()
    if values.size == 0 or total <= 0:
        return 0.0
    indices = np.arange(1, values.size + 1)
    return float((2.0 * np.sum(indices * values) / (values.size * total)) - (values.size + 1) / values.size)


def landmark_features(envelope: np.ndarray, onsets: np.ndarray, sampling_rate: int) -> dict[str, float]:
    derivative = np.gradient(envelope) * sampling_rate
    rows = []
    median_ioi = int(round(np.median(np.diff(onsets))))
    for index, onset in enumerate(onsets):
        stop = int(onsets[index + 1]) if index + 1 < onsets.size else min(envelope.size, int(onset) + median_ioi)
        segment = envelope[int(onset):stop]
        segment_derivative = derivative[int(onset):stop]
        if segment.size < 3:
            continue
        maximum = float(segment.max())
        max_index = int(np.argmax(segment))
        threshold = 0.8 * maximum
        above = np.flatnonzero(segment >= threshold)
        max_derivative_index = int(np.argmax(segment_derivative))
        rows.append({
            "sharpness": float(np.maximum(segment_derivative, 0.0).mean()),
            "maximum_amplitude": maximum,
            "maximum_amplitude_latency": max_index / sampling_rate,
            "eighty_percent_latency": (int(above[0]) / sampling_rate) if above.size else np.nan,
            "plateau_seconds": ((int(above[-1]) - int(above[0])) / sampling_rate) if above.size else np.nan,
            "maximum_derivative": float(segment_derivative[max_derivative_index]),
            "maximum_derivative_latency": max_derivative_index / sampling_rate,
            "gini": _gini(segment),
        })
    names = tuple(rows[0])
    return {name: float(np.nanmean([row[name] for row in rows])) for name in names}


def analyse_stream(wav_path: Path, stream_id: int, config: ReplicationConfig) -> AcousticResult:
    sampling_rate, raw = wavfile.read(wav_path)
    audio = float_audio(raw)
    envelope = cochlear_envelope(audio, sampling_rate, config.envelope_lowpass_hz)
    peakrate = np.maximum(np.gradient(envelope) * sampling_rate, 0.0)
    onsets, candidates, fallback = select_landmarks(
        peakrate,
        sampling_rate,
        config.expected_landmarks,
        config.candidate_min_distance_ms,
        config.ioi_target_ms,
        config.ioi_soft_scale_ms,
    )
    condition_id, condition = condition_for_stream(stream_id)
    features = landmark_features(envelope, onsets, sampling_rate)
    ioi = np.diff(onsets) / sampling_rate
    cycle = np.arange(onsets.size, dtype=float)
    slope, intercept = np.polyfit(cycle, onsets / sampling_rate, 1)
    features.update({
        "mean_ioi_seconds": float(ioi.mean()),
        "sd_ioi_seconds": float(ioi.std(ddof=1)),
        "fitted_ioi_seconds": float(slope),
        "fitted_frequency_hz": float(1.0 / slope),
        "fit_intercept_seconds": float(intercept),
    })
    return AcousticResult(
        stream_id=stream_id,
        condition_id=condition_id,
        condition=condition,
        sampling_rate=int(sampling_rate),
        duration_seconds=audio.size / sampling_rate,
        onsets_seconds=onsets / sampling_rate,
        candidate_indices=candidates,
        fallback=fallback,
        features=features,
    )
