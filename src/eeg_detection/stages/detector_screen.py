from __future__ import annotations

import time
import tracemalloc

import numpy as np
import pandas as pd
from scipy.io import wavfile

from ..detectors import deterministic_subsets, detector_score
from ..evaluation import auc_from_scores, binomial_interval, empirical_threshold
from ..runtime import RunContext, atomic_save_npz, atomic_write_json


STORED_DETECTORS = ("D1", "D2", "D3", "D4", "D5", "D6", "D8")


def _acoustic_phases(context: RunContext) -> np.ndarray:
    result = np.empty((45, 4), dtype=float)
    for stream in range(1, 46):
        rate, audio = wavfile.read(context.config.data_root / "Stimuli" / f"{stream}.wav")
        values = np.asarray(audio, dtype=float)
        if values.ndim > 1: values = values.mean(axis=1)
        start = int(round(0.5 * rate)); values = values[start:]
        time_axis = np.arange(values.size) / rate
        for harmonic, frequency in enumerate(context.config.harmonics_hz):
            result[stream - 1, harmonic] = np.angle(np.sum(values * np.hanning(values.size) * np.exp(-2j * np.pi * frequency * time_axis)))
    return result


def _score_one(detector: str, coefficients: np.ndarray, expected_phase=0.0) -> float:
    target = coefficients if detector == "D8" else coefficients[..., :1]
    expected = expected_phase if detector == "D4" else 0.0
    return detector_score(detector, target, expected_phase=expected)


def _vectorized_scores(detector: str, values: np.ndarray, expected_phase=0.0) -> np.ndarray:
    """Score subset x trial x channel x harmonic [x null-bin] arrays."""
    coefficients = np.asarray(values)
    k = coefficients.shape[1]
    target = coefficients if detector == "D8" else coefficients[:, :, :, :1, ...]
    magnitude = np.abs(target)
    unit = np.divide(target, magnitude, out=np.zeros_like(target), where=magnitude > 0)
    r2 = np.abs(unit.mean(axis=1)) ** 2  # subset x channel x harmonic [x null]
    if detector == "D1": statistic = r2
    elif detector == "D2": statistic = (k * r2 - 1.0) / (k - 1.0)
    elif detector == "D3": statistic = k * r2
    elif detector == "D4":
        expected = np.asarray(expected_phase)
        shape = (1, 1, 1, expected.size) + (1,) * (target.ndim - 4)
        statistic = np.cos(np.angle(target) - expected.reshape(shape)).mean(axis=1)
    elif detector == "D5":
        real, imaginary = target.real, target.imag
        mean_real, mean_imag = real.mean(axis=1), imaginary.mean(axis=1)
        centered_real = real - mean_real[:, None]
        centered_imag = imaginary - mean_imag[:, None]
        var_real = np.sum(centered_real ** 2, axis=1) / (k - 1)
        var_imag = np.sum(centered_imag ** 2, axis=1) / (k - 1)
        covariance = np.sum(centered_real * centered_imag, axis=1) / (k - 1)
        ridge = 1e-8 * np.maximum((var_real + var_imag) / 2.0, 1.0)
        var_real = var_real + ridge; var_imag = var_imag + ridge
        determinant = var_real * var_imag - covariance ** 2
        statistic = k * np.divide(
            var_imag * mean_real ** 2 - 2 * covariance * mean_real * mean_imag
            + var_real * mean_imag ** 2,
            determinant, out=np.zeros_like(determinant), where=determinant > 0,
        )
    elif detector == "D6":
        numerator = np.abs(np.sum(target, axis=1)) ** 2
        denominator = k * np.sum(np.abs(target) ** 2, axis=1)
        statistic = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    elif detector == "D8":
        return np.sum(2 * k * np.mean(r2, axis=1), axis=1)
    else: raise ValueError(f"Stored vectorization does not support {detector}")
    return np.mean(statistic, axis=(1, 2))


def _participant_scores(
    context: RunContext, participant: int, detector: str, k: int,
    expected_phase_by_stream: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cache = np.load(context.cache_dir / "stored_matlab" / f"participant_{participant:02d}.npz")
    signal, null = cache["signal"], cache["null"]
    subsets = deterministic_subsets(10, k, context.config.subset_cap, context.config.seed)
    positive = np.empty((45, len(subsets)), dtype=float)
    negative = np.empty((45, len(subsets), 18), dtype=float)
    for stream in range(45):
        expected = expected_phase_by_stream[stream, :1]
        positive[stream] = _vectorized_scores(detector, signal[stream][subsets], expected)
        negative[stream] = _vectorized_scores(detector, null[stream][subsets], expected)
    return positive, negative


def _participant_harmonic_scores(context: RunContext, participant: int, harmonic: int, k: int):
    cache = np.load(context.cache_dir / "stored_matlab" / f"participant_{participant:02d}.npz")
    signal, null = cache["signal"], cache["null"]
    subsets = deterministic_subsets(10, k, context.config.subset_cap, context.config.seed)
    positive = np.empty((45, len(subsets)))
    negative = np.empty((45, len(subsets), 18))
    for stream in range(45):
        positive[stream] = _vectorized_scores(
            "D1", signal[stream][subsets][..., harmonic:harmonic + 1]
        )
        negative[stream] = _vectorized_scores(
            "D1", null[stream][subsets][..., harmonic:harmonic + 1, :]
        )
    return positive, negative


def _training_phase_direction(context: RunContext, heldout: int, acoustic_phase: np.ndarray) -> np.ndarray:
    """Fold-specific phase target; held-out participant never enters the delay estimate."""
    phase_referenced = []
    for participant in range(25):
        if participant == heldout: continue
        signal = np.load(context.cache_dir / "stored_matlab" / f"participant_{participant:02d}.npz")["signal"]
        units = signal / np.maximum(np.abs(signal), np.finfo(float).tiny)
        phase_referenced.append(units * np.exp(-1j * acoustic_phase[:, None, None, :]))
    mean_by_harmonic = np.mean(np.concatenate(phase_referenced, axis=1), axis=(0, 1, 2))
    phase_difference = np.unwrap(np.angle(mean_by_harmonic))
    frequencies = np.asarray(context.config.harmonics_hz)
    delay = -np.polyfit(2 * np.pi * frequencies, phase_difference, 1)[0]
    return acoustic_phase - 2 * np.pi * frequencies[None, :] * delay


def run(context: RunContext, force: bool = False) -> tuple[list, dict]:
    stage = "03_detector_screen"
    config = context.config
    total = 25 * (len(STORED_DETECTORS) + len(config.harmonics_hz)) * len(config.k_values)
    context.start_stage(stage, total)
    acoustic_phase = _acoustic_phases(context)
    completed = 0
    outputs = []
    tracemalloc.start(); started = time.perf_counter()
    for participant in range(25):
        expected_phase = _training_phase_direction(context, participant, acoustic_phase)
        for detector in STORED_DETECTORS:
            for k in config.k_values:
                path = context.checkpoint_dir / stage / f"p{participant:02d}_{detector}_k{k:02d}.npz"
                if not context.checkpoint_valid(path) or force:
                    positive, negative = _participant_scores(context, participant, detector, k, expected_phase)
                    atomic_save_npz(path, positive=positive, negative=negative)
                    context.mark_checkpoint(path, {"branch": "stored_matlab", "fold": participant,
                                                   "detector": detector, "k": k})
                outputs.append(path); completed += 1
                context.progress(stage, completed, total, f"fold P{participant + 1} {detector} K={k}")

        for harmonic in range(len(config.harmonics_hz)):
            for k in config.k_values:
                path = context.checkpoint_dir / stage / f"p{participant:02d}_harmonic{harmonic}_k{k:02d}.npz"
                if not context.checkpoint_valid(path) or force:
                    positive, negative = _participant_harmonic_scores(context, participant, harmonic, k)
                    atomic_save_npz(path, positive=positive, negative=negative)
                    context.mark_checkpoint(path, {"branch": "stored_matlab", "fold": participant,
                                                   "harmonic_hz": config.harmonics_hz[harmonic], "k": k})
                outputs.append(path); completed += 1
                context.progress(stage, completed, total,
                                 f"fold P{participant + 1} {config.harmonics_hz[harmonic]:g} Hz K={k}")

    score_rows, calibration_rows = [], []
    for heldout in range(25):
        for detector in STORED_DETECTORS:
            for k in config.k_values:
                held = np.load(context.checkpoint_dir / stage / f"p{heldout:02d}_{detector}_k{k:02d}.npz")
                training_null = np.concatenate([
                    np.load(context.checkpoint_dir / stage / f"p{participant:02d}_{detector}_k{k:02d}.npz")["negative"].reshape(-1)
                    for participant in range(25) if participant != heldout
                ])
                threshold = empirical_threshold(training_null, config.target_fpr)
                held_null = held["negative"].reshape(-1)
                false_positives = int(np.count_nonzero(held_null > threshold))
                fpr = false_positives / held_null.size
                ci_low, ci_high = binomial_interval(false_positives, held_null.size)
                calibration_rows.append({"branch": "stored_matlab", "fold_participant": heldout,
                    "estimator": "F1", "detector": detector, "spatial": "S0", "k": k,
                    "threshold": threshold, "heldout_fpr": fpr, "ci_low": ci_low, "ci_high": ci_high,
                    "n_train_null": training_null.size, "n_heldout_null": held_null.size})
                for stream in range(45):
                    condition = stream // 3
                    stream_scores = held["positive"][stream]
                    score_rows.append({"branch": "stored_matlab", "participant": heldout,
                        "condition": condition, "stream": stream, "estimator": "F1", "detector": detector,
                        "spatial": "S0", "k": k, "mean_score": float(np.mean(stream_scores)),
                        "detection_rate": float(np.mean(stream_scores > threshold)),
                        "auc": auc_from_scores(stream_scores, held_null), "threshold": threshold})
    scores_path = context.table_dir / "detector_scores_long.csv"
    calibration_path = context.table_dir / "detector_calibration_long.csv"
    pd.DataFrame(score_rows).to_csv(scores_path, index=False)
    pd.DataFrame(calibration_rows).to_csv(calibration_path, index=False)
    harmonic_rows = []
    for heldout in range(25):
        for harmonic, frequency in enumerate(config.harmonics_hz):
            for k in config.k_values:
                held = np.load(context.checkpoint_dir / stage /
                               f"p{heldout:02d}_harmonic{harmonic}_k{k:02d}.npz")
                training_null = np.concatenate([
                    np.load(context.checkpoint_dir / stage /
                            f"p{participant:02d}_harmonic{harmonic}_k{k:02d}.npz")["negative"].reshape(-1)
                    for participant in range(25) if participant != heldout
                ])
                threshold = empirical_threshold(training_null, config.target_fpr)
                for stream in range(45):
                    scores = held["positive"][stream]
                    harmonic_rows.append({"branch": "stored_matlab", "participant": heldout,
                        "condition": stream // 3, "stream": stream, "harmonic_hz": frequency, "k": k,
                        "threshold": threshold, "mean_score": float(np.mean(scores)),
                        "detection_rate": float(np.mean(scores > threshold))})
    harmonic_path = context.table_dir / "detection_by_harmonic.csv"
    pd.DataFrame(harmonic_rows).to_csv(harmonic_path, index=False)
    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    report = {"branch": "stored_matlab", "folds": 25, "detectors": list(STORED_DETECTORS),
              "k_values": list(config.k_values), "threshold_training": "leave-one-participant-out",
              "calibration_bins": 18, "runtime_seconds": time.perf_counter() - started,
              "peak_python_memory_bytes": peak}
    report_path = context.run_dir / "stored_detector_screen.json"
    atomic_write_json(report_path, report)
    outputs.extend([scores_path, calibration_path, harmonic_path, report_path])
    context.finish_stage(stage, outputs, report)
    return outputs, report
