from __future__ import annotations

import numpy as np

from eeg_replication.io import inspect_set, load_stream_trials, natural_subject_key

from .detectors import compatible, detector_score
from .estimators import (
    dpss_multitaper_coefficients,
    hilbert_coefficients,
    least_squares_coefficients,
    published_hann_dft,
)
from .runtime import RunContext, atomic_write_json
from .spatial import best_channel, fit_dss, fit_regularized_cca, fit_ress
from .stages.spatial_screen import _stimulus_features


def run_smoke(context: RunContext) -> dict:
    """Exercise every method family on one participant and one stream."""
    config = context.config
    set_path = sorted((config.data_root / "Pre-processed EEGLAB datasets").glob("*.set"),
                      key=natural_subject_key)[0]
    epoch = inspect_set(set_path)
    trials = load_stream_trials(epoch, 1)
    frequencies = np.asarray(config.harmonics_hz)
    estimates = {
        "F1": published_hann_dft(trials, config.sampling_rate, frequencies, config.fft_crop_samples),
        "F2": hilbert_coefficients(trials, config.sampling_rate, frequencies,
                                    config.hilbert_order, config.hilbert_half_bandwidth_hz,
                                    config.hilbert_edge_crop_seconds),
        "F3": dpss_multitaper_coefficients(trials, config.sampling_rate, frequencies,
                                            config.fft_crop_samples, config.dpss_time_bandwidth,
                                            config.dpss_tapers),
        "F4_nominal": least_squares_coefficients(trials, config.sampling_rate, frequencies,
                                                  config.fft_crop_samples),
        "F4_measured": least_squares_coefficients(trials, config.sampling_rate, frequencies + 0.01 * np.arange(1, 5),
                                                   config.fft_crop_samples),
    }
    decimated = trials[..., ::10]
    observations = np.moveaxis(decimated, 1, -1).reshape(-1, decimated.shape[1])
    design = np.tile(_stimulus_features(context, 1, decimated.shape[-1], 10), (10, 1))
    channel_phase = estimates["F1"][..., 0]
    unit = channel_phase / np.maximum(np.abs(channel_phase), np.finfo(float).tiny)
    models = {
        "S1": best_channel((np.abs(unit.mean(axis=0)) ** 2)[None, :]),
        "S2": fit_ress(decimated, 4.0, config.sampling_rate // 10, config.spatial_regularization),
        "S3": fit_dss(decimated, config.spatial_regularization),
        "S4": fit_regularized_cca(observations, design, config.spatial_regularization),
    }
    scores = {}
    for estimator, values in estimates.items():
        base = estimator.split("_")[0]
        ordinary = values.mean(axis=-1) if estimator == "F3" else values
        for detector in config.detectors:
            if not compatible(base, detector): continue
            target = ordinary if detector == "D8" else ordinary[..., :1]
            taper_target = values[..., :1, :] if detector == "D7" else None
            scores[f"{estimator}+{detector}+S0"] = detector_score(
                detector, target, taper_coefficients=taper_target
            )
    if not all(np.isfinite(list(scores.values()))):
        raise AssertionError("A detector returned a non-finite smoke-test score")
    report = {
        "participant": epoch.participant, "stream": 1,
        "estimator_shapes": {name: list(value.shape) for name, value in estimates.items()},
        "detector_combinations_exercised": len(scores),
        "spatial_weight_shapes": {name: list(model.weights.shape) for name, model in models.items()},
        "passed": True,
    }
    path = context.run_dir / "one_participant_one_stream_smoke.json"
    atomic_write_json(path, report)
    print(f"Smoke test passed: {path}")
    return report
