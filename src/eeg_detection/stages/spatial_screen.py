from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import time
import tracemalloc

import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import wavfile

from eeg_replication.io import inspect_set, load_stream_trials, natural_subject_key

from ..detectors import compatible, deterministic_subsets
from ..evaluation import auc_from_scores, empirical_threshold
from ..spatial import SpatialModel, best_channel, fit_dss, fit_regularized_cca, fit_ress, lagged_stimulus
from ..runtime import RunContext, atomic_save_npz, atomic_write_json
from .detector_screen import _acoustic_phases, _vectorized_scores


def _condition_streams(condition: int) -> range:
    return range(3 * condition + 1, 3 * condition + 4)


def _stimulus_features(context: RunContext, stream: int, n_times: int, decimation: int) -> np.ndarray:
    rate, audio = wavfile.read(context.config.data_root / "Stimuli" / f"{stream}.wav")
    values = np.asarray(audio, dtype=float)
    if values.ndim > 1: values = values.mean(axis=1)
    values = signal.resample(values, n_times * decimation)[::decimation]
    envelope = np.abs(signal.hilbert(values))
    derivative = np.maximum(np.gradient(envelope), 0.0)
    features = []
    sampling_rate = context.config.sampling_rate // decimation
    lags = tuple(int(round(value * sampling_rate / 1000)) for value in context.config.cca_lags_ms)
    for feature in (envelope, derivative):
        standardized = (feature - feature.mean()) / max(feature.std(), np.finfo(float).eps)
        features.append(lagged_stimulus(standardized, lags))
    return np.column_stack(features)


def _training_arrays(context: RunContext, epoch, conditions: list[int], decimation: int = 10):
    eeg_trials, designs = [], []
    for condition in conditions:
        for stream in _condition_streams(condition):
            trials = load_stream_trials(epoch, stream, dtype=np.float32)[..., ::decimation]
            design = _stimulus_features(context, stream, trials.shape[-1], decimation)
            eeg_trials.append(trials)
            designs.append(np.tile(design, (trials.shape[0], 1)))
    eeg = np.concatenate(eeg_trials, axis=0)
    observations = np.moveaxis(eeg, 1, -1).reshape(-1, eeg.shape[1])
    stimulus = np.concatenate(designs, axis=0)
    return eeg, observations, stimulus


def _best_channel_model(context: RunContext, participant: int, conditions: list[int]) -> SpatialModel:
    scores = []
    for condition in conditions:
        for stream in _condition_streams(condition):
            coefficients = np.load(context.cache_dir / "public_set_fdt" /
                                   f"participant_{participant:02d}" / f"stream_{stream:02d}.npz")["F1"]
            unit = coefficients[..., 0] / np.maximum(np.abs(coefficients[..., 0]), np.finfo(float).tiny)
            scores.append(np.abs(unit.mean(axis=0)) ** 2)
    return best_channel(np.asarray(scores))


def _oracle_channel_model(context: RunContext, participant: int, condition: int) -> SpatialModel:
    """Held-out best channel, retained only as a descriptive upper ceiling."""
    model = _best_channel_model(context, participant, [condition])
    return SpatialModel("S1_oracle_ceiling", model.weights,
                        {**model.metadata, "oracle": True, "excluded_from_inference": True})


def _fit_models(context: RunContext, participant: int, epoch, conditions: list[int]) -> dict[str, SpatialModel | None]:
    eeg, observations, stimulus = _training_arrays(context, epoch, conditions)
    rate = context.config.sampling_rate // 10
    return {
        "S0": None,
        "S1": _best_channel_model(context, participant, conditions),
        "S2": fit_ress(eeg, 4.0, rate, context.config.spatial_regularization),
        "S3": fit_dss(eeg, context.config.spatial_regularization),
        "S4": fit_regularized_cca(observations, stimulus, context.config.spatial_regularization),
    }


def _project(coefficients: np.ndarray, model: SpatialModel | None) -> tuple[np.ndarray, np.ndarray | None]:
    values = np.asarray(coefficients)
    is_multitaper = values.ndim == 4
    if model is None:
        return (values.mean(axis=-1) if is_multitaper else values), (values if is_multitaper else None)
    if is_multitaper:
        projected = np.tensordot(values, model.weights, axes=(1, 0))  # trial x harmonic x taper
        return projected.mean(axis=-1)[:, None, :], projected[:, None, :, :]
    projected = np.tensordot(values, model.weights, axes=(1, 0))
    return projected[:, None, :], None


def _fit_phase_delays(
    context: RunContext, participant: int, conditions: list[int],
    models: dict[str, SpatialModel | None], acoustic_phase: np.ndarray,
) -> dict[tuple[str, str], float]:
    """Fit estimator/spatial neural delays using training conditions only."""
    frequencies = np.asarray(context.config.harmonics_hz)
    referenced: dict[tuple[str, str], list[np.ndarray]] = {
        (estimator, spatial_method): []
        for estimator in context.config.estimators
        for spatial_method in context.config.spatial_methods
    }
    for condition in conditions:
        for stream in _condition_streams(condition):
            archive = np.load(context.cache_dir / "public_set_fdt" /
                              f"participant_{participant:02d}" / f"stream_{stream:02d}.npz")
            for estimator in context.config.estimators:
                for spatial_method in context.config.spatial_methods:
                    coefficients, _ = _project(archive[estimator], models[spatial_method])
                    unit = coefficients / np.maximum(np.abs(coefficients), np.finfo(float).tiny)
                    mean = unit.mean(axis=tuple(range(unit.ndim - 1)))
                    referenced[(estimator, spatial_method)].append(
                        mean * np.exp(-1j * acoustic_phase[stream - 1])
                    )
    delays = {}
    for key, values in referenced.items():
        phase_difference = np.unwrap(np.angle(np.mean(values, axis=0)))
        delays[key] = float(-np.polyfit(2 * np.pi * frequencies, phase_difference, 1)[0])
    return delays


def _score_subsets(
    coefficients: np.ndarray, taper_coefficients: np.ndarray | None, detector: str,
    subsets: np.ndarray, phase_randomizations: int, rng: np.random.Generator,
    expected_phase: np.ndarray | float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    def score(values, taper_values):
        if detector != "D7":
            return _vectorized_scores(detector, values, expected_phase)
        line = taper_values.mean(axis=(1, -1))
        residual = taper_values - line[:, None, ..., None]
        numerator = taper_values.shape[1] * taper_values.shape[-1] * np.abs(line) ** 2
        denominator = np.mean(np.abs(residual) ** 2, axis=(1, -1))
        statistic = np.divide(numerator, denominator, out=np.zeros_like(numerator),
                              where=denominator > 0)
        return statistic.mean(axis=(1, 2))

    selected = coefficients[subsets]
    selected_tapers = None if taper_coefficients is None else taper_coefficients[subsets]
    target = selected if detector == "D8" else selected[..., :1]
    taper_target = None if selected_tapers is None else selected_tapers[..., :1, :]
    positive = score(target, taper_target)
    null = np.empty((len(subsets), phase_randomizations))
    for batch in range(phase_randomizations):
        rotations = np.exp(1j * rng.uniform(
            -np.pi, np.pi, size=(len(subsets), target.shape[1]) + (1,) * (target.ndim - 2)
        ))
        randomized = target * rotations
        randomized_tapers = None if taper_target is None else taper_target * rotations[..., None]
        null[:, batch] = score(randomized, randomized_tapers)
    return positive, null


def _combination_keys(context: RunContext):
    for estimator in context.config.estimators:
        base = estimator.split("_")[0]
        for detector in context.config.detectors:
            if not compatible(base, detector): continue
            for spatial in context.config.spatial_methods:
                for k in context.config.k_values:
                    yield estimator, detector, spatial, k


def run(context: RunContext, force: bool = False) -> tuple[list, dict]:
    stage = "05_spatial_screen"
    config = context.config
    set_paths = sorted((config.data_root / "Pre-processed EEGLAB datasets").glob("*.set"), key=natural_subject_key)
    epochs = [inspect_set(path) for path in set_paths]
    total = len(epochs) * 15
    context.start_stage(stage, total)
    acoustic_phase = _acoustic_phases(context)
    outputs: list[Path] = []; completed = 0; started = time.perf_counter(); tracemalloc.start()
    for participant, epoch in enumerate(epochs):
        for outer_condition in range(15):
            checkpoint = context.checkpoint_dir / stage / f"participant_{participant:02d}_condition_{outer_condition:02d}.csv"
            weights_path = checkpoint.with_suffix(".weights.npz")
            if not context.checkpoint_valid(checkpoint) or force:
                training_conditions = [value for value in range(15) if value != outer_condition]
                calibration: dict[tuple, list[np.ndarray]] = defaultdict(list)
                # Inner LOCO: each calibration condition is absent from its filter fit.
                for inner_condition in training_conditions:
                    inner_train = [value for value in training_conditions if value != inner_condition]
                    models = _fit_models(context, participant, epoch, inner_train)
                    phase_delays = _fit_phase_delays(
                        context, participant, inner_train, models, acoustic_phase
                    )
                    for stream in _condition_streams(inner_condition):
                        archive = np.load(context.cache_dir / "public_set_fdt" /
                                          f"participant_{participant:02d}" / f"stream_{stream:02d}.npz")
                        for estimator, detector, spatial_method, k in _combination_keys(context):
                            coefficients, tapers = _project(archive[estimator], models[spatial_method])
                            subsets = deterministic_subsets(10, k, config.subset_cap, config.seed)
                            rng = np.random.default_rng(np.random.SeedSequence([
                                config.seed, participant, outer_condition, inner_condition, stream, k,
                                config.estimators.index(estimator), config.detectors.index(detector),
                                config.spatial_methods.index(spatial_method),
                            ]))
                            expected_phase = 0.0
                            if detector == "D4":
                                delay = phase_delays[(estimator, spatial_method)]
                                expected_phase = acoustic_phase[stream - 1, :1] - 2 * np.pi * 4.0 * delay
                            _, null = _score_subsets(coefficients, tapers, detector, subsets, 2, rng,
                                                    expected_phase)
                            calibration[(estimator, detector, spatial_method, k)].append(null.reshape(-1))
                final_models = _fit_models(context, participant, epoch, training_conditions)
                final_models["S1_oracle_ceiling"] = _oracle_channel_model(
                    context, participant, outer_condition
                )
                final_phase_delays = _fit_phase_delays(
                    context, participant, training_conditions, final_models, acoustic_phase
                )
                atomic_save_npz(weights_path, **{
                    name: (np.full(32, 1 / 32) if model is None else model.weights)
                    for name, model in final_models.items()
                })
                rows = []
                for stream in _condition_streams(outer_condition):
                    archive = np.load(context.cache_dir / "public_set_fdt" /
                                      f"participant_{participant:02d}" / f"stream_{stream:02d}.npz")
                    for estimator, detector, spatial_method, k in _combination_keys(context):
                        key = (estimator, detector, spatial_method, k)
                        null_scores = np.concatenate(calibration[key])
                        threshold = empirical_threshold(null_scores, config.target_fpr)
                        coefficients, tapers = _project(archive[estimator], final_models[spatial_method])
                        subsets = deterministic_subsets(10, k, config.subset_cap, config.seed)
                        rng = np.random.default_rng(config.seed)
                        expected_phase = 0.0
                        if detector == "D4":
                            delay = final_phase_delays[(estimator, spatial_method)]
                            expected_phase = acoustic_phase[stream - 1, :1] - 2 * np.pi * 4.0 * delay
                        positive, held_null = _score_subsets(
                            coefficients, tapers, detector, subsets, 18, rng, expected_phase
                        )
                        rows.append({"branch": "public_set_fdt", "participant": participant,
                            "outer_condition": outer_condition, "stream": stream - 1,
                            "estimator": estimator, "detector": detector, "spatial": spatial_method,
                            "k": k, "threshold": threshold,
                            "detection_rate": float(np.mean(positive > threshold)),
                            "heldout_fpr": float(np.mean(held_null > threshold)),
                            "auc": auc_from_scores(positive, held_null.reshape(-1)),
                            "mean_score": float(np.mean(positive)),
                            "calibration_observations": len(null_scores),
                            "inner_cross_fitted": True})
                    # The held-out best channel is intentionally not part of the method screen.
                    # It uses the S1 training calibration only and is labelled as a ceiling.
                    oracle_coefficients, _ = _project(archive["F1"], final_models["S1_oracle_ceiling"])
                    for k in config.k_values:
                        calibration_key = ("F1", "D1", "S1", k)
                        null_scores = np.concatenate(calibration[calibration_key])
                        threshold = empirical_threshold(null_scores, config.target_fpr)
                        subsets = deterministic_subsets(10, k, config.subset_cap, config.seed)
                        positive, held_null = _score_subsets(
                            oracle_coefficients, None, "D1", subsets, 18,
                            np.random.default_rng(np.random.SeedSequence([
                                config.seed, participant, outer_condition, stream, k, 999,
                            ])),
                        )
                        rows.append({"branch": "public_set_fdt", "participant": participant,
                            "outer_condition": outer_condition, "stream": stream - 1,
                            "estimator": "F1", "detector": "D1", "spatial": "S1_oracle_ceiling",
                            "k": k, "threshold": threshold,
                            "detection_rate": float(np.mean(positive > threshold)),
                            "heldout_fpr": float(np.mean(held_null > threshold)),
                            "auc": auc_from_scores(positive, held_null.reshape(-1)),
                            "mean_score": float(np.mean(positive)),
                            "calibration_observations": len(null_scores),
                            "inner_cross_fitted": False, "oracle_ceiling": True,
                            "excluded_from_inference": True})
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_csv(checkpoint, index=False)
                context.mark_checkpoint(checkpoint, {"branch": "public_set_fdt", "participant": participant,
                                                     "outer_condition": outer_condition})
                context.mark_checkpoint(weights_path, {"branch": "public_set_fdt", "participant": participant,
                                                       "outer_condition": outer_condition})
            outputs.extend([checkpoint, weights_path]); completed += 1
            context.progress(stage, completed, total, f"P{participant + 1} held condition {outer_condition + 1}")
    combined_path = context.table_dir / "public_spatial_scores_long.csv"
    csv_outputs = [path for path in outputs if path.suffix == ".csv"]
    pd.concat((pd.read_csv(path) for path in csv_outputs), ignore_index=True).to_csv(combined_path, index=False)
    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    report = {"branch": "public_set_fdt", "outer_folds": total,
              "inner_calibration": "leave-one-condition-out within the other 14 conditions",
              "spatial_methods": list(config.spatial_methods), "runtime_seconds": time.perf_counter() - started,
              "peak_python_memory_bytes": peak,
              "oracle_channel": "S1_oracle_ceiling is descriptive only and excluded from inference",
              "cross_source_numerical_merge": False}
    report_path = context.run_dir / "public_spatial_screen.json"
    atomic_write_json(report_path, report); outputs.extend([combined_path, report_path])
    context.finish_stage(stage, outputs, report)
    return outputs, report
