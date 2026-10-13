from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from eeg_accuracy_study.features import region_indices
from eeg_algorithm_study.algorithms import fit_monotone_logistic
from eeg_advanced_study.algorithms import (
    calibrate_pocock_boundaries, crossing_rate, first_crossings, phase_trajectories,
)
from eeg_core.runtime import atomic_save_npz, atomic_write_json

from ..runtime import RunContext
from .coefficients import feature_path
from .hilbert_condition import FEATURE_KEYS, METHODS, _rotations


def _condition_trajectories(context: RunContext, participant_index: int, coefficients: np.ndarray,
                            regions: tuple[np.ndarray, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Secondary endpoint: natural K=2..10 regional 4-harmonic condition trajectories.

    Not directly comparable to the static single-look K=10 numbers from
    hilbert_condition.py. Pocock familywise correction over multiple looks typically
    lowers detection at K=10, but early crossings can make the cumulative endpoint
    behave differently from a single-look decision.
    """
    config = context.config
    _, positive_regional = phase_trajectories(coefficients, config.k_values, regions)
    null_regional = np.empty(
        (config.streams, config.surrogates, len(config.k_values), len(regions), len(config.harmonics_hz)), np.float32,
    )
    for stream in range(config.streams):
        rotations = _rotations(config, participant_index, stream)
        _, regional_scores = phase_trajectories(coefficients[stream:stream + 1], config.k_values, regions, rotations)
        null_regional[stream] = regional_scores[0]
    conditions, streams_per = config.conditions, config.streams_per_condition
    pos_r = positive_regional.reshape(conditions, streams_per, len(config.k_values), len(regions), -1).mean(axis=1)
    null_r = null_regional.reshape(conditions, streams_per, config.surrogates, len(config.k_values), len(regions), -1).mean(axis=1)
    k_array = np.asarray(config.k_values, dtype=float)
    pos_scaled = 2.0 * k_array[None, :, None, None] * pos_r
    null_scaled = 2.0 * k_array[None, None, :, None, None] * null_r
    return pos_scaled.astype(np.float32), null_scaled.astype(np.float32)


def run(context: RunContext, force: bool = False) -> tuple[list[Path], dict]:
    stage, config = "03_hilbert_sequential", context.config
    lock = json.loads((context.run_dir / "design_lock.json").read_text(encoding="utf-8"))
    participants = lock["participants"]
    regions = region_indices(tuple(lock["channel_labels"]), config.regions)
    checkpoint = context.checkpoint_dir / "03_hilbert_sequential" / "trajectories.npz"
    context.start_stage(stage, len(participants) + 1)

    if context.checkpoint_valid(checkpoint) and not force:
        with np.load(checkpoint, allow_pickle=False) as archive:
            bundle = {method: (archive[f"{method}_pos"], archive[f"{method}_null"]) for method in METHODS}
        context.progress(stage, len(participants), len(participants) + 1, "trajectory checkpoint")
    else:
        per_method: dict[str, tuple[list, list]] = {method: ([], []) for method in METHODS}
        for participant_index, participant in enumerate(participants):
            with np.load(feature_path(context, participant), allow_pickle=False) as archive:
                for method in METHODS:
                    coefficients = archive[FEATURE_KEYS[method]]
                    pos, null = _condition_trajectories(context, participant_index, coefficients, regions)
                    per_method[method][0].append(pos)
                    per_method[method][1].append(null)
            context.progress(stage, participant_index + 1, len(participants) + 1, participant)
        bundle = {method: (np.stack(pos_list), np.stack(null_list)) for method, (pos_list, null_list) in per_method.items()}
        save_kwargs: dict[str, np.ndarray] = {}
        for method, (pos, null) in bundle.items():
            save_kwargs[f"{method}_pos"] = pos
            save_kwargs[f"{method}_null"] = null
        atomic_save_npz(checkpoint, **save_kwargs)
        context.mark_checkpoint(checkpoint, {"unit": "participant_condition_sequential"})

    score_rows, boundary_rows, metric_rows = [], [], []
    for method in METHODS:
        pos, null = bundle[method]  # pos: participants x conditions x K x regions x harmonics
        pos_flat = pos.reshape(*pos.shape[:3], -1)
        null_flat = null.reshape(*null.shape[:4], -1)
        for held, participant in enumerate(participants):
            training = np.arange(len(participants)) != held
            held_pos_scores = np.empty((config.conditions, len(config.k_values)), float)
            held_null_scores = np.empty((config.conditions, config.surrogates, len(config.k_values)), float)
            calibration_scores = np.empty((int(training.sum()) * config.conditions * config.surrogates, len(config.k_values)), float)
            for look in range(len(config.k_values)):
                train_pos = pos_flat[training, :, look].reshape(-1, pos_flat.shape[-1])
                train_null = null_flat[training, :, :, look].reshape(-1, null_flat.shape[-1])
                values = np.concatenate((train_pos, train_null))
                labels = np.concatenate((np.ones(len(train_pos), np.int8), np.zeros(len(train_null), np.int8)))
                model = fit_monotone_logistic(values, labels, config.locked_c, config.maximum_iterations, config.tolerance)
                held_pos_scores[:, look] = model.decision_function(pos_flat[held, :, look])
                held_null_scores[:, :, look] = model.decision_function(
                    null_flat[held, :, :, look].reshape(-1, null_flat.shape[-1])
                ).reshape(config.conditions, config.surrogates)
                calibration_scores[:, look] = model.decision_function(train_null)
            calibrated = calibrate_pocock_boundaries(calibration_scores, config.target_fpr)
            boundaries = calibrated.boundaries
            for look, k in enumerate(config.k_values):
                boundary_rows.append({"participant": participant, "method": method, "look_k": k,
                                      "boundary": float(boundaries[look]),
                                      "marginal_tail_probability": calibrated.marginal_tail_probability,
                                      "training_familywise_fpr": calibrated.familywise_fpr})
            first_positive = first_crossings(held_pos_scores, boundaries, config.k_values)
            held_null_flat = held_null_scores.reshape(-1, len(config.k_values))
            metric_rows.append({"participant": participant, "method": method,
                                "sensitivity": float(np.mean(first_positive <= config.k_values[-1])),
                                "familywise_fpr": crossing_rate(held_null_flat, boundaries),
                                "restricted_mean_stopping_trials": float(np.mean(first_positive))})
            crossed = np.maximum.accumulate(held_pos_scores > boundaries[None, :], axis=1)
            for condition in range(config.conditions):
                row = {"participant": participant, "condition": condition + 1, "method": method,
                       "first_detection_k": int(first_positive[condition]),
                       "detected": bool(first_positive[condition] <= config.k_values[-1])}
                row.update({f"score_k{k}": float(held_pos_scores[condition, look]) for look, k in enumerate(config.k_values)})
                row.update({f"detected_by_k{k}": bool(crossed[condition, look]) for look, k in enumerate(config.k_values)})
                score_rows.append(row)

    outputs = [checkpoint]
    for name, frame in (("hilbert_sequential_scores.csv", pd.DataFrame(score_rows)),
                        ("hilbert_sequential_boundaries.csv", pd.DataFrame(boundary_rows)),
                        ("hilbert_sequential_metrics.csv", pd.DataFrame(metric_rows))):
        path = context.table_dir / name; frame.to_csv(path, index=False); outputs.append(path)
    summary = {"methods": list(METHODS), "endpoint": "secondary_sequential",
               "note": "not comparable to the static K=10 C4/H1/H2/H3 numbers"}
    report = context.run_dir / "hilbert_sequential.json"
    atomic_write_json(report, summary); outputs.append(report)
    context.progress(stage, len(participants) + 1, len(participants) + 1, "sequential benchmark")
    context.finish_stage(stage, outputs, summary)
    return outputs, summary
