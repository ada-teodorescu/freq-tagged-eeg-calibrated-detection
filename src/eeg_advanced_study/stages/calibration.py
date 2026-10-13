from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from eeg_core.runtime import atomic_write_json

from ..algorithms import (
    blend_boundaries, calibrate_pocock_boundaries, crossing_rate, first_crossings,
    fixed_stream_scores, phase_trajectories, select_shrinkage_weight,
)
from ..runtime import RunContext
from .features import feature_path
from .stream import METHODS, score_path


def _score_rows(participant: str, duration: float, method: str, calibration: str,
                scores: np.ndarray, boundaries: np.ndarray, k_values: tuple[int, ...],
                condition_offset: int = 0) -> list[dict]:
    crossed = np.maximum.accumulate(scores > boundaries[None, :], axis=1)
    first = first_crossings(scores, boundaries, k_values)
    rows = []
    for stream in range(scores.shape[0]):
        row = {
            "participant": participant, "condition": condition_offset + stream // 3 + 1,
            "stream_within_condition": stream % 3 + 1, "duration_seconds": duration,
            "method": method, "calibration": calibration, "first_detection_k": int(first[stream]),
            "detected": bool(first[stream] <= k_values[-1]),
        }
        row.update({f"score_k{k}": float(scores[stream, look]) for look, k in enumerate(k_values)})
        row.update({f"detected_by_k{k}": bool(crossed[stream, look]) for look, k in enumerate(k_values)})
        rows.append(row)
    return rows


def _population_analysis(context: RunContext, duration: float, archive,
                         score_rows: list[dict], boundary_rows: list[dict], metric_rows: list[dict]):
    config = context.config
    positive, null, calibration_null = archive["positive"], archive["null"], archive["calibration_null"]
    participants = archive["participants"].astype(str).tolist()
    boundaries = np.empty((len(participants), len(METHODS), len(config.k_values)), float)
    for held, participant in enumerate(participants):
        for method_index, method in enumerate(METHODS):
            calibrated = calibrate_pocock_boundaries(calibration_null[held, :, :, method_index], config.target_fpr)
            limits = calibrated.boundaries; boundaries[held, method_index] = limits
            for look, k in enumerate(config.k_values):
                boundary_rows.append({
                    "participant": participant, "condition": 0, "duration_seconds": duration,
                    "method": method, "calibration": "population", "look_k": k,
                    "boundary": float(limits[look]), "marginal_tail_probability": calibrated.marginal_tail_probability,
                    "training_familywise_fpr": calibrated.familywise_fpr, "shrinkage_weight": 0.0,
                })
            score_rows.extend(_score_rows(participant, duration, method, "population",
                                          positive[held, :, :, method_index], limits, config.k_values))
            held_null = null[held, :, :, :, method_index].reshape(-1, len(config.k_values))
            first = first_crossings(positive[held, :, :, method_index], limits, config.k_values)
            metric_rows.append({
                "participant": participant, "condition": 0, "duration_seconds": duration,
                "method": method, "calibration": "population",
                "sensitivity": float(np.mean(first <= config.k_values[-1])),
                "familywise_fpr": crossing_rate(held_null, limits),
                "restricted_mean_stopping_trials": float(np.mean(first)), "shrinkage_weight": 0.0,
            })
    return boundaries


def _shrinkage_analysis(context: RunContext, archive, population_boundaries: np.ndarray,
                        score_rows: list[dict], boundary_rows: list[dict], metric_rows: list[dict]):
    config = context.config
    positive, null = archive["positive"], archive["null"]
    participants = archive["participants"].astype(str).tolist()
    for held, participant in enumerate(participants):
        for method_index, method in enumerate(METHODS):
            pop = population_boundaries[held, method_index]
            for evaluated_condition in range(config.conditions):
                calibration_conditions = [c for c in range(config.conditions) if c != evaluated_condition]
                local_trajectories = np.concatenate([
                    null[held, c * 3:(c + 1) * 3, :, :, method_index].reshape(-1, len(config.k_values))
                    for c in calibration_conditions
                ])
                local = calibrate_pocock_boundaries(local_trajectories, config.target_fpr).boundaries
                inner_boundaries, inner_validation = [], []
                for validation_condition in calibration_conditions:
                    inner_training = [c for c in calibration_conditions if c != validation_condition]
                    training_paths = np.concatenate([
                        null[held, c * 3:(c + 1) * 3, :, :, method_index].reshape(-1, len(config.k_values))
                        for c in inner_training
                    ])
                    inner_boundaries.append(calibrate_pocock_boundaries(training_paths, config.target_fpr).boundaries)
                    inner_validation.append(
                        null[held, validation_condition * 3:(validation_condition + 1) * 3, :, :, method_index]
                        .reshape(-1, len(config.k_values))
                    )
                weight, errors = select_shrinkage_weight(
                    pop, np.asarray(inner_boundaries), np.asarray(inner_validation), config.shrinkage_weights,
                    config.target_fpr,
                )
                limits = blend_boundaries(pop, local, weight)
                for look, k in enumerate(config.k_values):
                    boundary_rows.append({
                        "participant": participant, "condition": evaluated_condition + 1,
                        "duration_seconds": config.primary_duration_seconds, "method": method,
                        "calibration": "shrinkage", "look_k": k, "boundary": float(limits[look]),
                        "marginal_tail_probability": np.nan, "training_familywise_fpr": np.nan,
                        "shrinkage_weight": weight, "selection_rmse": errors[weight],
                    })
                stream_slice = slice(evaluated_condition * 3, (evaluated_condition + 1) * 3)
                evaluated_scores = positive[held, stream_slice, :, method_index]
                score_rows.extend(_score_rows(
                    participant, config.primary_duration_seconds, method, "shrinkage", evaluated_scores,
                    limits, config.k_values, condition_offset=evaluated_condition,
                ))
                evaluated_null = null[held, stream_slice, :, :, method_index].reshape(-1, len(config.k_values))
                first = first_crossings(evaluated_scores, limits, config.k_values)
                metric_rows.append({
                    "participant": participant, "condition": evaluated_condition + 1,
                    "duration_seconds": config.primary_duration_seconds, "method": method,
                    "calibration": "shrinkage", "sensitivity": float(np.mean(first <= config.k_values[-1])),
                    "familywise_fpr": crossing_rate(evaluated_null, limits),
                    "restricted_mean_stopping_trials": float(np.mean(first)), "shrinkage_weight": weight,
                })


def _order_robustness(context: RunContext, population_boundaries: np.ndarray,
                      participants: list[str]) -> pd.DataFrame:
    config = context.config
    rows = []
    for held, participant in enumerate(participants):
        with np.load(feature_path(context, participant), allow_pickle=False) as archive:
            duration_index = list(config.durations_seconds).index(config.primary_duration_seconds)
            coefficients = archive["target"][duration_index]
        for permutation_index in range(config.order_permutations):
            permuted = np.empty_like(coefficients)
            for stream in range(config.streams):
                rng = np.random.default_rng(np.random.SeedSequence([config.seed, held, stream, permutation_index, 733]))
                permuted[stream] = coefficients[stream, rng.permutation(config.trials_per_stream)]
            global_r2, _ = phase_trajectories(permuted, config.k_values)
            scores = fixed_stream_scores(global_r2, config.k_values)
            for method_index, method in enumerate(METHODS[:2]):
                first = first_crossings(scores[..., method_index], population_boundaries[held, method_index], config.k_values)
                rows.append({"participant": participant, "permutation": permutation_index,
                             "method": method, "sensitivity": float(np.mean(first <= config.k_values[-1])),
                             "restricted_mean_stopping_trials": float(np.mean(first))})
    return pd.DataFrame(rows)


def _independent_sensitivity(context: RunContext, archive) -> pd.DataFrame:
    config = context.config
    positive, independent = archive["positive"], archive["independent_null"]
    participants = archive["participants"].astype(str).tolist()
    rows = []
    for held, participant in enumerate(participants):
        training = np.arange(len(participants)) != held
        for method_index, method in enumerate(METHODS[:2]):
            training_paths = independent[training, ..., method_index].reshape(-1, len(config.k_values))
            calibrated = calibrate_pocock_boundaries(training_paths, config.target_fpr)
            held_paths = independent[held, ..., method_index].reshape(-1, len(config.k_values))
            first = first_crossings(positive[held, ..., method_index], calibrated.boundaries, config.k_values)
            rows.append({"participant": participant, "method": method,
                         "null": "independent_harmonic_rotations",
                         "sensitivity": float(np.mean(first <= config.k_values[-1])),
                         "familywise_fpr": crossing_rate(held_paths, calibrated.boundaries),
                         "restricted_mean_stopping_trials": float(np.mean(first)),
                         "marginal_tail_probability": calibrated.marginal_tail_probability})
    return pd.DataFrame(rows)


def run(context: RunContext, force: bool = False) -> tuple[list[Path], dict]:
    del force
    stage, config = "03_calibration", context.config
    context.start_stage(stage, len(config.durations_seconds) + 3)
    score_rows, boundary_rows, metric_rows = [], [], []
    primary_archive = None; primary_boundaries = None
    for index, duration in enumerate(config.durations_seconds, 1):
        with np.load(score_path(context, duration), allow_pickle=False) as archive:
            copied = {name: archive[name] for name in archive.files}
        boundaries = _population_analysis(context, duration, copied, score_rows, boundary_rows, metric_rows)
        if duration == config.primary_duration_seconds:
            primary_archive, primary_boundaries = copied, boundaries
        context.progress(stage, index, len(config.durations_seconds) + 3, f"population {duration:g}s")
    assert primary_archive is not None and primary_boundaries is not None
    _shrinkage_analysis(context, primary_archive, primary_boundaries, score_rows, boundary_rows, metric_rows)
    context.progress(stage, len(config.durations_seconds) + 1, len(config.durations_seconds) + 3, "shrinkage calibration")
    participants = primary_archive["participants"].astype(str).tolist()
    order = _order_robustness(context, primary_boundaries, participants)
    context.progress(stage, len(config.durations_seconds) + 2, len(config.durations_seconds) + 3, "order robustness")
    independent = _independent_sensitivity(context, primary_archive)
    context.progress(stage, len(config.durations_seconds) + 3, len(config.durations_seconds) + 3, "null sensitivity")
    outputs = []
    for name, frame in (
        ("sequential_scores.csv", pd.DataFrame(score_rows)),
        ("sequential_boundaries.csv", pd.DataFrame(boundary_rows)),
        ("calibration_comparisons.csv", pd.DataFrame(metric_rows)),
        ("order_robustness.csv", order),
        ("surrogate_sensitivity.csv", independent),
    ):
        path = context.table_dir / name; frame.to_csv(path, index=False); outputs.append(path)
    summary = {"population_cells": len(config.durations_seconds) * len(participants) * len(METHODS),
               "shrinkage_cells": len(participants) * config.conditions * len(METHODS),
               "primary_duration_seconds": config.primary_duration_seconds,
               "order_permutations": config.order_permutations}
    report = context.run_dir / "sequential_calibration.json"
    atomic_write_json(report, summary); outputs.append(report)
    context.finish_stage(stage, outputs, summary)
    return outputs, summary
