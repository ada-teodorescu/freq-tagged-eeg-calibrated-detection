from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from cucu_accuracy_study.features import region_indices
from cucu_algorithm_study.algorithms import fit_monotone_logistic
from cucu_advanced_study.algorithms import (
    blend_boundaries, calibrate_pocock_boundaries, crossing_rate, first_crossings,
    fixed_stream_scores, phase_trajectories, regional_stream_features, select_shrinkage_weight,
    temporal_shift_multipliers,
)
from cucu_article.runtime import atomic_save_npz, atomic_write_json

from ..runtime import RunContext
from .coefficients import feature_path
from .trca_features import trca_path


METHODS = ("S2_Equal4plus8", "S4_RegionalMonotone4plus8", "TRCA_S2_Grouped")
SHRINKAGE_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _rotations(config, participant: int, stream: int, harmonics: int, seed_salt: int = 0) -> np.ndarray:
    blocks = []
    for seed in config.surrogate_seeds:
        sequence = np.random.SeedSequence([seed, participant, stream, seed_salt])
        blocks.append(temporal_shift_multipliers(
            config.trials_per_stream, tuple(range(1, harmonics + 1)), config.surrogates_per_seed, sequence,
        ))
    return np.concatenate(blocks, axis=0)


def _phase_bundle(context: RunContext, coefficients: np.ndarray, participant_index: int,
                  regions: tuple[np.ndarray, ...] | None, harmonics: int, seed_salt: int):
    config = context.config
    positive_global, positive_regional = phase_trajectories(coefficients, config.k_values, regions)
    null_global = np.empty((config.streams, config.surrogates, len(config.k_values), harmonics), np.float32)
    null_regional = None if regions is None else np.empty(
        (config.streams, config.surrogates, len(config.k_values), len(regions), harmonics), np.float32,
    )
    for stream in range(config.streams):
        rotations = _rotations(config, participant_index, stream, harmonics, seed_salt)
        g, r = phase_trajectories(coefficients[stream:stream + 1], config.k_values, regions, rotations)
        null_global[stream] = g[0]
        if regions is not None:
            null_regional[stream] = r[0]
    return positive_global, null_global, positive_regional, null_regional


def _training_rows(positive: np.ndarray, null: np.ndarray, training: np.ndarray, look: int):
    pos = positive[training, :, look].reshape(-1, positive.shape[-1])
    neg = null[training, :, :, look].reshape(-1, null.shape[-1])
    values = np.concatenate((pos, neg))
    labels = np.concatenate((np.ones(len(pos), np.int8), np.zeros(len(neg), np.int8)))
    return values, labels


def run(context: RunContext, force: bool = False) -> tuple[list[Path], dict]:
    stage, config = "05_trca_sequential", context.config
    lock = json.loads((context.run_dir / "design_lock.json").read_text(encoding="utf-8"))
    participants = lock["participants"]
    regions = region_indices(tuple(lock["channel_labels"]), config.regions)
    checkpoint = context.checkpoint_dir / "05_trca_sequential" / "scores.npz"
    n_k = len(config.k_values)
    context.start_stage(stage, len(participants) + 1)

    if context.checkpoint_valid(checkpoint) and not force:
        with np.load(checkpoint, allow_pickle=False) as archive:
            positive, null = archive["positive"], archive["null"]
    else:
        positive = np.empty((len(participants), config.streams, n_k, len(METHODS)), np.float32)
        null = np.empty((len(participants), config.streams, config.surrogates, n_k, len(METHODS)), np.float32)
        regional_positive = np.empty((len(participants), config.streams, n_k, len(regions) * 2), np.float32)
        regional_null = np.empty((len(participants), config.streams, config.surrogates, n_k, len(regions) * 2), np.float32)

        for participant_index, participant in enumerate(participants):
            with np.load(feature_path(context, participant), allow_pickle=False) as archive:
                s2_coefficients = archive["hann_full"][..., :2]
            pos_g, null_g, pos_r, null_r = _phase_bundle(context, s2_coefficients, participant_index, regions, 2, seed_salt=0)
            pos_fixed = fixed_stream_scores(pos_g, config.k_values)  # streams x K x 2 (s1, s2)
            null_fixed = fixed_stream_scores(null_g, config.k_values)
            positive[participant_index, :, :, 0] = pos_fixed[..., 1]
            null[participant_index, :, :, :, 0] = null_fixed[..., 1]
            regional_positive[participant_index] = regional_stream_features(pos_r, config.k_values)
            regional_null[participant_index] = regional_stream_features(null_r, config.k_values)

            with np.load(trca_path(context, participant), allow_pickle=False) as archive:
                trca_coefficients = archive["trca_coefficients"]
            trca_pos_g, trca_null_g, _, _ = _phase_bundle(context, trca_coefficients, participant_index, None, 2, seed_salt=1)
            trca_fixed = fixed_stream_scores(trca_pos_g, config.k_values)
            trca_null_fixed = fixed_stream_scores(trca_null_g, config.k_values)
            positive[participant_index, :, :, 2] = trca_fixed[..., 1]
            null[participant_index, :, :, :, 2] = trca_null_fixed[..., 1]
            context.progress(stage, participant_index + 1, len(participants) + 1, f"fixed scores {participant}")

        # S4 is the only method needing a per-held-out, per-look refit (regional monotone
        # fusion) -- S2 and TRCA_S2 are fixed formulas requiring no learned model.
        method_index = METHODS.index("S4_RegionalMonotone4plus8")
        for held, participant in enumerate(participants):
            training = np.arange(len(participants)) != held
            for look in range(n_k):
                values, labels = _training_rows(regional_positive, regional_null, training, look)
                model = fit_monotone_logistic(values, labels, config.locked_c, config.maximum_iterations, config.tolerance)
                positive[held, :, look, method_index] = model.decision_function(regional_positive[held, :, look])
                flat_null = regional_null[held, :, :, look].reshape(-1, regional_null.shape[-1])
                null[held, :, :, look, method_index] = model.decision_function(flat_null).reshape(config.streams, config.surrogates)

        atomic_save_npz(checkpoint, positive=positive, null=null,
                        participants=np.asarray(participants), k_values=np.asarray(config.k_values))
        context.mark_checkpoint(checkpoint, {"unit": "participant_stream", "methods": list(METHODS)})

    # Population Pocock calibration for all three methods, mirroring
    # cucu_advanced_study.stages.calibration's _population_analysis pattern.
    score_rows, boundary_rows, metric_rows = [], [], []
    population_boundaries = np.empty((len(participants), len(METHODS), n_k), float)
    for held, participant in enumerate(participants):
        training = np.arange(len(participants)) != held
        for method_index, method in enumerate(METHODS):
            calibration_null = null[training, :, :, :, method_index].reshape(-1, n_k)
            calibrated = calibrate_pocock_boundaries(calibration_null, config.target_fpr)
            boundaries = calibrated.boundaries
            population_boundaries[held, method_index] = boundaries
            for look, k in enumerate(config.k_values):
                boundary_rows.append({"participant": participant, "condition": 0, "method": method,
                                      "calibration": "population", "look_k": k, "boundary": float(boundaries[look]),
                                      "marginal_tail_probability": calibrated.marginal_tail_probability,
                                      "training_familywise_fpr": calibrated.familywise_fpr, "shrinkage_weight": 0.0})
            scores = positive[held, :, :, method_index]
            first = first_crossings(scores, boundaries, config.k_values)
            crossed = np.maximum.accumulate(scores > boundaries[None, :], axis=1)
            for stream in range(config.streams):
                row = {"participant": participant, "condition": stream // 3 + 1,
                       "stream_within_condition": stream % 3 + 1, "method": method, "calibration": "population",
                       "first_detection_k": int(first[stream]), "detected": bool(first[stream] <= config.k_values[-1])}
                row.update({f"score_k{k}": float(scores[stream, look]) for look, k in enumerate(config.k_values)})
                row.update({f"detected_by_k{k}": bool(crossed[stream, look]) for look, k in enumerate(config.k_values)})
                score_rows.append(row)
            held_null = null[held, :, :, :, method_index].reshape(-1, n_k)
            metric_rows.append({"participant": participant, "condition": 0, "method": method,
                                "calibration": "population", "sensitivity": float(np.mean(first <= config.k_values[-1])),
                                "familywise_fpr": crossing_rate(held_null, boundaries),
                                "restricted_mean_stopping_trials": float(np.mean(first)), "shrinkage_weight": 0.0})

    # Shrinkage calibration for S4 only -- the information-matched comparator TRCA is
    # judged against, since TRCA is already participant-adapted (its filter was fit via
    # leave-one-condition-out within the held participant), unlike population-calibrated S2/S4.
    method_index = METHODS.index("S4_RegionalMonotone4plus8")
    for held, participant in enumerate(participants):
        pop = population_boundaries[held, method_index]
        for evaluated_condition in range(config.conditions):
            calibration_conditions = [c for c in range(config.conditions) if c != evaluated_condition]
            local_trajectories = np.concatenate([
                null[held, c * 3:(c + 1) * 3, :, :, method_index].reshape(-1, n_k) for c in calibration_conditions
            ])
            local = calibrate_pocock_boundaries(local_trajectories, config.target_fpr).boundaries
            inner_boundaries, inner_validation = [], []
            for validation_condition in calibration_conditions:
                inner_training = [c for c in calibration_conditions if c != validation_condition]
                training_paths = np.concatenate([
                    null[held, c * 3:(c + 1) * 3, :, :, method_index].reshape(-1, n_k) for c in inner_training
                ])
                inner_boundaries.append(calibrate_pocock_boundaries(training_paths, config.target_fpr).boundaries)
                inner_validation.append(
                    null[held, validation_condition * 3:(validation_condition + 1) * 3, :, :, method_index].reshape(-1, n_k)
                )
            weight, errors = select_shrinkage_weight(
                pop, np.asarray(inner_boundaries), np.asarray(inner_validation), SHRINKAGE_WEIGHTS, config.target_fpr,
            )
            limits = blend_boundaries(pop, local, weight)
            for look, k in enumerate(config.k_values):
                boundary_rows.append({"participant": participant, "condition": evaluated_condition + 1,
                                      "method": "S4_RegionalMonotone4plus8", "calibration": "shrinkage", "look_k": k,
                                      "boundary": float(limits[look]), "marginal_tail_probability": np.nan,
                                      "training_familywise_fpr": np.nan, "shrinkage_weight": weight,
                                      "selection_rmse": errors[weight]})
            stream_slice = slice(evaluated_condition * 3, (evaluated_condition + 1) * 3)
            evaluated_scores = positive[held, stream_slice, :, method_index]
            first = first_crossings(evaluated_scores, limits, config.k_values)
            crossed = np.maximum.accumulate(evaluated_scores > limits[None, :], axis=1)
            for offset in range(3):
                row = {"participant": participant, "condition": evaluated_condition + 1,
                       "stream_within_condition": offset + 1, "method": "S4_RegionalMonotone4plus8",
                       "calibration": "shrinkage", "first_detection_k": int(first[offset]),
                       "detected": bool(first[offset] <= config.k_values[-1])}
                row.update({f"score_k{k}": float(evaluated_scores[offset, look]) for look, k in enumerate(config.k_values)})
                row.update({f"detected_by_k{k}": bool(crossed[offset, look]) for look, k in enumerate(config.k_values)})
                score_rows.append(row)
            evaluated_null = null[held, stream_slice, :, :, method_index].reshape(-1, n_k)
            metric_rows.append({"participant": participant, "condition": evaluated_condition + 1,
                                "method": "S4_RegionalMonotone4plus8", "calibration": "shrinkage",
                                "sensitivity": float(np.mean(first <= config.k_values[-1])),
                                "familywise_fpr": crossing_rate(evaluated_null, limits),
                                "restricted_mean_stopping_trials": float(np.mean(first)), "shrinkage_weight": weight})
    context.progress(stage, len(participants) + 1, len(participants) + 1, "population and shrinkage calibration")

    outputs = [checkpoint]
    for name, frame in (("trca_sequential_scores.csv", pd.DataFrame(score_rows)),
                        ("trca_sequential_boundaries.csv", pd.DataFrame(boundary_rows)),
                        ("trca_sequential_metrics.csv", pd.DataFrame(metric_rows))):
        path = context.table_dir / name; frame.to_csv(path, index=False); outputs.append(path)
    summary = {"methods": list(METHODS), "endpoint": "sequential_population_and_s4_shrinkage"}
    report = context.run_dir / "trca_sequential.json"
    atomic_write_json(report, summary); outputs.append(report)
    context.finish_stage(stage, outputs, summary)
    return outputs, summary
