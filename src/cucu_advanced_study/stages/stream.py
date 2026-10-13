from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from cucu_accuracy_study.features import region_indices
from cucu_algorithm_study.algorithms import fit_monotone_logistic, select_monotone_regularization
from cucu_article.algorithms import grouped_folds
from cucu_article.runtime import atomic_save_npz, atomic_write_json
from cucu_article.stages.benchmark import auc_fast

from ..algorithms import (
    fixed_stream_scores, phase_trajectories, regional_stream_features,
    ridge_complex_trajectories, temporal_shift_multipliers,
)
from ..runtime import RunContext
from .features import feature_path


METHODS = ("S1_ITPC4", "S2_Equal4plus8", "S3_RidgeComplex4plus8", "S4_RegionalMonotone4plus8")


def score_path(context: RunContext, duration: float) -> Path:
    label = str(duration).replace(".", "p")
    return context.checkpoint_dir / "02_stream_scores" / f"duration_{label}.npz"


def _rotations(context: RunContext, participant: int, stream: int, harmonics: int,
               independent: bool = False) -> np.ndarray:
    config = context.config
    blocks = []
    for seed in config.surrogate_seeds:
        sequence = np.random.SeedSequence([seed, participant, stream, 1 if independent else 0])
        blocks.append(temporal_shift_multipliers(
            config.trials_per_stream, tuple(range(1, harmonics + 1)), config.surrogates_per_seed,
            sequence, independent_harmonics=independent,
        ))
    return np.concatenate(blocks, axis=0)


def _phase_bundle(context: RunContext, coefficients: np.ndarray, participant: int,
                  *, full_null: bool, independent: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    config = context.config
    channel_labels = tuple(json.loads((context.run_dir / "design_lock.json").read_text())["channel_labels"])
    regions = region_indices(channel_labels, config.regions)
    positive_global, positive_regional = phase_trajectories(coefficients, config.k_values, regions)
    count = config.surrogates if full_null else config.training_nulls_per_stream
    null_global = np.empty((config.streams, count, len(config.k_values), coefficients.shape[-1]), np.float32)
    null_regional = np.empty((config.streams, count, len(config.k_values), len(regions), coefficients.shape[-1]), np.float32)
    for stream in range(config.streams):
        rotations = _rotations(context, participant, stream, coefficients.shape[-1], independent)
        if not full_null:
            rng = np.random.default_rng(np.random.SeedSequence([config.seed, participant, stream, 221]))
            rotations = rotations[np.sort(rng.choice(rotations.shape[0], count, replace=False))]
        global_scores, regional_scores = phase_trajectories(
            coefficients[stream:stream + 1], config.k_values, regions, rotations,
        )
        null_global[stream] = global_scores[0]
        null_regional[stream] = regional_scores[0]
    return positive_global, null_global, regional_stream_features(positive_regional, config.k_values), \
        regional_stream_features(null_regional, config.k_values)


def _ridge_sampled(context: RunContext, coefficients: np.ndarray, participant: int) -> tuple[np.ndarray, np.ndarray]:
    config = context.config
    positive = np.empty((len(config.ridge_grid), config.streams, len(config.k_values)), np.float32)
    null = np.empty((len(config.ridge_grid), config.streams, config.training_nulls_per_stream,
                     len(config.k_values)), np.float32)
    for stream in range(config.streams):
        rotations = _rotations(context, participant, stream, 2)
        rng = np.random.default_rng(np.random.SeedSequence([config.seed, participant, stream, 221]))
        rotations = rotations[np.sort(rng.choice(rotations.shape[0], config.training_nulls_per_stream, replace=False))]
        for ridge_index, ridge in enumerate(config.ridge_grid):
            positive[ridge_index, stream] = ridge_complex_trajectories(
                coefficients[stream:stream + 1], config.k_values, ridge,
            )[0]
            null[ridge_index, stream] = ridge_complex_trajectories(
                coefficients[stream:stream + 1], config.k_values, ridge, rotations,
            )[0]
    return positive, null


def _select_ridge(context: RunContext, positive: np.ndarray, null: np.ndarray,
                  training: np.ndarray, look: int) -> tuple[float, dict[float, float]]:
    config = context.config
    participant_ids = np.arange(positive.shape[1])[training]
    folds = grouped_folds(participant_ids, config.inner_group_folds, config.seed + look)
    scores = {}
    for ridge_index, ridge in enumerate(config.ridge_grid):
        fold_scores = []
        for validation_ids in folds:
            validation = np.isin(np.arange(positive.shape[1]), validation_ids) & training
            fold_scores.append(auc_fast(positive[ridge_index, validation, :, look],
                                        null[ridge_index, validation, :, :, look]))
        scores[float(ridge)] = float(np.mean(fold_scores))
    best = max(scores.values())
    selected = min(ridge for ridge, score in scores.items() if np.isclose(score, best, atol=1e-12, rtol=0))
    return float(selected), scores


def _training_rows(positive: np.ndarray, null: np.ndarray, training: np.ndarray, look: int):
    participants, streams = positive.shape[:2]
    pos = positive[training, :, look].reshape(-1, positive.shape[-1])
    neg = null[training, :, :, look].reshape(-1, null.shape[-1])
    ids = np.arange(participants)[training]
    groups = np.concatenate((np.repeat(ids, streams), np.repeat(ids, streams * null.shape[2])))
    labels = np.concatenate((np.ones(len(pos), np.int8), np.zeros(len(neg), np.int8)))
    return np.concatenate((pos, neg)), labels, groups


def run(context: RunContext, force: bool = False) -> tuple[list[Path], dict]:
    stage, config = "02_stream", context.config
    lock = json.loads((context.run_dir / "design_lock.json").read_text(encoding="utf-8"))
    participants = lock["participants"]
    outputs = []
    model_path = context.table_dir / "stream_model_records.csv"
    records = pd.read_csv(model_path).to_dict("records") if model_path.exists() and not force else []
    total = 2 * len(participants) * len(config.durations_seconds)
    context.start_stage(stage, total); completed = 0
    for duration_index, duration in enumerate(config.durations_seconds):
        destination = score_path(context, duration)
        if context.checkpoint_valid(destination) and not force:
            outputs.append(destination); completed += 2 * len(participants)
            context.progress(stage, completed, total, f"duration {duration:g}s checkpoint")
            continue
        records = [row for row in records if not np.isclose(float(row["duration_seconds"]), duration)]
        sampled_positive_global = []
        sampled_null_global = []
        regional_positive = []
        regional_null = []
        ridge_positive = []
        ridge_null = []
        coefficients_by_participant = []
        for participant_index, participant in enumerate(participants):
            with np.load(feature_path(context, participant), allow_pickle=False) as archive:
                coefficients = archive["target"][duration_index]
            coefficients_by_participant.append(coefficients)
            pos_g, null_g, pos_r, null_r = _phase_bundle(
                context, coefficients, participant_index, full_null=False,
            )
            pos_ridge, null_ridge = _ridge_sampled(context, coefficients[..., :2], participant_index)
            sampled_positive_global.append(pos_g); sampled_null_global.append(null_g)
            regional_positive.append(pos_r); regional_null.append(null_r)
            ridge_positive.append(pos_ridge); ridge_null.append(null_ridge)
            completed += 1; context.progress(stage, completed, total, f"{duration:g}s compact {participant}")
        pos_global = np.stack(sampled_positive_global)
        sampled_global = np.stack(sampled_null_global)
        pos_fixed = fixed_stream_scores(pos_global, config.k_values)
        sampled_fixed = fixed_stream_scores(sampled_global, config.k_values)
        pos_regional, null_regional = np.stack(regional_positive), np.stack(regional_null)
        pos_ridge, null_ridge = np.stack(ridge_positive, axis=1), np.stack(ridge_null, axis=1)
        positive = np.empty((len(participants), config.streams, len(config.k_values), len(METHODS)), np.float32)
        null = np.empty((len(participants), config.streams, config.surrogates, len(config.k_values), len(METHODS)), np.float32)
        calibration_count = (len(participants) - 1) * config.streams * config.training_nulls_per_stream
        calibration_null = np.empty((len(participants), calibration_count, len(config.k_values), len(METHODS)), np.float32)
        positive[..., :2] = pos_fixed
        independent_null = None
        if duration == config.primary_duration_seconds:
            independent_null = np.empty((len(participants), config.streams, config.surrogates,
                                         len(config.k_values), 2), np.float32)
        for held, participant in enumerate(participants):
            training = np.arange(len(participants)) != held
            calibration_null[held, ..., :2] = sampled_fixed[training].reshape(
                calibration_count, len(config.k_values), 2,
            )
            coefficients = coefficients_by_participant[held]
            _, full_global, _, full_regional = _phase_bundle(context, coefficients, held, full_null=True)
            null[held, ..., :2] = fixed_stream_scores(full_global, config.k_values)
            if independent_null is not None:
                _, independent_global, _, _ = _phase_bundle(
                    context, coefficients, held, full_null=True, independent=True,
                )
                independent_null[held] = fixed_stream_scores(independent_global, config.k_values)
            selected_by_look = []
            for look in range(len(config.k_values)):
                selected, inner_scores = _select_ridge(context, pos_ridge, null_ridge, training, look)
                selected_by_look.append(selected)
                records.append({"duration_seconds": duration, "heldout": participant,
                                "look_k": config.k_values[look], "model": "S3", "selected": selected,
                                "inner_scores": json.dumps(inner_scores, sort_keys=True)})
            for ridge in sorted(set(selected_by_look)):
                candidate = ridge_complex_trajectories(coefficients, config.k_values, ridge)
                full_candidate = np.empty((config.streams, config.surrogates, len(config.k_values)), np.float32)
                for stream in range(config.streams):
                    rotations = _rotations(context, held, stream, 2)
                    full_candidate[stream] = ridge_complex_trajectories(
                        coefficients[stream:stream + 1], config.k_values, ridge, rotations,
                    )[0]
                for look, chosen in enumerate(selected_by_look):
                    if chosen == ridge:
                        positive[held, :, look, 2] = candidate[:, look]
                        null[held, :, :, look, 2] = full_candidate[:, :, look]
                        ridge_index = config.ridge_grid.index(ridge)
                        calibration_null[held, :, look, 2] = null_ridge[
                            ridge_index, training, :, :, look
                        ].reshape(calibration_count)
            for look, k in enumerate(config.k_values):
                values, labels, groups = _training_rows(pos_regional, null_regional, training, look)
                c_value, inner_scores = select_monotone_regularization(
                    values, labels, groups, config.c_grid, config.inner_group_folds,
                    config.seed + 1000 + look, config.maximum_iterations, config.tolerance,
                )
                model = fit_monotone_logistic(values, labels, c_value,
                                              config.maximum_iterations, config.tolerance)
                positive[held, :, look, 3] = model.decision_function(pos_regional[held, :, look])
                flat = full_regional[:, :, look].reshape(-1, full_regional.shape[-1])
                null[held, :, :, look, 3] = model.decision_function(flat).reshape(config.streams, config.surrogates)
                training_flat = null_regional[training, :, :, look].reshape(-1, null_regional.shape[-1])
                calibration_null[held, :, look, 3] = model.decision_function(training_flat)
                records.append({"duration_seconds": duration, "heldout": participant, "look_k": k,
                                "model": "S4", "selected": c_value,
                                "inner_scores": json.dumps(inner_scores, sort_keys=True),
                                "coefficients": json.dumps(model.coefficients.tolist()),
                                "intercept": model.intercept,
                                "standardizer_mean": json.dumps(model.standardizer.mean.tolist()),
                                "standardizer_scale": json.dumps(model.standardizer.scale.tolist()),
                                "converged": model.converged})
            completed += 1; context.progress(stage, completed, total, f"{duration:g}s outer {participant}")
        arrays = {"positive": positive, "null": null, "calibration_null": calibration_null,
                  "methods": np.asarray(METHODS),
                  "participants": np.asarray(participants), "k_values": np.asarray(config.k_values),
                  "duration_seconds": np.asarray(duration)}
        if independent_null is not None:
            arrays["independent_null"] = independent_null
        atomic_save_npz(destination, **arrays)
        context.mark_checkpoint(destination, {"duration_seconds": duration, "outer_validation": "LOPO"})
        outputs.append(destination)
    pd.DataFrame(records).to_csv(model_path, index=False); outputs.append(model_path)
    summary = {"methods": list(METHODS), "durations": list(config.durations_seconds),
               "surrogates": config.surrogates, "outer_folds": len(participants)}
    report = context.run_dir / "stream_benchmark.json"
    atomic_write_json(report, summary); outputs.append(report)
    context.finish_stage(stage, outputs, summary)
    return outputs, summary
