from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from cucu_accuracy_study.features import region_indices
from cucu_algorithm_study.algorithms import fit_monotone_logistic, select_monotone_regularization
from cucu_article.runtime import atomic_save_npz, atomic_write_json
from cucu_article.stages.benchmark import empirical_threshold_fast

from ..algorithms import phase_trajectories
from ..runtime import RunContext
from .features import feature_path
from .stream import _rotations


METHODS = ("C1_ConditionITPC4", "C2_ConditionEqual4plus8",
           "C3_RegionalMonotone4plus8", "C4_PreviousRegionalMonotone4H")


def _features(context: RunContext, participant: int, coefficients: np.ndarray,
              regions: tuple[np.ndarray, ...]):
    config = context.config
    positive_global, positive_regional = phase_trajectories(coefficients, (config.trials_per_stream,), regions)
    null_global = np.empty((config.streams, config.surrogates, len(config.harmonics_hz)), np.float32)
    null_regional = np.empty((config.streams, config.surrogates, len(regions), len(config.harmonics_hz)), np.float32)
    for stream in range(config.streams):
        rotations = _rotations(context, participant, stream, len(config.harmonics_hz))
        global_scores, regional_scores = phase_trajectories(
            coefficients[stream:stream + 1], (config.trials_per_stream,), regions, rotations,
        )
        null_global[stream] = global_scores[0, :, 0]
        null_regional[stream] = regional_scores[0, :, 0]
    conditions = config.conditions; streams = config.streams_per_condition
    pos_g = positive_global[:, 0].reshape(conditions, streams, -1).mean(axis=1)
    pos_r = positive_regional[:, 0].reshape(conditions, streams, len(regions), -1).mean(axis=1)
    null_g = null_global.reshape(conditions, streams, config.surrogates, -1).mean(axis=1)
    null_r = null_regional.reshape(conditions, streams, config.surrogates, len(regions), -1).mean(axis=1)
    scale = 2.0 * config.trials_per_stream
    return (scale * pos_g).astype(np.float32), (scale * null_g).astype(np.float32), \
        (scale * pos_r).astype(np.float32), (scale * null_r).astype(np.float32)


def _sample_null(null: np.ndarray, count: int, seed: int) -> np.ndarray:
    result = np.empty((null.shape[0], null.shape[1], count, null.shape[-1]), null.dtype)
    for participant in range(null.shape[0]):
        for condition in range(null.shape[1]):
            rng = np.random.default_rng(np.random.SeedSequence([seed, participant, condition, 451]))
            selected = np.sort(rng.choice(null.shape[2], count, replace=False))
            result[participant, condition] = null[participant, condition, selected]
    return result


def _training_rows(positive: np.ndarray, null: np.ndarray, training: np.ndarray):
    pos = positive[training].reshape(-1, positive.shape[-1])
    neg = null[training].reshape(-1, null.shape[-1])
    ids = np.arange(positive.shape[0])[training]
    groups = np.concatenate((np.repeat(ids, positive.shape[1]),
                             np.repeat(ids, positive.shape[1] * null.shape[2])))
    labels = np.concatenate((np.ones(len(pos), np.int8), np.zeros(len(neg), np.int8)))
    return np.concatenate((pos, neg)), labels, groups


def run(context: RunContext, force: bool = False) -> tuple[list[Path], dict]:
    stage, config = "04_condition", context.config
    checkpoint = context.checkpoint_dir / "04_condition" / "condition_features.npz"
    lock = json.loads((context.run_dir / "design_lock.json").read_text(encoding="utf-8"))
    participants = lock["participants"]
    regions = region_indices(tuple(lock["channel_labels"]), config.regions)
    context.start_stage(stage, len(participants) + 1)
    if context.checkpoint_valid(checkpoint) and not force:
        with np.load(checkpoint, allow_pickle=False) as archive:
            pos_g, null_g, pos_r, null_r = [archive[name] for name in ("positive_global", "null_global", "positive_regional", "null_regional")]
        context.progress(stage, len(participants), len(participants) + 1, "feature checkpoint")
    else:
        bundles = []
        duration_index = list(config.durations_seconds).index(config.primary_duration_seconds)
        for participant_index, participant in enumerate(participants):
            with np.load(feature_path(context, participant), allow_pickle=False) as archive:
                coefficients = archive["target"][duration_index]
            bundles.append(_features(context, participant_index, coefficients, regions))
            context.progress(stage, participant_index + 1, len(participants) + 1, participant)
        pos_g, null_g, pos_r, null_r = [np.stack([bundle[index] for bundle in bundles]) for index in range(4)]
        atomic_save_npz(checkpoint, positive_global=pos_g, null_global=null_g,
                        positive_regional=pos_r, null_regional=null_r)
        context.mark_checkpoint(checkpoint, {"unit": "participant_condition", "streams_aggregated": 3})
    positive_scores = np.empty((len(participants), config.conditions, len(METHODS)), np.float32)
    null_scores = np.empty((len(participants), config.conditions, config.surrogates, len(METHODS)), np.float32)
    positive_scores[..., 0] = pos_g[..., 0]
    positive_scores[..., 1] = pos_g[..., 0] + pos_g[..., 1]
    null_scores[..., 0] = null_g[..., 0]
    null_scores[..., 1] = null_g[..., 0] + null_g[..., 1]
    feature_sets = ((pos_r[..., :2].reshape(*pos_r.shape[:2], -1),
                     null_r[..., :2].reshape(*null_r.shape[:3], -1)),
                    (pos_r.reshape(*pos_r.shape[:2], -1), null_r.reshape(*null_r.shape[:3], -1)))
    records = []
    thresholds = np.empty((len(participants), len(METHODS)), float)
    sampled_sets = [(positive, _sample_null(null, config.training_nulls_per_stream, config.seed + index))
                    for index, (positive, null) in enumerate(feature_sets)]
    for held, participant in enumerate(participants):
        training = np.arange(len(participants)) != held
        for method_index in range(2):
            thresholds[held, method_index] = empirical_threshold_fast(
                null_scores[training, ..., method_index], config.target_fpr,
            )
        for feature_index, (positive_features, sampled_null) in enumerate(sampled_sets):
            values, labels, groups = _training_rows(positive_features, sampled_null, training)
            c_value, inner_scores = select_monotone_regularization(
                values, labels, groups, config.c_grid, config.inner_group_folds,
                config.seed + 2300 + feature_index, config.maximum_iterations, config.tolerance,
            )
            model = fit_monotone_logistic(values, labels, c_value,
                                          config.maximum_iterations, config.tolerance)
            method_index = feature_index + 2
            positive_scores[held, :, method_index] = model.decision_function(positive_features[held])
            full_features = feature_sets[feature_index][1][held]
            null_scores[held, :, :, method_index] = model.decision_function(
                full_features.reshape(-1, full_features.shape[-1])
            ).reshape(config.conditions, config.surrogates)
            complete_training_null = feature_sets[feature_index][1][training]
            training_null_scores = model.decision_function(
                complete_training_null.reshape(-1, complete_training_null.shape[-1])
            )
            thresholds[held, method_index] = empirical_threshold_fast(training_null_scores, config.target_fpr)
            records.append({"heldout": participant, "method": METHODS[method_index], "selected_c": c_value,
                            "inner_scores": json.dumps(inner_scores, sort_keys=True),
                            "coefficients": json.dumps(model.coefficients.tolist()),
                            "intercept": model.intercept,
                            "standardizer_mean": json.dumps(model.standardizer.mean.tolist()),
                            "standardizer_scale": json.dumps(model.standardizer.scale.tolist()),
                            "converged": model.converged})
    score_rows, metric_rows = [], []
    for held, participant in enumerate(participants):
        for method_index, method in enumerate(METHODS):
            threshold = thresholds[held, method_index]
            detected = positive_scores[held, :, method_index] > threshold
            false_positive = null_scores[held, :, :, method_index] > threshold
            metric_rows.append({"participant": participant, "method": method,
                                "sensitivity": float(detected.mean()), "fpr": float(false_positive.mean()),
                                "threshold": threshold})
            for condition in range(config.conditions):
                score_rows.append({"participant": participant, "condition": condition + 1, "method": method,
                                   "score": float(positive_scores[held, condition, method_index]),
                                   "threshold": threshold, "detected": bool(detected[condition])})
    outputs = [checkpoint]
    for name, frame in (("condition_scores.csv", pd.DataFrame(score_rows)),
                        ("condition_metrics.csv", pd.DataFrame(metric_rows)),
                        ("condition_model_records.csv", pd.DataFrame(records))):
        path = context.table_dir / name; frame.to_csv(path, index=False); outputs.append(path)
    summary = {"unit": "participant_condition", "conditions": len(participants) * config.conditions,
               "methods": list(METHODS), "streams_aggregated": config.streams_per_condition}
    report = context.run_dir / "condition_benchmark.json"
    atomic_write_json(report, summary); outputs.append(report)
    context.progress(stage, len(participants) + 1, len(participants) + 1, "condition benchmark")
    context.finish_stage(stage, outputs, summary)
    return outputs, summary
