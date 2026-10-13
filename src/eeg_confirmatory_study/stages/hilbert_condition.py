from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from eeg_accuracy_study.features import region_indices
from eeg_algorithm_study.algorithms import fit_monotone_logistic
from eeg_advanced_study.algorithms import phase_trajectories, temporal_shift_multipliers
from eeg_core.runtime import atomic_save_npz, atomic_write_json
from eeg_detection.evaluation import empirical_threshold

from ..runtime import RunContext
from .coefficients import feature_path


METHODS = ("H1_Hann4s_Regional4H", "H2_Hilbert4s_Regional4H", "H3_Hann4p5s_Regional4H")
FEATURE_KEYS = {
    "H1_Hann4s_Regional4H": "hann_matched",
    "H2_Hilbert4s_Regional4H": "hilbert",
    "H3_Hann4p5s_Regional4H": "hann_full",
}


def _rotations(config, participant: int, stream: int, seed_salt: int = 0) -> np.ndarray:
    blocks = []
    for seed in config.surrogate_seeds:
        sequence = np.random.SeedSequence([seed, participant, stream, seed_salt])
        blocks.append(temporal_shift_multipliers(
            config.trials_per_stream, tuple(range(1, len(config.harmonics_hz) + 1)),
            config.surrogates_per_seed, sequence,
        ))
    return np.concatenate(blocks, axis=0)


def _condition_features(context: RunContext, participant_index: int, coefficients: np.ndarray,
                        regions: tuple[np.ndarray, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Static single-look K=10 regional 4-harmonic condition features (mirrors advanced_study's C3/C4)."""
    config = context.config
    _, positive_regional = phase_trajectories(coefficients, (config.trials_per_stream,), regions)
    null_regional = np.empty((config.streams, config.surrogates, len(regions), len(config.harmonics_hz)), np.float32)
    for stream in range(config.streams):
        rotations = _rotations(config, participant_index, stream)
        _, regional_scores = phase_trajectories(coefficients[stream:stream + 1], (config.trials_per_stream,), regions, rotations)
        null_regional[stream] = regional_scores[0, :, 0]
    conditions, streams_per = config.conditions, config.streams_per_condition
    pos_r = positive_regional[:, 0].reshape(conditions, streams_per, len(regions), -1).mean(axis=1)
    null_r = null_regional.reshape(conditions, streams_per, config.surrogates, len(regions), -1).mean(axis=1)
    scale = 2.0 * config.trials_per_stream
    return (scale * pos_r).astype(np.float32), (scale * null_r).astype(np.float32)


def run(context: RunContext, force: bool = False) -> tuple[list[Path], dict]:
    stage, config = "02_hilbert_condition", context.config
    lock = json.loads((context.run_dir / "design_lock.json").read_text(encoding="utf-8"))
    participants = lock["participants"]
    regions = region_indices(tuple(lock["channel_labels"]), config.regions)
    checkpoint = context.checkpoint_dir / "02_hilbert_condition" / "condition_features.npz"
    context.start_stage(stage, len(participants) + 1)

    if context.checkpoint_valid(checkpoint) and not force:
        with np.load(checkpoint, allow_pickle=False) as archive:
            bundle = {method: (archive[f"{method}_pos"], archive[f"{method}_null"]) for method in METHODS}
        context.progress(stage, len(participants), len(participants) + 1, "feature checkpoint")
    else:
        per_method: dict[str, tuple[list, list]] = {method: ([], []) for method in METHODS}
        for participant_index, participant in enumerate(participants):
            with np.load(feature_path(context, participant), allow_pickle=False) as archive:
                for method in METHODS:
                    coefficients = archive[FEATURE_KEYS[method]]
                    pos, null = _condition_features(context, participant_index, coefficients, regions)
                    per_method[method][0].append(pos)
                    per_method[method][1].append(null)
            context.progress(stage, participant_index + 1, len(participants) + 1, participant)
        bundle = {method: (np.stack(pos_list), np.stack(null_list)) for method, (pos_list, null_list) in per_method.items()}
        save_kwargs: dict[str, np.ndarray] = {}
        for method, (pos, null) in bundle.items():
            save_kwargs[f"{method}_pos"] = pos
            save_kwargs[f"{method}_null"] = null
        atomic_save_npz(checkpoint, **save_kwargs)
        context.mark_checkpoint(checkpoint, {"unit": "participant_condition", "streams_aggregated": 3})

    score_rows, metric_rows, model_rows = [], [], []
    for method in METHODS:
        pos, null = bundle[method]  # pos: participants x conditions x regions x harmonics
        pos_flat = pos.reshape(*pos.shape[:2], -1)
        null_flat = null.reshape(*null.shape[:3], -1)
        for held, participant in enumerate(participants):
            training = np.arange(len(participants)) != held
            train_pos = pos_flat[training].reshape(-1, pos_flat.shape[-1])
            train_null = null_flat[training].reshape(-1, null_flat.shape[-1])
            values = np.concatenate((train_pos, train_null))
            labels = np.concatenate((np.ones(len(train_pos), np.int8), np.zeros(len(train_null), np.int8)))
            model = fit_monotone_logistic(values, labels, config.locked_c, config.maximum_iterations, config.tolerance)
            positive_scores = model.decision_function(pos_flat[held])
            held_null_flat = null_flat[held].reshape(-1, null_flat.shape[-1])
            held_null_scores = model.decision_function(held_null_flat)
            threshold = empirical_threshold(model.decision_function(train_null), config.target_fpr)
            detected = positive_scores > threshold
            metric_rows.append({"participant": participant, "method": method,
                                "sensitivity": float(detected.mean()),
                                "fpr": float(np.mean(held_null_scores > threshold)), "threshold": threshold})
            for condition in range(config.conditions):
                score_rows.append({"participant": participant, "condition": condition + 1, "method": method,
                                   "score": float(positive_scores[condition]), "threshold": threshold,
                                   "detected": bool(detected[condition])})
            model_rows.append({"heldout": participant, "method": method, "locked_c": config.locked_c,
                               "coefficients": json.dumps(model.coefficients.tolist()),
                               "intercept": model.intercept, "converged": model.converged})

    outputs = [checkpoint]
    for name, frame in (("hilbert_condition_scores.csv", pd.DataFrame(score_rows)),
                        ("hilbert_condition_metrics.csv", pd.DataFrame(metric_rows)),
                        ("hilbert_condition_model_records.csv", pd.DataFrame(model_rows))):
        path = context.table_dir / name; frame.to_csv(path, index=False); outputs.append(path)
    summary = {"methods": list(METHODS), "unit": "participant_condition", "locked_c": config.locked_c,
               "endpoint": "static_single_look_k10"}
    report = context.run_dir / "hilbert_condition_benchmark.json"
    atomic_write_json(report, summary); outputs.append(report)
    context.progress(stage, len(participants) + 1, len(participants) + 1, "condition benchmark")
    context.finish_stage(stage, outputs, summary)
    return outputs, summary
