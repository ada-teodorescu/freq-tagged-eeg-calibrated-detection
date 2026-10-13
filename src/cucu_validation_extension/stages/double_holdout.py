from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from cucu_advanced_study.algorithms import calibrate_pocock_boundaries, first_crossings
from cucu_algorithm_study.algorithms import fit_monotone_logistic
from cucu_article.runtime import atomic_write_json
from cucu_detection.evaluation import empirical_threshold
from ..algorithms import equal_weight_score
from ..runtime import RunContext
from .benchmarks import BASELINE_NAMES, static_bundle_path


def _fit(pos: np.ndarray, null: np.ndarray, config):
    values = np.concatenate((pos, null)); labels = np.concatenate((np.ones(len(pos), np.int8), np.zeros(len(null), np.int8)))
    return fit_monotone_logistic(values, labels, config.locked_c, config.maximum_iterations, config.tolerance)


def _training_arrays(positive: np.ndarray, null: np.ndarray, held_p: int, held_c: int, look: int | None = None):
    p_mask = np.arange(positive.shape[0]) != held_p; c_mask = np.arange(positive.shape[1]) != held_c
    if look is None:
        train_pos = positive[np.ix_(p_mask, c_mask)].reshape(-1, positive.shape[-1])
        train_null = null[np.ix_(p_mask, c_mask)].reshape(-1, null.shape[-1])
    else:
        train_pos = positive[np.ix_(p_mask, c_mask, [look])].reshape(-1, positive.shape[-1])
        train_null = null[np.ix_(p_mask, c_mask, np.arange(null.shape[2]), [look])].reshape(-1, null.shape[-1])
    return train_pos, train_null


def _static_fold(method: str, detector: str, positive: np.ndarray, null: np.ndarray,
                 held_p: int, held_c: int, participant: str, config) -> dict:
    train_pos, train_null = _training_arrays(positive, null, held_p, held_c)
    if detector == "learned":
        model = _fit(train_pos, train_null, config); train_scores = model.decision_function(train_null)
        score = float(model.decision_function(positive[held_p, held_c:held_c + 1])[0])
        held_null = model.decision_function(null[held_p, held_c])
    else:
        train_scores = equal_weight_score(train_null, 4); score = float(equal_weight_score(positive[held_p, held_c], 4))
        held_null = equal_weight_score(null[held_p, held_c], 4)
    threshold = empirical_threshold(train_scores, config.target_fpr)
    return {"participant": participant, "condition": held_c + 1, "method": method, "detector": detector,
            "static_score": score, "static_threshold": threshold, "static_detected": bool(score > threshold),
            "static_fpr": float(np.mean(held_null > threshold))}


def _sequential_fold(method: str, detector: str, positive: np.ndarray, null: np.ndarray,
                     held_p: int, held_c: int, participant: str, config) -> dict:
    looks = len(config.k_values); scores = np.empty(looks); held_null = np.empty((config.surrogates, looks))
    training_paths = np.empty(((positive.shape[0] - 1) * (positive.shape[1] - 1) * config.surrogates, looks))
    for look in range(looks):
        train_pos, train_null = _training_arrays(positive, null, held_p, held_c, look)
        if detector == "learned":
            model = _fit(train_pos, train_null, config)
            training_paths[:, look] = model.decision_function(train_null)
            scores[look] = model.decision_function(positive[held_p, held_c, look:look + 1])[0]
            held_null[:, look] = model.decision_function(null[held_p, held_c, :, look])
        else:
            training_paths[:, look] = equal_weight_score(train_null, 4)
            scores[look] = equal_weight_score(positive[held_p, held_c, look], 4)
            held_null[:, look] = equal_weight_score(null[held_p, held_c, :, look], 4)
    boundaries = calibrate_pocock_boundaries(training_paths, config.target_fpr).boundaries
    first = int(first_crossings(scores[None, :], boundaries, config.k_values)[0])
    null_first = first_crossings(held_null, boundaries, config.k_values)
    row = {"participant": participant, "condition": held_c + 1, "method": method, "detector": detector,
           "first_detection_k": first, "sequential_detected": bool(first <= 10), "rmst_k": first,
           "sequential_fpr": float(np.mean(null_first <= 10))}
    row.update({f"score_k{k}": float(scores[look]) for look, k in enumerate(config.k_values)})
    row.update({f"boundary_k{k}": float(boundaries[look]) for look, k in enumerate(config.k_values)})
    row.update({f"detected_by_k{k}": bool(np.any(scores[:look + 1] > boundaries[:look + 1])) for look, k in enumerate(config.k_values)})
    return row


def _sequential_bundle(context: RunContext, method: str) -> tuple[np.ndarray, np.ndarray]:
    path = context.config.baseline_confirmatory_run / "checkpoints/03_hilbert_sequential/trajectories.npz"
    key = BASELINE_NAMES[method]
    with np.load(path, allow_pickle=False) as archive:
        positive = archive[f"{key}_pos"].reshape(25, 15, len(context.config.k_values), -1)
        null = archive[f"{key}_null"].reshape(25, 15, 200, len(context.config.k_values), -1)
    return positive, null


def run(context: RunContext, force: bool = False) -> tuple[list[Path], dict]:
    del force
    stage, config = "03_double_holdout", context.config
    participants = json.loads((context.run_dir / "design_lock.json").read_text(encoding="utf-8"))["participants"]
    methods = (("H1_Hann4s", "learned"), ("H2_Hilbert4s", "learned"), ("H2_Hilbert4s", "simple_equal_4h"))
    total = len(participants) * config.conditions * len(methods) * 2; context.start_stage(stage, total)
    static_rows, sequential_rows, completed = [], [], 0
    for method, detector in methods:
        with np.load(static_bundle_path(context, method), allow_pickle=False) as archive:
            static_positive, static_null = archive["positive"], archive["null"]
        seq_positive, seq_null = _sequential_bundle(context, method)
        for held_p, participant in enumerate(participants):
            for held_c in range(config.conditions):
                static_rows.append(_static_fold(method, detector, static_positive, static_null,
                                                held_p, held_c, participant, config))
                completed += 1; context.progress(stage, completed, total, f"{method} {detector} {participant} C{held_c + 1} static")
                sequential_rows.append(_sequential_fold(method, detector, seq_positive, seq_null,
                                                        held_p, held_c, participant, config))
                completed += 1; context.progress(stage, completed, total, f"{method} {detector} {participant} C{held_c + 1} sequential")
    static_path = context.table_dir / "double_holdout_static.csv"; pd.DataFrame(static_rows).to_csv(static_path, index=False)
    sequential_path = context.table_dir / "double_holdout_sequential.csv"; pd.DataFrame(sequential_rows).to_csv(sequential_path, index=False)
    summary = {"folds_per_detector": len(participants) * config.conditions,
               "training_contract": "exclude held participant and held condition from fitting and null calibration"}
    report = context.run_dir / "double_holdout_summary.json"; atomic_write_json(report, summary)
    outputs = [static_path, sequential_path, report]; context.finish_stage(stage, outputs, summary); return outputs, summary
