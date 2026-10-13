from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from cucu_accuracy_study.features import region_indices
from cucu_advanced_study.algorithms import calibrate_pocock_boundaries, first_crossings
from cucu_algorithm_study.algorithms import fit_monotone_logistic
from cucu_article.runtime import atomic_save_npz, atomic_write_json
from cucu_detection.evaluation import empirical_threshold
from ..algorithms import equal_weight_score
from ..features import condition_features
from ..runtime import RunContext
from .estimators import coefficient_path


BASELINE_NAMES = {"H1_Hann4s": "H1_Hann4s_Regional4H", "H2_Hilbert4s": "H2_Hilbert4s_Regional4H"}
STATIC_METHODS = ("H1_Hann4s", "H2_Hilbert4s", "M1_RectLS4s", "M2_FilteredLS4s",
                  "S_HilbertBW0p25", "S_HilbertBW1p0", "S_HilbertCrop0p25", "S_HilbertCrop0p75",
                  "S_CausalDemod")


def static_bundle_path(context: RunContext, method: str) -> Path:
    return context.checkpoint_dir / "02_benchmarks" / f"static_{method}.npz"


def _baseline_static(context: RunContext, method: str) -> tuple[np.ndarray, np.ndarray]:
    path = context.config.baseline_confirmatory_run / "checkpoints/02_hilbert_condition/condition_features.npz"
    key = BASELINE_NAMES[method]
    with np.load(path, allow_pickle=False) as archive:
        return archive[f"{key}_pos"].reshape(25, 15, -1), archive[f"{key}_null"].reshape(25, 15, 200, -1)


def _make_static_bundle(context: RunContext, method: str, participants: list[str],
                        regions: tuple[np.ndarray, ...], force: bool) -> tuple[np.ndarray, np.ndarray, Path]:
    destination = static_bundle_path(context, method)
    if context.checkpoint_valid(destination) and not force:
        with np.load(destination, allow_pickle=False) as archive: return archive["positive"], archive["null"], destination
    if method in BASELINE_NAMES:
        positive, null = _baseline_static(context, method)
    else:
        positives, nulls = [], []
        for participant_index, participant in enumerate(participants):
            with np.load(coefficient_path(context, participant), allow_pickle=False) as archive:
                pos, nul = condition_features(archive[method], context.config, regions, participant_index,
                                              (context.config.trials_per_stream,))
            positives.append(pos[:, 0]); nulls.append(nul[:, :, 0])
        positive, null = np.stack(positives), np.stack(nulls)
    atomic_save_npz(destination, positive=positive, null=null, method=np.asarray(method))
    context.mark_checkpoint(destination, {"method": method, "unit": "participant_condition"})
    return positive, null, destination


def _fit(pos: np.ndarray, null: np.ndarray, config):
    values = np.concatenate((pos, null)); labels = np.concatenate((np.ones(len(pos), np.int8), np.zeros(len(null), np.int8)))
    return fit_monotone_logistic(values, labels, config.locked_c, config.maximum_iterations, config.tolerance)


def _learned_static_rows(method: str, positive: np.ndarray, null: np.ndarray,
                         participants: list[str], config) -> tuple[list[dict], list[dict]]:
    score_rows, model_rows = [], []
    for held, participant in enumerate(participants):
        training = np.arange(len(participants)) != held
        train_pos = positive[training].reshape(-1, positive.shape[-1]); train_null = null[training].reshape(-1, null.shape[-1])
        model = _fit(train_pos, train_null, config); threshold = empirical_threshold(model.decision_function(train_null), config.target_fpr)
        scores = model.decision_function(positive[held]); held_null = model.decision_function(null[held].reshape(-1, null.shape[-1]))
        for condition, score in enumerate(scores, 1):
            score_rows.append({"participant": participant, "condition": condition, "method": method,
                               "detector": "learned", "score": float(score), "threshold": threshold,
                               "detected": bool(score > threshold), "heldout_fpr": float(np.mean(held_null > threshold))})
        model_rows.append({"participant": participant, "method": method, "coefficients": json.dumps(model.coefficients.tolist()),
                           "intercept": model.intercept, "threshold": threshold, "converged": model.converged})
    return score_rows, model_rows


def _simple_static_rows(positive: np.ndarray, null: np.ndarray, participants: list[str], config,
                        harmonics: int, name: str) -> list[dict]:
    rows = []
    for held, participant in enumerate(participants):
        training = np.arange(len(participants)) != held
        train_null = equal_weight_score(null[training], harmonics).reshape(-1)
        threshold = empirical_threshold(train_null, config.target_fpr)
        scores = equal_weight_score(positive[held], harmonics); held_null = equal_weight_score(null[held], harmonics).reshape(-1)
        for condition, score in enumerate(scores, 1):
            rows.append({"participant": participant, "condition": condition, "method": "H2_Hilbert4s",
                         "detector": name, "score": float(score), "threshold": threshold,
                         "detected": bool(score > threshold), "heldout_fpr": float(np.mean(held_null > threshold))})
    return rows


def _sequential_bundle(context: RunContext, method: str) -> tuple[np.ndarray, np.ndarray]:
    path = context.config.baseline_confirmatory_run / "checkpoints/03_hilbert_sequential/trajectories.npz"
    key = BASELINE_NAMES[method]
    with np.load(path, allow_pickle=False) as archive:
        positive = archive[f"{key}_pos"].reshape(25, 15, len(context.config.k_values), -1)
        null = archive[f"{key}_null"].reshape(25, 15, 200, len(context.config.k_values), -1)
    return positive, null


def _sequential_rows(context: RunContext, method: str, positive: np.ndarray, null: np.ndarray,
                     participants: list[str], detector: str) -> tuple[list[dict], list[dict]]:
    config = context.config; rows, metrics = [], []
    for held, participant in enumerate(participants):
        training = np.arange(len(participants)) != held
        held_scores = np.empty((config.conditions, len(config.k_values)))
        held_null_scores = np.empty((config.conditions, config.surrogates, len(config.k_values)))
        calibration = np.empty((int(training.sum()) * config.conditions * config.surrogates, len(config.k_values)))
        for look in range(len(config.k_values)):
            if detector == "learned":
                train_pos = positive[training, :, look].reshape(-1, positive.shape[-1])
                train_null = null[training, :, :, look].reshape(-1, null.shape[-1])
                model = _fit(train_pos, train_null, config)
                held_scores[:, look] = model.decision_function(positive[held, :, look])
                held_null_scores[:, :, look] = model.decision_function(null[held, :, :, look].reshape(-1, null.shape[-1])).reshape(config.conditions, config.surrogates)
                calibration[:, look] = model.decision_function(train_null)
            else:
                held_scores[:, look] = equal_weight_score(positive[held, :, look], 4)
                held_null_scores[:, :, look] = equal_weight_score(null[held, :, :, look], 4)
                calibration[:, look] = equal_weight_score(null[training, :, :, look], 4).reshape(-1)
        calibrated = calibrate_pocock_boundaries(calibration, config.target_fpr); boundaries = calibrated.boundaries
        first = first_crossings(held_scores, boundaries, config.k_values)
        null_first = first_crossings(held_null_scores.reshape(-1, len(config.k_values)), boundaries, config.k_values)
        crossed = np.maximum.accumulate(held_scores > boundaries[None, :], axis=1)
        metrics.append({"participant": participant, "method": method, "detector": detector,
                        "detection": float(np.mean(first <= 10)), "rmst_k": float(np.mean(first)),
                        "familywise_fpr": float(np.mean(null_first <= 10))})
        for condition in range(config.conditions):
            row = {"participant": participant, "condition": condition + 1, "method": method,
                   "detector": detector, "first_detection_k": int(first[condition]), "detected": bool(first[condition] <= 10)}
            row.update({f"score_k{k}": float(held_scores[condition, look]) for look, k in enumerate(config.k_values)})
            row.update({f"boundary_k{k}": float(boundaries[look]) for look, k in enumerate(config.k_values)})
            row.update({f"detected_by_k{k}": bool(crossed[condition, look]) for look, k in enumerate(config.k_values)})
            rows.append(row)
    return rows, metrics


def run(context: RunContext, force: bool = False) -> tuple[list[Path], dict]:
    stage, config = "02_benchmarks", context.config
    design = json.loads((context.run_dir / "design_lock.json").read_text(encoding="utf-8")); participants = design["participants"]
    with np.load(coefficient_path(context, participants[0]), allow_pickle=False) as sample:
        regions = region_indices(tuple(sample["channel_labels"].astype(str)), config.regions)
    context.start_stage(stage, len(STATIC_METHODS) + 2); outputs, static_rows, model_rows = [], [], []
    h2_bundle = None
    for index, method in enumerate(STATIC_METHODS, 1):
        positive, null, path = _make_static_bundle(context, method, participants, regions, force); outputs.append(path)
        rows, models = _learned_static_rows(method, positive, null, participants, config); static_rows += rows; model_rows += models
        if method == "H2_Hilbert4s": h2_bundle = (positive, null)
        context.progress(stage, index, len(STATIC_METHODS) + 2, method)
    assert h2_bundle is not None
    for harmonics, name in ((4, "simple_equal_4h"), (2, "simple_equal_4_8hz"), (1, "simple_equal_4hz")):
        static_rows += _simple_static_rows(*h2_bundle, participants, config, harmonics, name)
    static_path = context.table_dir / "static_scores.csv"; pd.DataFrame(static_rows).to_csv(static_path, index=False)
    models_path = context.table_dir / "static_model_records.csv"; pd.DataFrame(model_rows).to_csv(models_path, index=False)
    outputs += [static_path, models_path]; context.progress(stage, len(STATIC_METHODS) + 1, len(STATIC_METHODS) + 2, "static scores")
    sequential_rows, metric_rows = [], []
    for method in ("H1_Hann4s", "H2_Hilbert4s"):
        positive, null = _sequential_bundle(context, method)
        rows, metrics = _sequential_rows(context, method, positive, null, participants, "learned")
        sequential_rows += rows; metric_rows += metrics
        if method == "H2_Hilbert4s":
            rows, metrics = _sequential_rows(context, method, positive, null, participants, "simple_equal_4h")
            sequential_rows += rows; metric_rows += metrics
    sequential_path = context.table_dir / "sequential_scores.csv"; pd.DataFrame(sequential_rows).to_csv(sequential_path, index=False)
    metrics_path = context.table_dir / "sequential_participant_metrics.csv"; pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    outputs += [sequential_path, metrics_path]; context.progress(stage, len(STATIC_METHODS) + 2, len(STATIC_METHODS) + 2, "sequential scores")
    summary = {"static_methods": list(STATIC_METHODS), "sequential_methods": ["H1_Hann4s", "H2_Hilbert4s", "simple_equal_4h"]}
    report = context.run_dir / "benchmark_summary.json"; atomic_write_json(report, summary); outputs.append(report)
    context.finish_stage(stage, outputs, summary); return outputs, summary
