from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cucu_accuracy_study.features import region_indices
from cucu_advanced_study.algorithms import first_crossings, phase_trajectories
from cucu_algorithm_study.algorithms import MonotoneModel
from cucu_article.algorithms import Standardizer
from ..runtime import RunContext
from .deployment import model_path
from .features import feature_path


def _model(coefficients: np.ndarray, intercept: float, mean: np.ndarray, scale: np.ndarray,
           c_value: float) -> MonotoneModel:
    return MonotoneModel(np.asarray(coefficients, float), float(intercept),
                         Standardizer(np.asarray(mean, float), np.asarray(scale, float)), c_value, True)


def condition_trajectories(coefficients: np.ndarray, k_values: tuple[int, ...], regions: tuple[np.ndarray, ...]) -> np.ndarray:
    # unit x stream x trial x channel x harmonic -> unit x K x (region*harmonic)
    units = []
    for unit in coefficients:
        _, regional = phase_trajectories(unit, k_values, regions)
        averaged = regional.mean(axis=0)
        scale = 2.0 * np.asarray(k_values, float)[:, None, None]
        units.append((scale * averaged).reshape(len(k_values), -1))
    return np.stack(units)


def run(context: RunContext, force: bool = False) -> tuple[list[Path], dict]:
    del force
    stage, config = "06_external_scores", context.config
    membership = pd.read_csv(context.table_dir / "negative_unit_membership.csv")
    groups = list(membership.groupby(["participant", "task"], sort=True))
    context.start_stage(stage, len(groups))
    with np.load(model_path(context), allow_pickle=False) as archive:
        static_model = _model(archive["static_coefficients"], float(archive["static_intercept"]),
                              archive["static_mean"], archive["static_scale"], config.locked_c)
        static_threshold = float(archive["static_threshold"])
        sequential_models = [_model(coef, intercept, mean, scale, config.locked_c)
                             for coef, intercept, mean, scale in zip(
                                 archive["sequential_coefficients"], archive["sequential_intercepts"],
                                 archive["sequential_means"], archive["sequential_scales"], strict=True)]
        boundaries = archive["sequential_boundaries"].astype(float)
    regions = region_indices(config.target_channels, config.regions)
    rows = []
    for index, ((participant, task), _) in enumerate(groups, 1):
        with np.load(feature_path(context, participant, task), allow_pickle=False) as archive:
            coefficients = archive["coefficients"]
        features = condition_trajectories(coefficients, config.k_values, regions)
        sequential_scores = np.column_stack([
            model.decision_function(features[:, look]) for look, model in enumerate(sequential_models)
        ])
        static_scores = static_model.decision_function(features[:, -1])
        first = first_crossings(sequential_scores, boundaries, config.k_values)
        for unit in range(len(features)):
            row = {"participant": participant, "task": task, "unit": unit + 1,
                   "static_score": float(static_scores[unit]), "static_threshold": static_threshold,
                   "static_false_positive": bool(static_scores[unit] > static_threshold),
                   "first_detection_k": int(first[unit]),
                   "sequential_false_positive": bool(first[unit] <= config.k_values[-1])}
            row.update({f"score_k{k}": float(sequential_scores[unit, look]) for look, k in enumerate(config.k_values)})
            row.update({f"boundary_k{k}": float(boundaries[look]) for look, k in enumerate(config.k_values)})
            rows.append(row)
        context.progress(stage, index, len(groups), f"{participant} {task}")
    path = context.table_dir / "external_negative_scores.csv"; pd.DataFrame(rows).to_csv(path, index=False)
    summary = {"units": len(rows), "participants": len(set(row["participant"] for row in rows)),
               "external_data_used_for_training_or_calibration": False}
    context.finish_stage(stage, [path], summary); return [path], summary
