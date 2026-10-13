from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from eeg_core.runtime import atomic_write_json
from eeg_detection.evaluation import crossed_bootstrap_gain, holm_adjust, sign_flip_test

from ..runtime import RunContext
from .stream import METHODS


def _cluster_ci(values: np.ndarray, iterations: int, seed: int) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations)
    for index in range(iterations):
        draws[index] = array[rng.integers(0, len(array), len(array))].mean()
    return tuple(float(value) for value in np.quantile(draws, (0.025, 0.975)))


def _detection_matrix(frame: pd.DataFrame, method: str, calibration: str = "population") -> np.ndarray:
    selected = frame[(frame["method"] == method) & (frame["calibration"] == calibration)].copy()
    selected = selected.sort_values(["participant", "condition", "stream_within_condition"])
    participants = selected["participant"].nunique(); conditions = selected["condition"].nunique()
    return selected["detected"].astype(float).to_numpy().reshape(participants, conditions, 3).mean(axis=2)


def _comparison(frame: pd.DataFrame, candidate: str, comparator: str, config, salt: int) -> dict:
    gains = _detection_matrix(frame, candidate) - _detection_matrix(frame, comparator)
    low, high = crossed_bootstrap_gain(gains, config.bootstrap_iterations, config.seed + salt)
    return {"candidate": candidate, "comparator": comparator, "mean_detection_gain": float(gains.mean()),
            "crossed_bootstrap_ci95": [low, high],
            "condition_sign_flip_p": sign_flip_test(gains.mean(axis=0), config.sign_flip_iterations,
                                                     config.seed + 100 + salt)}


def run(context: RunContext, force: bool = False) -> tuple[list[Path], dict]:
    del force
    stage, config = "06_statistics", context.config
    context.start_stage(stage, 5)
    sequential = pd.read_csv(context.table_dir / "sequential_scores.csv")
    sequential = sequential[np.isclose(sequential["duration_seconds"], config.primary_duration_seconds)]
    calibration = pd.read_csv(context.table_dir / "calibration_comparisons.csv")
    calibration = calibration[np.isclose(calibration["duration_seconds"], config.primary_duration_seconds)]
    primary = _comparison(sequential, METHODS[1], METHODS[0], config, 1)
    context.progress(stage, 1, 5, "primary crossed bootstrap")
    secondary = [_comparison(sequential, method, METHODS[1], config, index + 10)
                 for index, method in enumerate(METHODS[2:])]
    adjusted = holm_adjust(np.asarray([row["condition_sign_flip_p"] for row in secondary]))
    for row, value in zip(secondary, adjusted, strict=True):
        row["holm_adjusted_p"] = float(value)
    context.progress(stage, 2, 5, "secondary multiplicity control")

    fpr_summary = {}
    for method in METHODS:
        rows = calibration[(calibration["method"] == method) & (calibration["calibration"] == "population")]
        participant_fpr = rows.groupby("participant")["familywise_fpr"].mean().to_numpy()
        participant_sensitivity = rows.groupby("participant")["sensitivity"].mean().to_numpy()
        fpr_summary[method] = {"heldout_familywise_fpr": float(participant_fpr.mean()),
                               "participant_cluster_ci95": list(_cluster_ci(
                                   participant_fpr, config.bootstrap_iterations, config.seed + 30)),
                               "sensitivity": float(participant_sensitivity.mean())}
    primary_fpr = fpr_summary[METHODS[1]]["heldout_familywise_fpr"]
    primary["heldout_surrogate_familywise_fpr"] = primary_fpr
    primary["heldout_surrogate_fpr_ci95"] = fpr_summary[METHODS[1]]["participant_cluster_ci95"]
    primary["gain_ci_above_zero"] = bool(primary["crossed_bootstrap_ci95"][0] > 0)
    primary["fpr_in_locked_interval"] = bool(config.fpr_acceptance_low <= primary_fpr <= config.fpr_acceptance_high)
    primary["success"] = bool(primary["gain_ci_above_zero"] and primary["fpr_in_locked_interval"])
    context.progress(stage, 3, 5, "calibration uncertainty")

    shrinkage = {}
    for method in METHODS:
        pop = calibration[(calibration["method"] == method) & (calibration["calibration"] == "population")]
        local = calibration[(calibration["method"] == method) & (calibration["calibration"] == "shrinkage")]
        pop_fpr = pop.set_index("participant")["familywise_fpr"]
        pop_sensitivity = pop.set_index("participant")["sensitivity"]
        local_fpr = local.groupby("participant")["familywise_fpr"].mean()
        local_sensitivity = local.groupby("participant")["sensitivity"].mean()
        pop_rmse = float(np.sqrt(np.mean((pop_fpr - config.target_fpr) ** 2)))
        local_rmse = float(np.sqrt(np.mean((local_fpr - config.target_fpr) ** 2)))
        sensitivity_change = float(local_sensitivity.mean() - pop_sensitivity.mean())
        shrinkage[method] = {"population_participant_fpr_rmse": pop_rmse,
                             "shrinkage_participant_fpr_rmse": local_rmse,
                             "sensitivity_change": sensitivity_change,
                             "success": bool(local_rmse < pop_rmse and
                                             sensitivity_change >= -config.sensitivity_noninferiority_margin)}

    condition_scores = pd.read_csv(context.table_dir / "condition_scores.csv")
    condition_summary = condition_scores.groupby("method")["detected"].mean().to_dict()
    negative_status = json.loads((context.run_dir / "negative_control_status.json").read_text(encoding="utf-8"))
    negative_summary_path = context.table_dir / "negative_control_summary.csv"
    negative_summary = pd.read_csv(negative_summary_path).to_dict("records") if negative_summary_path.stat().st_size else []
    context.progress(stage, 4, 5, "secondary endpoints and controls")
    conclusions = {
        "analysis_id": context.analysis_id,
        "primary": primary,
        "secondary_stream_comparisons": secondary,
        "stream_population_performance": fpr_summary,
        "shrinkage_calibration": shrinkage,
        "condition_sensitivity": {str(key): float(value) for key, value in condition_summary.items()},
        "negative_controls": negative_summary,
        "external_negative_status": negative_status["external_negative_status"],
        "operational_fpr_claim_allowed": negative_status["operational_fpr_claim_allowed"],
        "interpretation_guardrail": "Off-target controls are pseudo-negatives; operational FPR requires genuine external negatives.",
    }
    conclusion_path = context.run_dir / "advanced_conclusions.json"
    atomic_write_json(conclusion_path, conclusions)
    comparison_rows = [primary, *secondary]
    comparison_path = context.table_dir / "primary_comparisons.csv"
    pd.json_normalize(comparison_rows).to_csv(comparison_path, index=False)
    calibration_path = context.table_dir / "calibration_success.csv"
    pd.DataFrame([{"method": method, **values} for method, values in shrinkage.items()]).to_csv(calibration_path, index=False)
    context.progress(stage, 5, 5, "conclusions locked")
    outputs = [conclusion_path, comparison_path, calibration_path]
    context.finish_stage(stage, outputs, {"primary_success": primary["success"]})
    return outputs, conclusions
