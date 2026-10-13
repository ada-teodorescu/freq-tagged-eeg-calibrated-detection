from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from eeg_core.runtime import atomic_write_json
from eeg_detection.evaluation import crossed_bootstrap_gain, holm_adjust, sign_flip_test

from ..runtime import RunContext


def _condition_detection_matrix(frame: pd.DataFrame, method: str) -> np.ndarray:
    selected = frame[frame["method"] == method].sort_values(["participant", "condition"])
    participants = selected["participant"].nunique()
    conditions = selected["condition"].nunique()
    return selected["detected"].astype(float).to_numpy().reshape(participants, conditions)


def _stream_detection_matrix(frame: pd.DataFrame, method: str, calibration: str) -> np.ndarray:
    selected = frame[(frame["method"] == method) & (frame["calibration"] == calibration)].copy()
    selected = selected.sort_values(["participant", "condition", "stream_within_condition"])
    participants = selected["participant"].nunique()
    conditions = selected["condition"].nunique()
    return selected["detected"].astype(float).to_numpy().reshape(participants, conditions, 3).mean(axis=2)


def _comparison(candidate_matrix: np.ndarray, comparator_matrix: np.ndarray, candidate: str, comparator: str,
                config, salt: int) -> dict:
    gains = candidate_matrix - comparator_matrix
    low, high = crossed_bootstrap_gain(gains, config.bootstrap_iterations, config.seed + salt)
    return {"candidate": candidate, "comparator": comparator, "mean_detection_gain": float(gains.mean()),
            "crossed_bootstrap_ci95": [low, high],
            "condition_sign_flip_p": sign_flip_test(gains.mean(axis=0), config.sign_flip_iterations,
                                                     config.seed + 100 + salt)}


def run(context: RunContext, force: bool = False) -> tuple[list[Path], dict]:
    del force
    stage, config = "06_statistics", context.config
    context.start_stage(stage, 3)

    hilbert_scores = pd.read_csv(context.table_dir / "hilbert_condition_scores.csv")
    hilbert_metrics = pd.read_csv(context.table_dir / "hilbert_condition_metrics.csv")
    trca_scores = pd.read_csv(context.table_dir / "trca_sequential_scores.csv")
    trca_metrics = pd.read_csv(context.table_dir / "trca_sequential_metrics.csv")

    h1 = _condition_detection_matrix(hilbert_scores, "H1_Hann4s_Regional4H")
    h2 = _condition_detection_matrix(hilbert_scores, "H2_Hilbert4s_Regional4H")
    h3 = _condition_detection_matrix(hilbert_scores, "H3_Hann4p5s_Regional4H")
    trca_pop = _stream_detection_matrix(trca_scores, "TRCA_S2_Grouped", "population")
    s2_pop = _stream_detection_matrix(trca_scores, "S2_Equal4plus8", "population")
    s4_pop = _stream_detection_matrix(trca_scores, "S4_RegionalMonotone4plus8", "population")
    s4_shrink = _stream_detection_matrix(trca_scores, "S4_RegionalMonotone4plus8", "shrinkage")

    confirmatory = [
        _comparison(h2, h1, "H2_Hilbert4s_Regional4H", "H1_Hann4s_Regional4H", config, 1),
        _comparison(trca_pop, s2_pop, "TRCA_S2_Grouped", "S2_Equal4plus8", config, 2),
        _comparison(trca_pop, s4_pop, "TRCA_S2_Grouped", "S4_RegionalMonotone4plus8", config, 3),
    ]
    context.progress(stage, 1, 3, "confirmatory three-hypothesis family")

    adjusted = holm_adjust(np.asarray([row["condition_sign_flip_p"] for row in confirmatory]))
    candidate_fprs = (
        float(hilbert_metrics[hilbert_metrics["method"] == "H2_Hilbert4s_Regional4H"]["fpr"].mean()),
        float(trca_metrics[(trca_metrics["method"] == "TRCA_S2_Grouped") &
                           (trca_metrics["calibration"] == "population")]["familywise_fpr"].mean()),
        float(trca_metrics[(trca_metrics["method"] == "TRCA_S2_Grouped") &
                           (trca_metrics["calibration"] == "population")]["familywise_fpr"].mean()),
    )
    for row, value, candidate_fpr in zip(confirmatory, adjusted, candidate_fprs, strict=True):
        row["holm_adjusted_p"] = float(value)
        row["gain_ci_above_zero"] = bool(row["crossed_bootstrap_ci95"][0] > 0)
        row["heldout_candidate_fpr"] = candidate_fpr
        row["fpr_in_locked_interval"] = bool(
            config.fpr_acceptance_low <= candidate_fpr <= config.fpr_acceptance_high
        )
        row["confirmatory_pass"] = bool(
            row["gain_ci_above_zero"] and row["holm_adjusted_p"] < 0.05 and row["fpr_in_locked_interval"]
        )
    context.progress(stage, 2, 3, "Holm correction across the three-hypothesis family")

    secondary = [
        _comparison(h2, h3, "H2_Hilbert4s_Regional4H", "H3_Hann4p5s_Regional4H", config, 10),
        _comparison(trca_pop, s4_shrink, "TRCA_S2_Grouped", "S4_RegionalMonotone4plus8_shrinkage", config, 11),
    ]
    for row in secondary:
        row["note"] = "secondary; excluded from the confirmatory Holm family"

    h2_pass = confirmatory[0]["confirmatory_pass"]
    trca_pass_s2 = confirmatory[1]["confirmatory_pass"]
    trca_pass_s4 = confirmatory[2]["confirmatory_pass"]
    if trca_pass_s4:
        trca_classification = "new_stream_leader"
    elif trca_pass_s2:
        trca_classification = "useful_spatial_improvement"
    else:
        trca_classification = "no_confirmed_benefit"
    stop_and_report = not (h2_pass or trca_pass_s2)

    conclusions = {
        "analysis_id": context.analysis_id,
        "confirmatory_family": confirmatory,
        "secondary_comparisons": secondary,
        "hilbert_verdict": {"h2_beats_matched_window_h1": h2_pass},
        "trca_verdict": {"beats_s2": trca_pass_s2, "beats_s4": trca_pass_s4, "classification": trca_classification},
        "stop_and_report_gate_triggered": stop_and_report,
        "interpretation_guardrail": "Surrogate temporal-shift nulls calibrate familywise FPR; they are not genuine "
                                    "sham recordings, and operational FPR is not claimed beyond this study's own "
                                    "surrogate-based calibration.",
    }
    conclusion_path = context.run_dir / "confirmatory_conclusions.json"
    atomic_write_json(conclusion_path, conclusions)
    comparison_path = context.table_dir / "confirmatory_comparisons.csv"
    pd.json_normalize(confirmatory + secondary).to_csv(comparison_path, index=False)
    context.progress(stage, 3, 3, "conclusions locked")
    outputs = [conclusion_path, comparison_path]
    context.finish_stage(stage, outputs, {"stop_and_report_gate_triggered": stop_and_report})
    return outputs, conclusions
