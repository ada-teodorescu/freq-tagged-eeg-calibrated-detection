from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ..evaluation import crossed_bootstrap_gain, holm_adjust, sign_flip_test, trials_to_target
from ..runtime import RunContext, atomic_write_json


def _method_summary(table: pd.DataFrame, target: float) -> pd.DataFrame:
    grouping = ["branch", "estimator", "detector", "spatial", "k"]
    aggregate = table.groupby(grouping, as_index=False).agg(
        detection_rate=("detection_rate", "mean"), auc=("auc", "mean"),
        mean_score=("mean_score", "mean"),
    )
    if "heldout_fpr" in table:
        fpr = table.groupby(grouping, as_index=False)["heldout_fpr"].mean()
        aggregate = aggregate.merge(fpr, on=grouping, how="left")
    rows = []
    for key, frame in aggregate.groupby(["branch", "estimator", "detector", "spatial"]):
        frame = frame.sort_values("k")
        k80 = trials_to_target(frame["k"].to_numpy(), frame["detection_rate"].to_numpy(), target)
        for record in frame.to_dict(orient="records"):
            record["trials_to_80_percent"] = k80
            rows.append(record)
    return pd.DataFrame(rows)


def _consistency_rows(table: pd.DataFrame, condition_column: str) -> list[dict]:
    rows = []
    keys = ["branch", "estimator", "detector", "spatial"]
    selected = table[table.k == 10]
    for key, frame in selected.groupby(keys):
        matrix = frame.groupby(["participant", condition_column]).detection_rate.mean().unstack()
        participants = matrix.index.to_numpy()
        conditions = matrix.columns.to_numpy()
        condition_left = matrix.loc[participants[::2]].mean(axis=0)
        condition_right = matrix.loc[participants[1::2]].mean(axis=0)
        participant_left = matrix.loc[:, conditions[::2]].mean(axis=1)
        participant_right = matrix.loc[:, conditions[1::2]].mean(axis=1)
        rows.append({"branch": key[0], "estimator": key[1], "detector": key[2], "spatial": key[3],
                     "k": 10,
                     "condition_order_split_half_spearman": stats.spearmanr(condition_left, condition_right).statistic,
                     "participant_order_split_half_spearman": stats.spearmanr(participant_left, participant_right).statistic})
    return rows


def run(context: RunContext, force: bool = False) -> tuple[list, dict]:
    stage = "06_outcomes"
    context.start_stage(stage, 5)
    stored = pd.read_csv(context.table_dir / "detector_scores_long.csv")
    calibration = pd.read_csv(context.table_dir / "detector_calibration_long.csv")
    public = pd.read_csv(context.table_dir / "public_spatial_scores_long.csv")
    context.progress(stage, 1, 5, "load held-out scores")
    stored_summary = _method_summary(stored, context.config.target_detection)
    public_summary = _method_summary(public, context.config.target_detection)
    summary = pd.concat((stored_summary, public_summary), ignore_index=True)
    stored_runtime = json.loads((context.run_dir / "stored_detector_screen.json").read_text(encoding="utf-8"))
    public_estimator_runtime = json.loads((context.run_dir / "public_estimator_cache.json").read_text(encoding="utf-8"))
    public_spatial_runtime = json.loads((context.run_dir / "public_spatial_screen.json").read_text(encoding="utf-8"))
    summary["pipeline_runtime_seconds"] = np.where(
        summary.branch == "stored_matlab", stored_runtime["runtime_seconds"],
        public_estimator_runtime["runtime_seconds"] + public_spatial_runtime["runtime_seconds"],
    )
    summary["pipeline_peak_python_memory_bytes"] = np.where(
        summary.branch == "stored_matlab", stored_runtime["peak_python_memory_bytes"],
        max(public_estimator_runtime["peak_python_memory_bytes"], public_spatial_runtime["peak_python_memory_bytes"]),
    )
    summary_path = context.table_dir / "trials_to_detect_auc_runtime.csv"
    summary.to_csv(summary_path, index=False)
    context.progress(stage, 2, 5, "trial efficiency")

    curves = stored.groupby(["participant", "condition", "detector", "k"], as_index=False)["detection_rate"].mean()
    baseline_curve = curves[curves.detector == "D1"]
    candidate_curve = curves[curves.detector == "D8"]
    mean_baseline = baseline_curve.groupby("k").detection_rate.mean().sort_index()
    mean_candidate = candidate_curve.groupby("k").detection_rate.mean().sort_index()
    baseline_k = trials_to_target(mean_baseline.index.to_numpy(), mean_baseline.to_numpy(), context.config.target_detection)
    candidate_k = trials_to_target(mean_candidate.index.to_numpy(), mean_candidate.to_numpy(), context.config.target_detection)
    comparison_k = candidate_k if candidate_k is not None else max(context.config.k_values)
    baseline_matrix = baseline_curve[baseline_curve.k == comparison_k].pivot(index="participant", columns="condition", values="detection_rate").to_numpy()
    candidate_matrix = candidate_curve[candidate_curve.k == comparison_k].pivot(index="participant", columns="condition", values="detection_rate").to_numpy()
    gains = candidate_matrix - baseline_matrix
    bootstrap_ci = crossed_bootstrap_gain(gains, context.config.bootstrap_iterations, context.config.seed)
    condition_gains = gains.mean(axis=0)
    sign_p = sign_flip_test(condition_gains, context.config.permutation_iterations, context.config.seed)
    candidate_fpr = calibration[(calibration.detector == "D8") & (calibration.k == comparison_k)].heldout_fpr.mean()
    context.progress(stage, 3, 5, "primary comparison")

    secondary_rows = []
    p_values = []
    baseline_k10 = curves[(curves.detector == "D1") & (curves.k == 10)].groupby(["participant", "condition"]).detection_rate.mean()
    for detector in sorted(set(curves.detector) - {"D1"}):
        method = curves[(curves.detector == detector) & (curves.k == 10)].groupby(["participant", "condition"]).detection_rate.mean()
        differences = (method - baseline_k10).dropna().to_numpy()
        test = stats.ttest_1samp(differences, 0.0)
        secondary_rows.append({"detector": detector, "k": 10, "mean_gain": float(np.mean(differences)),
                               "t": float(test.statistic), "p_uncorrected": float(test.pvalue)})
        p_values.append(test.pvalue)
    adjusted = holm_adjust(np.asarray(p_values))
    for row, value in zip(secondary_rows, adjusted, strict=True): row["p_holm"] = float(value)
    secondary_path = context.table_dir / "secondary_method_comparisons.csv"
    pd.DataFrame(secondary_rows).to_csv(secondary_path, index=False)
    consistency_path = context.table_dir / "participant_condition_consistency.csv"
    consistency = _consistency_rows(stored, "condition") + _consistency_rows(public, "outer_condition")
    pd.DataFrame(consistency).to_csv(consistency_path, index=False)
    context.progress(stage, 4, 5, "secondary comparisons")

    primary_success = bool(
        baseline_k is not None and candidate_k is not None and candidate_k <= baseline_k - 1
        and 0.04 <= candidate_fpr <= 0.06 and bootstrap_ci[0] > 0 and sign_p < 0.05
    )
    report = {
        "primary_baseline": context.config.primary_baseline,
        "primary_candidate": context.config.primary_candidate,
        "baseline_trials_to_80": baseline_k, "candidate_trials_to_80": candidate_k,
        "comparison_k": comparison_k, "candidate_heldout_fpr": float(candidate_fpr),
        "crossed_bootstrap_gain_ci95": list(bootstrap_ci),
        "condition_sign_flip_p": sign_p, "primary_candidate_succeeds": primary_success,
        "d1_d2_d3_interpretation": "At fixed K, D2 and D3 are monotonic transformations of D1; only calibration and bias differ.",
        "branches_numerically_merged": False,
    }
    report_path = context.run_dir / "benchmark_outcomes.json"
    atomic_write_json(report_path, report)
    context.progress(stage, 5, 5, "claim criteria")
    outputs = [summary_path, secondary_path, consistency_path, report_path]
    context.finish_stage(stage, outputs, report)
    return outputs, report
