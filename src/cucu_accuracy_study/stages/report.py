from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

from cucu_detection.evaluation import crossed_bootstrap_gain, sign_flip_test
from ..runtime import RunContext, atomic_write_json, atomic_write_text


def _comparison(frame: pd.DataFrame, candidate: str, baseline: str, config, salt: int) -> dict:
    pivot = frame.pivot(index=["participant_index", "condition"], columns="algorithm", values="detected")
    gains = (pivot[candidate] - pivot[baseline]).to_numpy().reshape(25, 15)
    low, high = crossed_bootstrap_gain(gains, config.bootstrap_iterations, config.seed + salt)
    return {
        "candidate": candidate, "baseline": baseline, "mean_sensitivity_gain": float(gains.mean()),
        "crossed_bootstrap_ci95": [low, high],
        "condition_sign_flip_p": sign_flip_test(
            gains.mean(axis=0), config.sign_flip_iterations, config.seed + 100 + salt,
        ),
        "participants_positive": int((gains.mean(axis=1) > 0).sum()),
        "conditions_positive": int((gains.mean(axis=0) > 0).sum()),
    }


def run(context: RunContext, force: bool = False) -> tuple[list[Path], dict]:
    del force
    stage = "04_report"
    config = context.config
    context.start_stage(stage, 5)
    scores = pd.read_csv(context.table_dir / "condition_scores.csv")
    pooled = scores[scores.surrogate_block == "pooled"]
    summary = pooled.groupby(["algorithm", "target_fpr"], as_index=False).agg(
        sensitivity=("detected", "mean"), heldout_fpr=("heldout_fpr", "mean"), auc=("auc", "mean"),
    )
    summary["specificity"] = 1 - summary.heldout_fpr
    summary["balanced_accuracy"] = (summary.sensitivity + summary.specificity) / 2
    summary["fpr_accepted"] = summary.heldout_fpr.between(
        config.fpr_acceptance_low, config.fpr_acceptance_high,
    ) | (summary.target_fpr != config.target_fpr)
    summary_path = context.table_dir / "accuracy_summary.csv"
    summary.to_csv(summary_path, index=False)
    context.progress(stage, 1, 5, "operating points")

    primary = pooled[np.isclose(pooled.target_fpr, config.target_fpr)]
    comparisons = []
    for salt, candidate in enumerate((
        "C2_Condition4plus8", "C3_Reliability4plus8",
        "C4_MonotoneHarmonic", "C5_RegionalMonotone",
    ), 1):
        comparisons.append(_comparison(primary, candidate, "C1_ConditionITPC4", config, salt))
    for salt, candidate in enumerate((
        "C3_Reliability4plus8", "C4_MonotoneHarmonic", "C5_RegionalMonotone",
    ), 20):
        comparisons.append(_comparison(primary, candidate, "C2_Condition4plus8", config, salt))
    comparison_path = context.table_dir / "primary_comparisons.csv"
    pd.DataFrame(comparisons).to_csv(comparison_path, index=False)
    context.progress(stage, 2, 5, "paired inference")

    primary_summary = summary[np.isclose(summary.target_fpr, config.target_fpr)].copy()
    primary_summary["target_reached"] = (
        (primary_summary.sensitivity >= config.target_sensitivity)
        & (primary_summary.heldout_fpr <= config.fpr_acceptance_high)
    )
    best = primary_summary.sort_values(["sensitivity", "auc"], ascending=False).iloc[0]
    target_reached = bool(primary_summary.target_reached.any())
    models = pd.read_csv(context.table_dir / "models.csv")
    model_summary = []
    feature_names = [f"{region}_{frequency:g}Hz" for region in config.region_names for frequency in config.harmonics_hz]
    for algorithm, group in models.groupby("algorithm"):
        coefficients = np.stack(group.coefficients.map(json.loads))
        normalized = coefficients / np.maximum(coefficients.sum(axis=1, keepdims=True), np.finfo(float).eps)
        names = [f"{frequency:g}Hz" for frequency in config.harmonics_hz] if algorithm == "C4_MonotoneHarmonic" else feature_names
        for index, name in enumerate(names):
            model_summary.append({
                "algorithm": algorithm, "feature": name,
                "mean_normalized_weight": float(normalized[:, index].mean()),
                "std_normalized_weight": float(normalized[:, index].std(ddof=1)),
            })
    model_summary_path = context.table_dir / "model_weight_summary.csv"
    pd.DataFrame(model_summary).to_csv(model_summary_path, index=False)
    context.progress(stage, 3, 5, "model interpretation")

    onset = pd.read_csv(context.table_dir / "sliding_window_scores.csv")
    onset_summary = onset.groupby([
        "algorithm", "window_start_seconds_after_stimulus", "window_seconds",
    ], as_index=False).agg(sensitivity=("detected", "mean"), heldout_fpr=("heldout_fpr", "mean"), auc=("auc", "mean"))
    onset_path = context.table_dir / "sliding_window_summary.csv"
    onset_summary.to_csv(onset_path, index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    figure, axis = plt.subplots(figsize=(5.4, 3.2))
    for algorithm, group in summary.groupby("algorithm"):
        group = group.sort_values("heldout_fpr")
        axis.plot(group.heldout_fpr, group.sensitivity, marker="o", label=algorithm.replace("_", " "))
    axis.axhline(config.target_sensitivity, color=".45", linestyle="--", linewidth=.8)
    axis.axvline(config.target_fpr, color=".45", linestyle="--", linewidth=.8)
    axis.set(xlabel="Held-out false-positive rate", ylabel="Condition sensitivity", xlim=(0, .12), ylim=(0, 1))
    axis.legend(fontsize=6, frameon=False)
    tradeoff_path = context.figure_dir / "condition_detection_tradeoff.png"
    figure.savefig(tradeoff_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(5.4, 3.0))
    for algorithm, group in onset_summary.groupby("algorithm"):
        axis.plot(group.window_start_seconds_after_stimulus, group.sensitivity, marker="o",
                  label=algorithm.replace("_", " "))
    axis.set(xlabel="0.5-s window start after stimulus (s)", ylabel="Condition sensitivity", ylim=(0, 1))
    axis.legend(fontsize=6, frameon=False)
    onset_figure = context.figure_dir / "equal_window_onset_control.png"
    figure.savefig(onset_figure, dpi=180, bbox_inches="tight")
    plt.close(figure)
    context.progress(stage, 4, 5, "figures and onset summary")

    conclusions = {
        "target": {"sensitivity": config.target_sensitivity, "fpr": config.target_fpr},
        "target_reached": target_reached,
        "best_primary_algorithm": str(best.algorithm),
        "best_primary_sensitivity": float(best.sensitivity),
        "best_primary_fpr": float(best.heldout_fpr),
        "best_primary_balanced_accuracy": float(best.balanced_accuracy),
        "best_primary_auc": float(best.auc),
        "primary_results": primary_summary.to_dict(orient="records"),
        "comparisons": comparisons,
        "ml_training_evaluation_units_match": True,
        "manuscript_updated": False,
    }
    conclusion_path = context.run_dir / "accuracy_conclusions.json"
    atomic_write_json(conclusion_path, conclusions)
    outcome = "reached" if target_reached else "not reached"
    report = f"""# Condition-level accuracy study

The locked target of at least {100 * config.target_sensitivity:.0f}% sensitivity at a held-out {100 * config.target_fpr:.0f}% false-positive rate was **{outcome}**.

Best primary method: **{best.algorithm}**, sensitivity {100 * best.sensitivity:.1f}%, FPR {100 * best.heldout_fpr:.2f}%, balanced accuracy {100 * best.balanced_accuracy:.1f}%, AUC {best.auc:.3f}.

All methods operate on the paper-aligned participant-condition unit formed from three streams. Learned models use matched condition-level training and evaluation observations, grouped inner validation, nonnegative coefficients, and training-only thresholds.

The result is an algorithm-development experiment, not a manuscript update. Failure to reach 90% is retained rather than hidden through threshold or endpoint changes.
"""
    report_path = context.run_dir / "report.md"
    atomic_write_text(report_path, report)
    context.progress(stage, 5, 5, "conclusions locked")
    outputs = [
        summary_path, comparison_path, model_summary_path, onset_path, tradeoff_path,
        onset_figure, conclusion_path, report_path,
    ]
    context.finish_stage(stage, outputs, conclusions)
    return outputs, conclusions
