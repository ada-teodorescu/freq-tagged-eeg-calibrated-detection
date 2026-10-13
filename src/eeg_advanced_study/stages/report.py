from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eeg_core.runtime import atomic_write_text

from ..runtime import RunContext
from .stream import METHODS


def _sequential_figure(context: RunContext, scores: pd.DataFrame) -> Path:
    config = context.config
    selected = scores[(scores["calibration"] == "population") &
                      np.isclose(scores["duration_seconds"], config.primary_duration_seconds)]
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    for method in METHODS:
        rows = selected[selected["method"] == method]
        rates = [rows[f"detected_by_k{k}"].astype(float).mean() for k in config.k_values]
        axis.plot(config.k_values, rates, marker="o", linewidth=1.8, label=method.replace("_", " "))
    axis.axhline(config.target_detection, color="0.4", linestyle="--", linewidth=1, label="80% target")
    axis.set(xlabel="Trials observed (K)", ylabel="Cumulative stream detection", ylim=(0, 1.02),
             title="Natural-order sequential detection (4.5 s)")
    axis.grid(alpha=.2); axis.legend(fontsize=7, loc="lower right")
    fig.tight_layout(); path = context.figure_dir / "sequential_detection.png"
    fig.savefig(path, dpi=180); plt.close(fig); return path


def _calibration_figure(context: RunContext, metrics: pd.DataFrame) -> Path:
    config = context.config
    selected = metrics[(metrics["method"] == METHODS[1]) &
                       np.isclose(metrics["duration_seconds"], config.primary_duration_seconds)]
    pop = selected[selected["calibration"] == "population"].set_index("participant")["familywise_fpr"]
    shrink = selected[selected["calibration"] == "shrinkage"].groupby("participant")["familywise_fpr"].mean()
    fig, axis = plt.subplots(figsize=(4.8, 4.4))
    axis.scatter(pop, shrink, s=28, alpha=.8)
    limit = max(.12, float(max(pop.max(), shrink.max()) * 1.08))
    axis.plot([0, limit], [0, limit], color="0.5", linestyle="--", linewidth=1)
    axis.axhline(config.target_fpr, color="tab:red", linewidth=1); axis.axvline(config.target_fpr, color="tab:red", linewidth=1)
    axis.set(xlabel="Population-calibrated participant FWER", ylabel="Shrinkage-calibrated participant FWER",
             xlim=(0, limit), ylim=(0, limit), title="S2 calibration by participant")
    fig.tight_layout(); path = context.figure_dir / "calibration_comparison.png"
    fig.savefig(path, dpi=180); plt.close(fig); return path


def _condition_figure(context: RunContext, scores: pd.DataFrame) -> Path:
    rates = scores.groupby("method")["detected"].mean().reindex([
        "C1_ConditionITPC4", "C2_ConditionEqual4plus8",
        "C3_RegionalMonotone4plus8", "C4_PreviousRegionalMonotone4H",
    ])
    fig, axis = plt.subplots(figsize=(7.2, 4.0))
    axis.bar(np.arange(len(rates)), rates.to_numpy(), color=["#777777", "#4477AA", "#228833", "#AA3377"])
    axis.set_xticks(np.arange(len(rates)), [name.replace("_", "\n") for name in rates.index], fontsize=7)
    axis.set(ylabel="Held-out condition sensitivity", ylim=(0, 1.0), title="Separate condition-level endpoint")
    axis.grid(axis="y", alpha=.2); fig.tight_layout()
    path = context.figure_dir / "condition_benchmark.png"; fig.savefig(path, dpi=180); plt.close(fig); return path


def _negative_figure(context: RunContext, summary: pd.DataFrame) -> Path:
    selected = summary[(summary["control_kind"] == "off-target control") & (summary["method"] == METHODS[1])]
    fig, axis = plt.subplots(figsize=(7.2, 3.8))
    axis.bar(np.arange(len(selected)), selected["false_positive_rate"], color="#CC6677")
    axis.axhline(context.config.target_fpr, color="black", linestyle="--", linewidth=1)
    axis.set_xticks(np.arange(len(selected)), selected["family"], rotation=25, ha="right")
    axis.set(ylabel="Apparent detection rate", ylim=(0, max(.12, float(selected["false_positive_rate"].max() * 1.15))),
             title="S2 off-target controls (not sham recordings)")
    axis.grid(axis="y", alpha=.2); fig.tight_layout()
    path = context.figure_dir / "off_target_controls.png"; fig.savefig(path, dpi=180); plt.close(fig); return path


def run(context: RunContext, force: bool = False) -> tuple[list[Path], dict]:
    del force
    stage = "07_report"
    context.start_stage(stage, 5)
    conclusions = json.loads((context.run_dir / "advanced_conclusions.json").read_text(encoding="utf-8"))
    scores = pd.read_csv(context.table_dir / "sequential_scores.csv")
    metrics = pd.read_csv(context.table_dir / "calibration_comparisons.csv")
    condition = pd.read_csv(context.table_dir / "condition_scores.csv")
    negative = pd.read_csv(context.table_dir / "negative_control_summary.csv")
    figures = [
        _sequential_figure(context, scores),
        _calibration_figure(context, metrics),
        _condition_figure(context, condition),
        _negative_figure(context, negative),
    ]
    for index, path in enumerate(figures, 1): context.progress(stage, index, 5, path.name)
    primary = conclusions["primary"]
    population = conclusions["stream_population_performance"]
    secondary_lines = "\n".join(
        f"- {row['candidate']} vs {row['comparator']}: gain {100 * row['mean_detection_gain']:.1f} pp, "
        f"crossed-bootstrap 95% CI [{100 * row['crossed_bootstrap_ci95'][0]:.1f}, "
        f"{100 * row['crossed_bootstrap_ci95'][1]:.1f}], Holm p={row['holm_adjusted_p']:.4g}."
        for row in conclusions["secondary_stream_comparisons"]
    )
    performance_lines = "\n".join(
        f"- {method}: sensitivity {100 * values['sensitivity']:.1f}%; held-out surrogate FWER "
        f"{100 * values['heldout_familywise_fpr']:.2f}% (participant-cluster 95% CI "
        f"{100 * values['participant_cluster_ci95'][0]:.2f}–{100 * values['participant_cluster_ci95'][1]:.2f}%)."
        for method, values in population.items()
    )
    shrinkage_lines = "\n".join(
        f"- {method}: population FPR RMSE {values['population_participant_fpr_rmse']:.4f}; "
        f"shrinkage RMSE {values['shrinkage_participant_fpr_rmse']:.4f}; sensitivity change "
        f"{100 * values['sensitivity_change']:.1f} pp; success={values['success']}."
        for method, values in conclusions["shrinkage_calibration"].items()
    )
    condition_lines = "\n".join(
        f"- {method}: {100 * value:.1f}% sensitivity."
        for method, value in conclusions["condition_sensitivity"].items()
    )
    verdict = "PASSED" if primary["success"] else "DID NOT PASS"
    report = f"""# Advanced Detector Study

Analysis ID: `{context.analysis_id}`

## Locked primary verdict

The S2 sequential detector **{verdict}** the locked success rule. Its detection gain over S1 was
{100 * primary['mean_detection_gain']:.1f} percentage points (crossed-bootstrap 95% CI
[{100 * primary['crossed_bootstrap_ci95'][0]:.1f}, {100 * primary['crossed_bootstrap_ci95'][1]:.1f}]),
and its held-out surrogate trajectory-wise FWER was {100 * primary['heldout_surrogate_familywise_fpr']:.2f}%.
The gain-CI criterion was `{primary['gain_ci_above_zero']}` and the locked 4.5–5.5% calibration criterion
was `{primary['fpr_in_locked_interval']}`.

## Stream-level results

{performance_lines}

Sequential detection used the natural recorded repetition order. Undetected streams were censored at 11
for restricted mean stopping-trial summaries. The 100 deterministic order permutations are exported only
as robustness evidence.

## Secondary stream comparisons

{secondary_lines}

## Population versus shrinkage calibration

{shrinkage_lines}

Participant adaptation used only null trajectories from the held participant's other 14 conditions. The
evaluated condition was excluded both from local boundary estimation and null-only pseudo-LOPO weight selection.

## Separate condition-level endpoint

{condition_lines}

The condition analysis aggregates three independently scored streams. It is not directly comparable with
the stream-level primary endpoint.

## Negative-control scope

External negative status: **{conclusions['external_negative_status']}**.

The six frequency families are off-target controls, not genuine sham or response-absent recordings. Therefore
operational FPR is {'supported by the supplied external-negative files' if conclusions['operational_fpr_claim_allowed'] else 'not claimed'}.

## Figures

- `figures/sequential_detection.png`
- `figures/calibration_comparison.png`
- `figures/condition_benchmark.png`
- `figures/off_target_controls.png`

All thresholds, fold records, model choices, scores, calibration comparisons, and control results are in
the exported CSV tables. Existing study and manuscript outputs were not modified.
"""
    report_path = context.run_dir / "report.md"
    atomic_write_text(report_path, report)
    context.progress(stage, 5, 5, "report.md")
    outputs = [report_path, *figures]
    summary = {"primary_success": primary["success"], "report": str(report_path)}
    context.finish_stage(stage, outputs, summary)
    return outputs, summary
