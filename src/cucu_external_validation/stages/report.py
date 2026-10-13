from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cucu_article.runtime import atomic_write_text
from ..runtime import RunContext


def _figure(context: RunContext, frame: pd.DataFrame) -> Path:
    primary = frame[frame["analysis"] == "primary_all_epochs"].copy()
    labels = [f"{row.task.replace('_', ' ')}\n{row.endpoint}" for row in primary.itertuples()]
    values = 100 * primary["point_estimate"].to_numpy(float)
    low = 100 * primary["ci95_low"].to_numpy(float); high = 100 * primary["ci95_high"].to_numpy(float)
    fig, axis = plt.subplots(figsize=(8.2, 4.6))
    positions = np.arange(len(primary))
    axis.bar(positions, values, color=["#4477AA" if value < 10 else "#AA3377" for value in values])
    axis.errorbar(positions, values, yerr=[values - low, high - values], fmt="none", ecolor="black", capsize=3)
    axis.axhline(5, color="0.35", linestyle="--", label="Cucu surrogate target (5%)")
    axis.axhline(10, color="#AA3377", linestyle=":", label="External upper-bound gate (10%)")
    axis.set_xticks(positions, labels, rotation=20, ha="right", fontsize=7)
    axis.set(ylabel="External false-positive rate (%)", title="Independent external-negative validation")
    axis.legend(fontsize=7); axis.grid(axis="y", alpha=.2); fig.tight_layout()
    path = context.figure_dir / "external_negative_fpr.png"; fig.savefig(path, dpi=180); plt.close(fig); return path


def run(context: RunContext, force: bool = False) -> tuple[list[Path], dict]:
    del force
    stage = "08_report"; context.start_stage(stage, 3)
    conclusions = json.loads((context.run_dir / "external_validation_conclusions.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(context.table_dir / "external_fpr_summary.csv")
    figure = _figure(context, summary); context.progress(stage, 1, 3, figure.name)
    lines = []
    for row in summary[summary["analysis"] == "primary_all_epochs"].itertuples():
        lines.append(f"- {row.task}, {row.endpoint}: FPR {100*row.point_estimate:.2f}% "
                     f"(cluster 95% CI {100*row.ci95_low:.2f}--{100*row.ci95_high:.2f}%; "
                     f"one-sided upper {100*row.one_sided_upper:.2f}%).")
    context.progress(stage, 2, 3, "narrative assembled")
    gate = conclusions["submission_gate_passed"]
    report = f"""# Independent External-Negative Validation

Analysis ID: `{context.analysis_id}`

## Locked primary and secondary results

{chr(10).join(lines)}

Primary jittered-auditory static pass: **{conclusions['endpoint_passes'].get('static', False)}**.
Primary jittered-auditory sequential pass: **{conclusions['endpoint_passes'].get('sequential', False)}**.
Minimum primary cohort satisfied: **{conclusions['primary_cohort_ok']}**.
Mandatory resting cohorts satisfied: **{conclusions['secondary_cohorts_ok']}**.

## Submission gate

Both primary upper confidence bounds must be below 10%, and the locked minimum cohorts must be present.
Submission gate passed: **{gate}**. Manuscript replacement allowed: **{conclusions['manuscript_replacement_allowed']}**.

## Montage limitation

The BioSemi-64 source lacks TP9, TP10, PO9, and PO10. These four Cucu target sites were reconstructed by
locked spherical interpolation from the full source montage. Dataset-marked bad targets and every interpolation
or exclusion are exported. This is an independent-cohort/device transfer test, not a within-Cucu sham.

## Guardrail

{conclusions['interpretation_guardrail']}

If the gate fails, the result is preserved and no preprocessing, model, threshold, boundary, or margin is retuned.
"""
    report_path = context.run_dir / "report.md"; atomic_write_text(report_path, report)
    context.progress(stage, 3, 3, report_path.name)
    outputs = [report_path, figure]
    result = {"submission_gate_passed": gate, "report": str(report_path)}
    context.finish_stage(stage, outputs, result); return outputs, result
