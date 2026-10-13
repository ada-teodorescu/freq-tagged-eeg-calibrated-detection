from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cucu_article.runtime import atomic_write_text

from ..runtime import RunContext


def _confirmatory_figure(context: RunContext, rows: list[dict]) -> Path:
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    labels = [f"{row['candidate']}\nvs {row['comparator']}" for row in rows]
    gains = [100 * row["mean_detection_gain"] for row in rows]
    los = [100 * row["crossed_bootstrap_ci95"][0] for row in rows]
    his = [100 * row["crossed_bootstrap_ci95"][1] for row in rows]
    positions = np.arange(len(rows))
    colors = ["#228833" if row["confirmatory_pass"] else "#AA3377" for row in rows]
    axis.bar(positions, gains, color=colors)
    axis.errorbar(positions, gains, yerr=[np.subtract(gains, los), np.subtract(his, gains)],
                 fmt="none", ecolor="black", capsize=4)
    axis.axhline(0, color="0.3", linewidth=1)
    axis.set_xticks(positions, labels, fontsize=7)
    axis.set(ylabel="Detection gain (percentage points)", title="Confirmatory three-hypothesis family")
    axis.grid(axis="y", alpha=.2)
    fig.tight_layout()
    path = context.figure_dir / "confirmatory_family.png"
    fig.savefig(path, dpi=180); plt.close(fig)
    return path


def _sequential_figure(context: RunContext, scores: pd.DataFrame) -> Path:
    config = context.config
    selected = scores[scores["calibration"] == "population"]
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    for method in ("S2_Equal4plus8", "S4_RegionalMonotone4plus8", "TRCA_S2_Grouped"):
        rows = selected[selected["method"] == method]
        rates = [rows[f"detected_by_k{k}"].astype(float).mean() for k in config.k_values]
        axis.plot(config.k_values, rates, marker="o", linewidth=1.8, label=method.replace("_", " "))
    axis.set(xlabel="Trials observed (K)", ylabel="Cumulative stream detection", ylim=(0, 1.02),
             title="TRCA+S2 vs S2 vs S4 (population-calibrated, 4.5s window)")
    axis.grid(alpha=.2); axis.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    path = context.figure_dir / "trca_sequential.png"
    fig.savefig(path, dpi=180); plt.close(fig)
    return path


def run(context: RunContext, force: bool = False) -> tuple[list[Path], dict]:
    del force
    stage = "07_report"
    context.start_stage(stage, 4)

    conclusions = json.loads((context.run_dir / "confirmatory_conclusions.json").read_text(encoding="utf-8"))
    lock = json.loads((context.run_dir / "design_lock.json").read_text(encoding="utf-8"))
    trca_scores = pd.read_csv(context.table_dir / "trca_sequential_scores.csv")

    figures = [
        _confirmatory_figure(context, conclusions["confirmatory_family"]),
        _sequential_figure(context, trca_scores),
    ]
    for index, path in enumerate(figures, 1):
        context.progress(stage, index, 4, path.name)

    def _row_line(row: dict) -> str:
        return (f"- {row['candidate']} vs {row['comparator']}: gain {100 * row['mean_detection_gain']:.1f} pp, "
                f"crossed-bootstrap 95% CI [{100 * row['crossed_bootstrap_ci95'][0]:.1f}, "
                f"{100 * row['crossed_bootstrap_ci95'][1]:.1f}], Holm p={row.get('holm_adjusted_p', float('nan')):.4g}, "
        f"FPR={100 * row.get('heldout_candidate_fpr', float('nan')):.2f}%, "
        f"pass={row.get('confirmatory_pass', 'n/a')}.")

    confirmatory_lines = "\n".join(_row_line(row) for row in conclusions["confirmatory_family"])
    secondary_lines = "\n".join(
        f"- {row['candidate']} vs {row['comparator']}: gain {100 * row['mean_detection_gain']:.1f} pp, "
        f"crossed-bootstrap 95% CI [{100 * row['crossed_bootstrap_ci95'][0]:.1f}, "
        f"{100 * row['crossed_bootstrap_ci95'][1]:.1f}] ({row['note']})."
        for row in conclusions["secondary_comparisons"]
    )
    context.progress(stage, 3, 4, "narrative assembled")

    gate = conclusions["stop_and_report_gate_triggered"]
    trca_verdict = conclusions["trca_verdict"]
    report = f"""# Confirmatory Cucu Detector Study

Analysis ID: `{context.analysis_id}`

## Locked three-hypothesis confirmatory family

{confirmatory_lines}

Each pass criterion requires the ordinary crossed-bootstrap 95% CI entirely above zero, the
Holm-adjusted sign-flip p-value below .05, and held-out surrogate FPR inside the locked 4.5--5.5%
interval. Multiplicity is adjusted across exactly these three hypotheses.

## Secondary comparisons (excluded from the Holm family)

{secondary_lines}

The H2-vs-H3 comparison is duration-confounded (H3 has 0.5s more data than H2). The sequential
condition-level trajectories reported separately in `hilbert_sequential_*` tables are Pocock
familywise-corrected and are **not** comparable to the static single-look K=10 numbers above, or to
the historical C4 figure (82.1%) from `cucu_advanced_study`. Repeated-look correction typically lowers
the K=10 boundary, while cumulative early crossings make it a distinct endpoint rather than a strict
numerical transformation of the single-look result.

## TRCA classification

TRCA+S2 beats population-calibrated S2: **{trca_verdict['beats_s2']}**. TRCA+S2 beats
population-calibrated S4: **{trca_verdict['beats_s4']}**. Classification: **{trca_verdict['classification']}**
(useful_spatial_improvement / new_stream_leader / no_confirmed_benefit -- a genuine improvement over
the fixed detector is acknowledged even without also beating S4).

TRCA is a subject-calibrated detector: its spatial filter was fit per participant via
leave-one-condition-out (42 training streams from the held participant's other 14 conditions), unlike
S2/S4's population calibration. The TRCA-vs-shrinkage-S4 secondary comparison above probes calibration
adaptation, but is not fully information-matched because the fitted spatial models differ.

## Stop-and-report gate

Gate triggered (neither H2 nor TRCA_S2 passed its locked confirmatory comparison): **{gate}**.
{"Per the locked design, detector development stops here: report both null results as-is. Do not "
 "expand the C grid, change regions, change harmonics, retune TRCA shrinkage, or select a favorable "
 "duration after observing this held-out performance." if gate else
 "At least one candidate passed its locked confirmatory comparison; see the classification above for "
 "which."}

## Baselines this study compares against

Locked `cucu_advanced_study` baseline (`{lock['baseline_advanced_run']}`): S2_Equal4plus8 35.5%,
S4_RegionalMonotone4plus8 44.4% (stream-level sequential detection by K=10);
C4_PreviousRegionalMonotone4H 82.1% (condition-level, single-look K=10). This study recomputes S2 and
S4 fresh in its own harness rather than reusing those exported numbers directly, so the population
comparisons above are apples-to-apples within this study's own folds and nulls.

## Figures

- `figures/confirmatory_family.png`
- `figures/trca_sequential.png`

## Guardrail

{conclusions['interpretation_guardrail']}

All thresholds, fold records, model coefficients, scores, calibration comparisons, and control results
are in the exported CSV tables. Existing study packages (`cucu_detection`, `cucu_accuracy_study`,
`cucu_algorithm_study`, `cucu_advanced_study`) and their exported results were not modified.
"""
    report_path = context.run_dir / "report.md"
    atomic_write_text(report_path, report)
    context.progress(stage, 4, 4, "report.md")
    outputs = [report_path, *figures]
    summary = {"stop_and_report_gate_triggered": gate, "trca_classification": trca_verdict["classification"],
               "report": str(report_path)}
    context.finish_stage(stage, outputs, summary)
    return outputs, summary
