from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..runtime import RunContext, atomic_write_text


def _percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def run(context: RunContext, force: bool = False) -> tuple[list, dict]:
    del force
    stage, config = "07_report", context.config
    context.start_stage(stage, 4)
    conclusions = json.loads((context.run_dir / "assr_validation_conclusions.json").read_text(encoding="utf-8"))
    primary = conclusions["primary"]
    static = pd.read_csv(context.table_dir / "static_participant_metrics.csv")
    sequential = pd.read_csv(context.table_dir / "sequential_participant_metrics.csv")
    gain = pd.read_csv(context.table_dir / "sequential_gain_curve.csv")
    summary = (static.groupby(["method", "frequency_hz"], as_index=False)
               .agg(detection=("detection", "mean"), heldout_fpr=("heldout_fpr", "mean")))
    summary_path = context.table_dir / "estimator_summary.csv"
    summary.to_csv(summary_path, index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    primary_summary = summary[summary["frequency_hz"] == config.primary_frequency_hz]
    ax.bar(primary_summary["method"], 100 * primary_summary["detection"], color=("#4472C4", "#70AD47", "#ED7D31"))
    ax.set(ylabel="Detected blocks (%)", ylim=(0, 100), title="40-Hz K=10 participant-held-out detection")
    fig.tight_layout(); detection_figure = context.figure_dir / "static_detection.png"
    fig.savefig(detection_figure, dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(gain["k"], 100 * gain["gain"], marker="o", color="#ED7D31")
    ax.fill_between(gain["k"], 100 * gain["simultaneous_ci95_low"],
                    100 * gain["simultaneous_ci95_high"], alpha=0.2, color="#ED7D31")
    ax.axhline(0, color="black", linewidth=1)
    ax.set(xlabel="Trials per block (K)", ylabel="Hilbert - Hann detection (percentage points)",
           title="40-Hz sequential gain with simultaneous 95% band")
    fig.tight_layout(); sequential_figure = context.figure_dir / "sequential_gain.png"
    fig.savefig(sequential_figure, dpi=180); plt.close(fig)

    seq40 = sequential[sequential["frequency_hz"] == config.primary_frequency_hz]
    fig, ax = plt.subplots(figsize=(7, 4))
    groups = [seq40.loc[seq40["method"] == method, "rmst_k"].to_numpy(float)
              for method in ("hann", "rectangular_ls", "hilbert")]
    ax.boxplot(groups, tick_labels=("Hann", "Rectangular LS", "Hilbert"), showmeans=True)
    ax.set(ylabel="Restricted mean stopping K", title="40-Hz sequential stopping")
    fig.tight_layout(); stopping_figure = context.figure_dir / "stopping_distribution.png"
    fig.savefig(stopping_figure, dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    fpr = static[static["frequency_hz"] == config.primary_frequency_hz]
    for offset, method in enumerate(("hann", "rectangular_ls", "hilbert")):
        values = fpr.loc[fpr["method"] == method, "heldout_fpr"].to_numpy(float)
        ax.scatter(np.full(len(values), offset), 100 * values, alpha=0.55, s=18)
    ax.axhline(100 * config.target_fpr, color="black", linestyle="--", linewidth=1)
    ax.axhline(100 * config.fpr_acceptance_high, color="#C00000", linestyle=":", linewidth=1)
    ax.set(xticks=range(3), xticklabels=("Hann", "Rectangular LS", "Hilbert"),
           ylabel="Held-out null FPR (%)", title="Participant-held-out 40-Hz calibration")
    fig.tight_layout(); calibration_figure = context.figure_dir / "heldout_fpr.png"
    fig.savefig(calibration_figure, dpi=180); plt.close(fig)

    methods = f"""# Methods

The prospectively locked external-positive analysis used {primary['eligible_participants']} typically developing participants from OpenNeuro {config.dataset_id} version {config.dataset_version}. The primary 40-Hz analysis used the first 170 standard trials in 17 chronological blocks of ten; the secondary 27-Hz analysis used the first 160 standards in 16 blocks. Oddballs, responses, block markers, and surplus standards were excluded.

Continuous 64-channel EEG was zero-phase filtered from {config.l_freq:g} to {config.h_freq:g} Hz, notch filtered at {config.notch_hz:g} Hz, and average referenced. Hann weighted least squares, rectangular least squares, and fourth-order Hilbert coefficients with a +/-{config.hilbert_half_bandwidth_hz:g}-Hz band were evaluated on the identical [{config.epoch_start_seconds:.2f}, {config.epoch_end_seconds:.2f})-s support. Scores were 2K times equal-channel ITPC-squared.

Thresholds and sequential boundaries were calibrated using only other participants' phase-rotation nulls. The primary interval used {config.bootstrap_iterations} paired participant-bootstrap draws. Generalization required at least {config.minimum_participants} participants, a strictly positive lower 95% confidence bound for Hilbert minus Hann at K=10, and Hilbert held-out FPR no greater than {_percent(config.fpr_acceptance_high)}.
"""
    result_text = f"""# Results

The primary 40-Hz Hilbert-minus-Hann gain was {_percent(primary['estimate'])} (95% CI {_percent(primary['ci95_low'])} to {_percent(primary['ci95_high'])}). Hilbert held-out FPR was {_percent(primary['hilbert_heldout_fpr'])}. The predefined external-positive gate **{'passed' if primary['passed'] else 'did not pass'}**.

The Hann-minus-Hilbert restricted mean stopping difference was {conclusions['sequential']['estimate']:.2f} trials (95% CI {conclusions['sequential']['ci95_low']:.2f} to {conclusions['sequential']['ci95_high']:.2f}). Full estimator, frequency, sequential, and artifact-sensitivity results are supplied in the exported tables.
"""
    limitations = """# Limitations

This dataset tests estimator behavior on short auditory steady-state responses in children. It does not reproduce the syllable paradigm, its four-harmonic feature space, the learned regional detector, or its clinical context. The artifact-rejection analysis is explicitly secondary and cannot change the primary decision. Equal channel weighting avoids learned spatial leakage but may dilute focal responses.
"""
    methods_body = methods.removeprefix("# Methods\n\n")
    report = f"# Independent positive ASSR validation\n\n{methods_body}\n{result_text}\n{limitations}\n"
    methods_path, results_path = context.run_dir / "methods.md", context.run_dir / "results.md"
    limitations_path, report_path = context.run_dir / "limitations.md", context.run_dir / "report.md"
    atomic_write_text(methods_path, methods); atomic_write_text(results_path, result_text)
    atomic_write_text(limitations_path, limitations); atomic_write_text(report_path, report)
    context.progress(stage, 4, 4, "reports and figures")
    outputs = [summary_path, detection_figure, sequential_figure, stopping_figure, calibration_figure,
               methods_path, results_path, limitations_path, report_path]
    context.finish_stage(stage, outputs, {"primary_passed": primary["passed"]})
    return outputs, {"primary_passed": primary["passed"]}
