from __future__ import annotations

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from eeg_core.runtime import atomic_write_text
from ..runtime import RunContext


def _percent(value: float) -> str: return f"{100 * value:.1f}%"


def run(context: RunContext, force: bool = False) -> tuple[list[Path], dict]:
    del force
    stage, config = "07_report", context.config; context.start_stage(stage, 4)
    conclusions = json.loads((context.run_dir / "validation_extension_conclusions.json").read_text(encoding="utf-8"))
    static = pd.read_csv(context.table_dir / "static_scores.csv")
    static_summary = (static.groupby(["method", "detector"], as_index=False)
                      .agg(detection=("detected", "mean"), heldout_fpr=("heldout_fpr", "mean")))
    static_path = context.table_dir / "static_summary.csv"; static_summary.to_csv(static_path, index=False)
    context.progress(stage, 1, 4, "paper tables")

    outputs = [static_path]
    response = pd.read_csv(context.table_dir / "hilbert_kernel_response.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.2)); ax.plot(response["frequency_hz"], response["magnitude"])
    ax.set(xlabel="Frequency (Hz)", ylabel="Effective-kernel magnitude", title="Locked Hilbert estimator response")
    ax.axvline(-config.kernel_frequency_hz, color="black", linestyle="--", linewidth=1); fig.tight_layout()
    kernel_figure = context.figure_dir / "hilbert_kernel_response.png"; fig.savefig(kernel_figure, dpi=180); plt.close(fig); outputs.append(kernel_figure)
    curve = pd.read_csv(context.table_dir / "sequential_gain_curve.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.2)); ax.plot(curve["look_k"], 100 * curve["gain"], marker="o")
    ax.fill_between(curve["look_k"], 100 * curve["simultaneous_ci95_low"], 100 * curve["simultaneous_ci95_high"], alpha=.2)
    ax.axhline(0, color="black", linewidth=1); ax.set(xlabel="Trials per stream (K)", ylabel="Hilbert - Hann detection (percentage points)",
                                                       title="Earlier detection with simultaneous 95% band"); fig.tight_layout()
    sequential_figure = context.figure_dir / "sequential_gain.png"; fig.savefig(sequential_figure, dpi=180); plt.close(fig); outputs.append(sequential_figure)
    assignment = pd.read_csv(context.table_dir / "external_window_selection_robustness.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), sharey=True)
    for ax, column, title in zip(axes, ("static_fpr", "sequential_fpr"), ("Static", "Sequential"), strict=True):
        for task, group in assignment.groupby("task"):
            by_seed = group.groupby("resample")[column].mean(); ax.hist(by_seed, bins=15, alpha=.45, label=task)
        ax.axvline(.05, color="black", linestyle="--"); ax.set(title=title, xlabel="False-positive rate")
    axes[0].set_ylabel("Resamples"); axes[1].legend(fontsize=7); fig.tight_layout()
    external_figure = context.figure_dir / "external_window_robustness.png"; fig.savefig(external_figure, dpi=180); plt.close(fig); outputs.append(external_figure)
    context.progress(stage, 2, 4, "figures")

    mechanism = conclusions["mechanism"]; sequential = conclusions["sequential"]
    filter_matched = conclusions["mechanism_exploratory_filter_matched"]
    methods = f"""# Methods: robustness and generalization extension

This post-hoc extension preserved the frozen confirmatory inputs and compared matched-support coefficient estimators over 0.5-4.5 s. All learned comparisons used the same five-region, four-harmonic features, temporal-shift nulls, participant-held-out fitting, and empirical 5% null calibration. Mechanism contrasts used participant-by-condition crossed bootstrap intervals and participant-level sign-flip tests with Holm adjustment.

Sequential analyses used K=2 through 10, Pocock-style familywise boundaries, simultaneous crossed-bootstrap bands, and restricted mean stopping K with non-detections assigned K=11. Double-holdout folds excluded both the tested participant and tested condition from every fitting and calibration step. The transparent detector equally averaged all 20 regional-by-harmonic 2K R-squared features and used no fitted weights.

External robustness preserved the original ds007615 gate. It added complete-case artifact summaries, a detector trained on the 28 genuinely shared channels, and 100 deterministic non-overlapping raw-window selections per recording.
"""
    simple = conclusions["simple_detector"]; double = conclusions["double_holdout"]
    results = f"""# Results: robustness and generalization extension

The matched estimator contrasts were: {mechanism[0]['contrast']} = {_percent(mechanism[0]['estimate'])} (95% CI {_percent(mechanism[0]['ci95_low'])} to {_percent(mechanism[0]['ci95_high'])}, Holm p={mechanism[0]['holm_p_value']:.4g}); {mechanism[1]['contrast']} = {_percent(mechanism[1]['estimate'])} (95% CI {_percent(mechanism[1]['ci95_low'])} to {_percent(mechanism[1]['ci95_high'])}, Holm p={mechanism[1]['holm_p_value']:.4g}).

At K=10, the sequential Hilbert-Hann gain was {_percent(sequential[0]['estimate'])} (95% CI {_percent(sequential[0]['ci95_low'])} to {_percent(sequential[0]['ci95_high'])}). The reduction in restricted mean stopping K was {sequential[1]['estimate']:.2f} repetitions per stream (95% CI {sequential[1]['ci95_low']:.2f} to {sequential[1]['ci95_high']:.2f}).

The learned-minus-simple detection gap was {_percent(simple['estimate_learned_minus_simple'])}; its one-sided 95% upper bound was {_percent(simple['one_sided_95_upper'])} against the predeclared {_percent(simple['margin'])} margin. Non-inferiority was {simple['noninferior']}.

Under participant-and-condition double holdout, the Hilbert-Hann gain was {_percent(double['hilbert_minus_hann']['estimate'])} (95% CI {_percent(double['hilbert_minus_hann']['ci95_low'])} to {_percent(double['hilbert_minus_hann']['ci95_high'])}); the predefined generalization rule was {double['generalization_supported']}.

Post hoc, exploratory, outside the locked mechanism family: a filter-matched control (the same Hilbert Butterworth bandpass, then sinusoidal regression without the Hilbert transform) reached the same detection decision as full Hilbert on every participant-condition (McNemar discordant={filter_matched['mcnemar']['discordant']}, p={filter_matched['mcnemar']['p_value']:.4g}; paired gain {_percent(filter_matched['estimate'])}, 95% CI {_percent(filter_matched['ci95_low'])} to {_percent(filter_matched['ci95_high'])}).
"""
    limitations = """# Limitations of the extension

- These analyses were designed after observing the original estimator result and must be labelled post-hoc.
- Repeated participant-condition units are not independent; crossed resampling and participant sign flips mitigate but do not eliminate this limitation.
- Alternative external windows use continuous-recording filtering and cumulative complex demodulation; this avoids per-window filter transients but is not numerically identical to filtering every five-second window separately.
- The 28-channel analysis changes the trained detector and is an exploratory transfer sensitivity, not a replacement for the locked external gate.
- OpenNeuro ds006780 tests ASSR estimator generalization at different frequencies and trial durations; it cannot validate the full four-harmonic Cucu syllable detector.
"""
    methods_path = context.run_dir / "methods.md"; results_path = context.run_dir / "results.md"; limitations_path = context.run_dir / "limitations.md"
    atomic_write_text(methods_path, methods); atomic_write_text(results_path, results); atomic_write_text(limitations_path, limitations)
    outputs += [methods_path, results_path, limitations_path]; context.progress(stage, 3, 4, "manuscript text")
    report = f"""# Cucu validation extension

Status: post-hoc robustness and generalization analysis. Existing confirmatory and external gates were not modified.

## Main outputs

- `methods.md`, `results.md`, and `limitations.md`: manuscript-ready draft text.
- `tables/static_summary.csv`: matched estimator and simple-detector outcomes.
- `tables/inferential_comparisons.csv`: corrected mechanism and sequential inference.
- `tables/double_holdout_static.csv` and `tables/double_holdout_sequential.csv`: strict unseen-condition folds.
- `tables/chronological_stopping.csv`: actual presentation/time stopping outcomes.
- `tables/external_robustness_summary.csv`: locked, artifact, and shared-channel external results.
- `assr_positive_protocol.md`: frozen but deferred ds006780 protocol.

Simple-detector non-inferiority: **{simple['noninferior']}**. Double-holdout Hilbert generalization rule: **{double['generalization_supported']}**. Locked external gate remained: **{conclusions['external_locked_gate_unchanged']}**.
"""
    report_path = context.run_dir / "report.md"; atomic_write_text(report_path, report); outputs.append(report_path)
    context.progress(stage, 4, 4, "report")
    context.finish_stage(stage, outputs, {"paper_ready": True}); return outputs, {"paper_ready": True}
