from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from ..runtime import RunContext, atomic_write_json, atomic_write_text


def _plot_detection(stored: pd.DataFrame, path) -> None:
    curves = stored.groupby(["detector", "k"], as_index=False).detection_rate.mean()
    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    for detector, frame in curves.groupby("detector"):
        emphasis = detector in {"D1", "D8"}
        axis.plot(frame.k, frame.detection_rate, marker="o" if emphasis else None,
                  linewidth=2.5 if emphasis else 1.0, alpha=1.0 if emphasis else 0.55,
                  label=detector)
    axis.axhline(0.8, color="black", linestyle="--", linewidth=1)
    axis.set(xlabel="Trials (K)", ylabel="Held-out detection rate", ylim=(0, 1.02))
    axis.legend(ncol=4, frameon=False)
    figure.tight_layout(); figure.savefig(path, dpi=180); plt.close(figure)


def _plot_spatial(public: pd.DataFrame, path) -> None:
    mean = public.groupby(["spatial", "k"], as_index=False).detection_rate.mean()
    baseline = mean[mean.spatial == "S0"].set_index("k").detection_rate
    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    for method, frame in mean.groupby("spatial"):
        gain = frame.set_index("k").detection_rate - baseline
        axis.plot(gain.index, gain, marker="o", label=method)
    axis.axhline(0, color="black", linewidth=1)
    axis.set(xlabel="Trials (K)", ylabel="Detection-rate gain versus S0")
    axis.legend(frameon=False); figure.tight_layout(); figure.savefig(path, dpi=180); plt.close(figure)


def _ranking_table(stored: pd.DataFrame, public: pd.DataFrame) -> pd.DataFrame:
    baseline = stored[(stored.detector == "D1") & (stored.k == 10)].groupby("condition").mean_score.mean()
    rows = []
    for key, frame in public[public.k == 10].groupby(["estimator", "detector", "spatial"]):
        values = frame.groupby("outer_condition").mean_score.mean().reindex(baseline.index)
        correlation = stats.spearmanr(baseline, values)
        rows.append({"estimator": key[0], "detector": key[1], "spatial": key[2],
                     "spearman_rho_vs_stored_baseline": correlation.statistic,
                     "p": correlation.pvalue, "cross_source_equivalence_claim": False})
    return pd.DataFrame(rows)


def _plot_ranking(table: pd.DataFrame, path) -> None:
    selected = table.sort_values("spearman_rho_vs_stored_baseline", ascending=False).head(20)
    labels = selected.estimator + "+" + selected.detector + "+" + selected.spatial
    figure, axis = plt.subplots(figsize=(8, 5.5))
    axis.barh(np.arange(len(selected)), selected.spearman_rho_vs_stored_baseline)
    axis.set_yticks(np.arange(len(selected)), labels=labels, fontsize=7)
    axis.invert_yaxis(); axis.set_xlabel("Spearman correlation with stored F1+D1+S0 ordering")
    figure.tight_layout(); figure.savefig(path, dpi=180); plt.close(figure)


def _plot_topographies(context: RunContext, path) -> None:
    files = sorted((context.checkpoint_dir / "05_spatial_screen").glob("*.weights.npz"))
    methods = ["S1", "S2", "S3", "S4"]
    matrix = []
    for method in methods:
        weights = [np.abs(np.load(file)[method]) for file in files]
        mean = np.mean(weights, axis=0) if weights else np.zeros(32)
        matrix.append(mean / max(mean.max(), np.finfo(float).eps))
    figure, axis = plt.subplots(figsize=(9, 2.8))
    image = axis.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    axis.set_yticks(range(len(methods)), labels=methods)
    axis.set_xlabel("Channel index (normalized absolute fold weight)")
    figure.colorbar(image, ax=axis, label="Relative weight")
    figure.tight_layout(); figure.savefig(path, dpi=180); plt.close(figure)


def run(context: RunContext, force: bool = False) -> tuple[list, dict]:
    stage = "07_report"
    context.start_stage(stage, 5)
    outcomes = json.loads((context.run_dir / "benchmark_outcomes.json").read_text(encoding="utf-8"))
    provenance = json.loads((context.run_dir / "branch_provenance.json").read_text(encoding="utf-8"))
    stored = pd.read_csv(context.table_dir / "detector_scores_long.csv")
    public = pd.read_csv(context.table_dir / "public_spatial_scores_long.csv")
    context.progress(stage, 1, 5, "inputs")
    detection_figure = context.figure_dir / "detection_versus_trials.png"
    spatial_figure = context.figure_dir / "spatial_filter_gain.png"
    ranking_figure = context.figure_dir / "condition_ranking_robustness.png"
    topography_figure = context.figure_dir / "spatial_filter_topographies.png"
    _plot_detection(stored, detection_figure); context.progress(stage, 2, 5, "trial curves")
    _plot_spatial(public, spatial_figure); context.progress(stage, 3, 5, "spatial gains")
    ranking = _ranking_table(stored, public)
    ranking_path = context.table_dir / "condition_ranking_robustness.csv"
    ranking.to_csv(ranking_path, index=False)
    _plot_ranking(ranking, ranking_figure)
    _plot_topographies(context, topography_figure); context.progress(stage, 4, 5, "ranking and weights")
    claims = {
        "primary_candidate": {
            "status": "supported" if outcomes["primary_candidate_succeeds"] else "not_supported",
            "source": "stored_matlab",
            "prespecified": True,
            "evidence": outcomes,
        },
        "d2_d3_sensitivity": {
            "status": "not_an_independent_sensitivity_improvement",
            "evidence": outcomes["d1_d2_d3_interpretation"],
        },
        "public_spatial_benchmark": {
            "status": "within_source_benchmark",
            "source": "public_set_fdt",
            "cross_source_equivalence_claim": False,
        },
        "source_separation": {"status": "enforced", "evidence": provenance},
    }
    claim_path = context.run_dir / "claim_ledger.json"
    atomic_write_json(claim_path, claims)
    lines = [
        "# Full ITPC algorithm benchmark",
        "",
        f"Analysis ID: `{context.analysis_id}`",
        "",
        "## Prespecified comparison",
        "",
        f"- Baseline trials to 80%: `{outcomes['baseline_trials_to_80']}`.",
        f"- D8 candidate trials to 80%: `{outcomes['candidate_trials_to_80']}`.",
        f"- Candidate held-out FPR: `{outcomes['candidate_heldout_fpr']:.4f}`.",
        f"- Crossed-bootstrap gain CI: `{outcomes['crossed_bootstrap_gain_ci95']}`.",
        f"- Condition sign-flip p-value: `{outcomes['condition_sign_flip_p']:.6g}`.",
        f"- All four success criteria passed: **{outcomes['primary_candidate_succeeds']}**.",
        "",
        "D2 and D3 are reported as monotonic recalibrations of D1 at fixed K, not as new sensitivity methods.",
        "",
        "## Provenance boundary",
        "",
        "The archived FFT detector screen (`stored_matlab`) and time-domain estimator/spatial screen "
        "(`public_set_fdt`) are reported separately. Their scores are never pooled.",
        "",
        "## Commands",
        "",
        "```bash",
        "uv run eeg-detection list",
        "uv run eeg-detection run 03_detector_screen --workers 1",
        "uv run eeg-detection run-all --workers 1",
        "uv run eeg-detection status --watch",
        "```",
    ]
    summary_path = context.run_dir / "detection_summary.md"
    atomic_write_text(summary_path, "\n".join(lines) + "\n")
    report = {"analysis_id": context.analysis_id, "outcomes": outcomes,
              "claim_ledger": str(claim_path), "summary": str(summary_path)}
    report_path = context.run_dir / "final_detection_report.json"
    atomic_write_json(report_path, report); context.progress(stage, 5, 5, "claim ledger")
    outputs = [detection_figure, spatial_figure, ranking_figure, topography_figure,
               ranking_path, claim_path, summary_path, report_path]
    context.finish_stage(stage, outputs, report)
    return outputs, report
