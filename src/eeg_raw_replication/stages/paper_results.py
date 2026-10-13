from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats
from scipy.io import loadmat

from ..config import CONDITIONS
from ..runtime import RunContext, atomic_write_json
from ..statistics import covariance_pca, paired_comparisons, project_pca, rm_anova_oneway


GROUPS = {
    "fricatives_sibilants": (10, 11, 14, 15),
    "nasals_liquids": (8, 9, 12, 13),
    "stops": (2, 3, 4, 5, 6, 7),
}


def _group_values(pc1: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    labels = tuple(GROUPS)
    columns = [pc1[:, np.asarray(GROUPS[label]) - 1].mean(axis=1) for label in labels]
    return np.column_stack(columns), labels


def _save_pca_figure(scores: np.ndarray, branch: str, destination) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    means = scores.mean(axis=0)
    errors = scores.std(axis=0, ddof=1) / np.sqrt(scores.shape[0])
    figure, axis = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    axis.errorbar(means[:, 0], means[:, 1], xerr=errors[:, 0], yerr=errors[:, 1], fmt="none", alpha=0.4)
    axis.scatter(means[:, 0], means[:, 1], s=35)
    for index, label in enumerate(CONDITIONS):
        axis.annotate(label, means[index, :2], xytext=(4, 4), textcoords="offset points", fontsize=8)
    axis.axhline(0, color="black", linewidth=0.6)
    axis.axvline(0, color="black", linewidth=0.6)
    axis.set(xlabel="ITPC principal component 1", ylabel="ITPC principal component 2", title=branch)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    pearson = stats.pearsonr(left, right)
    spearman = stats.spearmanr(left, right)
    return {
        "pearson_r": float(pearson.statistic), "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic), "spearman_p": float(spearman.pvalue),
    }


def run(context: RunContext, force: bool = False) -> tuple[list, dict]:
    stage = "06_paper_results"
    context.start_stage(stage, len(context.branches))
    with np.load(context.run_dir / "harmonic_itpc.npz", allow_pickle=False) as archive:
        stream_values = np.asarray(archive["mean_itpc"], dtype=float)
    sound_path = context.config.paths.original_repo_root / "Data files" / "Sound_properties_may21.mat"
    sound = loadmat(sound_path, squeeze_me=True)
    sound_condition = np.asarray(sound["sound_prop2_cond"], dtype=float)
    derivative_latency = sound_condition[:, 1]
    reports = {}
    score_rows = []
    test_rows = []
    outputs = []
    for branch_index, branch in enumerate(context.branches):
        participant_stream = stream_values[branch_index]
        condition = participant_stream.reshape(len(context.participants), 15, 3, 4).mean(axis=2)
        paper_pca = covariance_pca(condition.mean(axis=1), orient_first_index=0)
        scores = project_pca(condition, paper_pca)
        repository_values = participant_stream.transpose(1, 0, 2).reshape(-1, 4, order="F")
        repository_pca = covariance_pca(repository_values, orient_first_index=0)
        group_values, group_labels = _group_values(scores[:, :, 0])
        group_anova = rm_anova_oneway(group_values)
        comparisons = paired_comparisons(group_values, group_labels)
        condition_anova = rm_anova_oneway(scores[:, :, 0])
        voiced = scores[:, np.asarray((2, 3, 4)) - 1, 0].mean(axis=1)
        unvoiced = scores[:, np.asarray((5, 6, 7)) - 1, 0].mean(axis=1)
        voicing = stats.ttest_rel(voiced, unvoiced)
        mean_itpc4 = condition[:, :, 0].mean(axis=0)
        acoustic_relation = _safe_correlation(mean_itpc4, derivative_latency)
        figure_path = context.figure_dir / f"pca_conditions_{branch}.png"
        _save_pca_figure(scores, branch, figure_path)
        outputs.append(figure_path)
        for participant_index, participant in enumerate(context.participants):
            for condition_index, condition_name in enumerate(CONDITIONS):
                score_rows.append({
                    "participant": participant, "branch": branch,
                    "condition_id": condition_index + 1, "condition": condition_name,
                    "pc1": float(scores[participant_index, condition_index, 0]),
                    "pc2": float(scores[participant_index, condition_index, 1]),
                    **{
                        f"itpc_{int(frequency)}hz": float(condition[participant_index, condition_index, h_index])
                        for h_index, frequency in enumerate(context.config.harmonics_hz)
                    },
                })
        for comparison in comparisons:
            test_rows.append({"branch": branch, "test": "planned_group_pair", **comparison})
        reports[branch] = {
            "paper_pca_25x4": {
                "observations": len(context.participants),
                "mean": paper_pca.mean.tolist(), "loadings": paper_pca.loadings.tolist(),
                "explained_variance_ratio": paper_pca.explained_variance_ratio.tolist(),
            },
            "repository_pca_1125x4": {
                "observations": int(repository_values.shape[0]),
                "mean": repository_pca.mean.tolist(), "loadings": repository_pca.loadings.tolist(),
                "explained_variance_ratio": repository_pca.explained_variance_ratio.tolist(),
                "provenance_analysis_only": True,
            },
            "condition_pc1_anova": condition_anova,
            "three_group_pc1_anova": group_anova,
            "planned_group_comparisons": comparisons,
            "voiced_vs_unvoiced_stops": {
                "mean_difference": float(np.mean(voiced - unvoiced)),
                "t": float(voicing.statistic), "p": float(voicing.pvalue),
            },
            "maximum_derivative_latency_vs_itpc4": acoustic_relation,
            "acoustic_source": str(sound_path),
        }
        context.progress(stage, branch_index + 1, len(context.branches), branch)
    scores_path = context.table_dir / "raw_condition_scores.csv"
    tests_path = context.table_dir / "raw_planned_comparisons.csv"
    report_path = context.run_dir / "paper_results.json"
    pd.DataFrame(score_rows).to_csv(scores_path, index=False)
    pd.DataFrame(test_rows).to_csv(tests_path, index=False)
    atomic_write_json(report_path, {
        "branches": reports,
        "interpretation": "Effects are compared by direction, magnitude, interval, and conclusion; exact raw agreement is not required.",
    })
    outputs.extend([scores_path, tests_path, report_path])
    context.finish_stage(stage, outputs, {"branches": list(reports)})
    return outputs, reports
