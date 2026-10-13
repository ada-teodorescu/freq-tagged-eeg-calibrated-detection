from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.io import loadmat

from ..acoustics import analyse_stream
from ..config import CONDITIONS
from ..runtime import RunContext, atomic_save_npz, atomic_write_json
from ..statistics import PCAResult, covariance_pca, paired_comparisons, project_pca, rm_anova_oneway


GROUPS = {
    "stops": (2, 3, 4, 5, 6, 7),
    "fricatives_sibilants": (10, 11, 14, 15),
    "nasals_liquids": (8, 9, 12, 13),
}

ACOUSTIC_NAMES = (
    "maximum_derivative", "maximum_derivative_latency_ms", "sharpness",
    "normalized_sharpness", "gini", "maximum_amplitude",
    "maximum_amplitude_latency_ms", "rise_20_80_ms",
    "derivative_rise_20_80_ms", "plateau_ms", "latency_80_percent_ms",
)


def _condition_harmonics(context: RunContext) -> np.ndarray:
    values = loadmat(context.config.data_root / "Matlab variables" / "Itc_for_stats.mat", squeeze_me=True)
    stacked = np.stack([
        np.asarray(values[f"meanitc{int(harmonic)}"], dtype=float)
        for harmonic in context.config.harmonics_hz
    ], axis=-1)
    return np.transpose(stacked, (1, 0, 2))


def _orient_to_reference(
    pca: PCAResult, scores: np.ndarray, references: tuple[np.ndarray, ...]
) -> tuple[PCAResult, np.ndarray]:
    loadings = pca.loadings.copy()
    all_scores = pca.scores.copy()
    condition_scores = scores.copy()
    for component, reference in enumerate(references):
        correlation = np.corrcoef(condition_scores[..., component].ravel(), reference.ravel())[0, 1]
        if correlation < 0:
            loadings[:, component] *= -1
            all_scores[:, component] *= -1
            condition_scores[..., component] *= -1
    return PCAResult(pca.mean, loadings, all_scores, pca.explained_variance_ratio), condition_scores


def _save_pca_figure(scores: np.ndarray, destination) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    means = scores.mean(axis=0)
    errors = scores.std(axis=0, ddof=1) / np.sqrt(scores.shape[0])
    figure, axis = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    axis.errorbar(means[:, 0], means[:, 1], xerr=errors[:, 0], yerr=errors[:, 1], fmt="none", alpha=0.45)
    axis.scatter(means[:, 0], means[:, 1], s=38)
    for index, condition in enumerate(CONDITIONS):
        axis.annotate(condition, means[index, :2], xytext=(4, 4), textcoords="offset points", fontsize=8)
    axis.axhline(0, color="black", linewidth=0.6)
    axis.axvline(0, color="black", linewidth=0.6)
    axis.set_xlabel("Archived ITPC principal component 1")
    axis.set_ylabel("Archived ITPC principal component 2")
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def run(context: RunContext, force: bool = False) -> tuple[list, dict]:
    stage = "03_published_results"
    config = context.config
    context.start_stage(stage, 46)
    original_data = config.original_repo_root / "Data files"
    r_folder = config.original_repo_root / "R Stats"

    condition_harmonics = _condition_harmonics(context)
    with np.load(context.run_dir / "stored_fft_harmonics.npz") as archive:
        stored_stream = np.asarray(archive["harmonic_itpc"], dtype=float)
    original_stream = np.asarray(
        loadmat(original_data / "Itc_for_pca_oana.mat", squeeze_me=True)["Itc_for_pca"], dtype=float
    )
    stream_gate_error = float(np.max(np.abs(stored_stream - original_stream.transpose(1, 0, 2))))
    if stream_gate_error >= config.exact_tolerance:
        raise AssertionError(f"Itc_for_pca_oana gate failed: {stream_gate_error}")

    paper_pca = covariance_pca(condition_harmonics.mean(axis=1), orient_first_index=0)
    expected_paper = np.asarray([0.762027, 0.200788])
    paper_gate = float(np.max(np.abs(paper_pca.explained_variance_ratio[:2] - expected_paper))) < 5e-4
    if not paper_gate:
        raise AssertionError(f"Paper PCA variance gate failed: {paper_pca.explained_variance_ratio[:2].tolist()}")
    paper_condition_scores = project_pca(condition_harmonics, paper_pca)

    repository_values = original_stream.reshape(1125, 4, order="F")
    repository_pca = covariance_pca(repository_values, orient_first_index=0)
    repository_stream_scores = repository_pca.scores.reshape(45, 25, 4, order="F")
    repository_condition_scores = np.stack([
        repository_stream_scores[index:index + 3].mean(axis=0)
        for index in range(0, 45, 3)
    ], axis=1)
    archived_pc1 = np.asarray(loadmat(original_data / "new_itc_pca1_for_stats.mat", squeeze_me=True)["itc_pca1"], dtype=float)
    archived_pc2 = np.asarray(loadmat(original_data / "new_itc_pca2_for_stats.mat", squeeze_me=True)["itc_pca2"], dtype=float)
    repository_pca, repository_condition_scores = _orient_to_reference(
        repository_pca, repository_condition_scores, (archived_pc1, archived_pc2)
    )
    archived_errors = [
        float(np.max(np.abs(repository_condition_scores[:, :, 0] - archived_pc1))),
        float(np.max(np.abs(repository_condition_scores[:, :, 1] - archived_pc2))),
    ]
    if max(archived_errors) >= config.exact_tolerance:
        raise AssertionError(f"Archived PCA score gate failed: {archived_errors}")
    archived_scores = repository_condition_scores
    context.progress(stage, 1, 46, "exact archived and paper PCA gates")

    r_three = pd.read_csv(r_folder / "itc1_consonants2.csv")
    r_three_values = r_three[["Sibilants", "Nasals", "Stops"]].to_numpy(dtype=float)
    group_anova = rm_anova_oneway(r_three_values)
    comparisons = paired_comparisons(r_three_values, ("sibilants", "nasals", "stops"))
    r_voicing = pd.read_csv(r_folder / "itc1_consonants4.csv")
    voiced = r_voicing["VoicedStops"].to_numpy(dtype=float)
    unvoiced = r_voicing["UnvoicedStops"].to_numpy(dtype=float)
    voicing_test = stats.ttest_rel(voiced, unvoiced)
    r_conditions = pd.read_csv(r_folder / "itc1_cond.csv").iloc[:, 1:].to_numpy(dtype=float)
    archived_r_correlation = float(stats.pearsonr(archived_pc1.ravel(), r_conditions.ravel()).statistic)
    archived_r_fit = np.polyfit(archived_pc1.ravel(), r_conditions.ravel(), 1)

    sound = loadmat(original_data / "Sound_properties_may21.mat", squeeze_me=True)
    sound_condition = np.asarray(sound["sound_prop2_cond"], dtype=float)
    sound_ordered = np.asarray(sound["sound_prop2_ordered"], dtype=float)
    mean_itpc4 = condition_harmonics[:, :, 0].mean(axis=0)
    derivative_latency = sound_condition[:, 1]
    derivative_relation = stats.pearsonr(mean_itpc4, derivative_latency)
    derivative_spearman = stats.spearmanr(mean_itpc4, derivative_latency)
    if abs(float(derivative_relation.statistic) + 0.9129993328907952) > 1e-12:
        raise AssertionError(f"Original derivative-latency gate failed: {derivative_relation.statistic}")

    zscores = np.asarray(
        loadmat(original_data / "Zscores_sound_prop_pca_may21.mat", squeeze_me=True)["zscores_for_pca"], dtype=float
    )[:, [1, 2, 4, 6, 9]]
    acoustic_pca = covariance_pca(zscores, orient_first_index=0)
    acoustic_condition_pc1 = acoustic_pca.scores[:, 0].reshape(5, 15, order="F").mean(axis=0)
    archived_condition_pc1 = archived_pc1.mean(axis=0)
    acoustic_pc_relation = stats.pearsonr(archived_condition_pc1, acoustic_condition_pc1)
    acoustic_pca_expected = np.asarray([0.5304371514445876, 0.2976582360118266])
    if np.max(np.abs(acoustic_pca.explained_variance_ratio[:2] - acoustic_pca_expected)) >= 1e-12:
        raise AssertionError("Original acoustic PCA variance gate failed")
    if abs(abs(float(acoustic_pc_relation.statistic)) - 0.839658953238523) >= 1e-12:
        raise AssertionError("Original acoustic-PC relationship gate failed")

    exact_acoustic_rows = []
    for feature_index, feature in enumerate(ACOUSTIC_NAMES):
        values = sound_condition[:, feature_index]
        for outcome_name, outcome in (("archived_itpc_pc1", archived_condition_pc1), ("itpc_4hz", mean_itpc4)):
            pearson = stats.pearsonr(values, outcome)
            spearman = stats.spearmanr(values, outcome)
            exact_acoustic_rows.append({
                "feature": feature, "outcome": outcome_name,
                "pearson_r": float(pearson.statistic), "pearson_p": float(pearson.pvalue),
                "spearman_rho": float(spearman.statistic), "spearman_p": float(spearman.pvalue),
                "source": "original_repository_stored_acoustic_matrix",
            })

    acoustic_checkpoint = context.checkpoint_dir / stage / "acoustics"
    acoustic_rows = []
    for stream_id in range(1, 46):
        checkpoint = acoustic_checkpoint / f"stream_{stream_id:02d}.npz"
        if context.checkpoint_valid(checkpoint) and not force:
            with np.load(checkpoint) as archive:
                feature_names = [str(value) for value in archive["feature_names"]]
                row = {
                    "stream_id": stream_id, "condition_id": int(archive["condition_id"]),
                    "condition": str(archive["condition"]), "duration_seconds": float(archive["duration_seconds"]),
                    "fallback": bool(archive["fallback"]), "onsets_seconds": archive["onsets_seconds"],
                    **dict(zip(feature_names, archive["feature_values"])),
                }
        else:
            result = analyse_stream(config.data_root / "Stimuli" / f"{stream_id}.wav", stream_id, config)
            feature_names = tuple(result.features)
            feature_values = np.asarray([result.features[name] for name in feature_names], dtype=float)
            atomic_save_npz(
                checkpoint, stream_id=stream_id, condition_id=result.condition_id, condition=result.condition,
                duration_seconds=result.duration_seconds, onsets_seconds=result.onsets_seconds,
                candidate_indices=result.candidate_indices, fallback=result.fallback,
                feature_names=np.asarray(feature_names), feature_values=feature_values,
            )
            context.mark_checkpoint(checkpoint, {"stream_id": stream_id, "source": "continuous_stream_reconstruction"})
            row = {
                "stream_id": stream_id, "condition_id": result.condition_id, "condition": result.condition,
                "duration_seconds": result.duration_seconds, "fallback": result.fallback,
                "onsets_seconds": result.onsets_seconds, **result.features,
            }
        row["onsets_seconds"] = "|".join(f"{value:.9f}" for value in row["onsets_seconds"])
        acoustic_rows.append(row)
        context.progress(stage, stream_id + 1, 46, f"independent acoustic stream {stream_id}")

    reconstructed_table = pd.DataFrame(acoustic_rows)
    numeric_features = [
        column for column in reconstructed_table.columns
        if column not in {"stream_id", "condition_id", "condition", "fallback", "onsets_seconds"}
    ]
    reconstructed_conditions = reconstructed_table.groupby(["condition_id", "condition"], as_index=False)[numeric_features].mean()
    reconstructed_correlations = []
    for feature in numeric_features:
        x = reconstructed_conditions[feature].to_numpy(dtype=float)
        if not np.all(np.isfinite(x)) or np.std(x) == 0:
            continue
        for outcome_name, outcome in (("archived_itpc_pc1", archived_condition_pc1), ("itpc_4hz", mean_itpc4)):
            pearson = stats.pearsonr(x, outcome)
            spearman = stats.spearmanr(x, outcome)
            reconstructed_correlations.append({
                "feature": feature, "outcome": outcome_name,
                "pearson_r": float(pearson.statistic), "pearson_p": float(pearson.pvalue),
                "spearman_rho": float(spearman.statistic), "spearman_p": float(spearman.pvalue),
                "source": "independently_reconstructed_from_45_continuous_stream_wavs",
            })

    score_rows = []
    for participant in range(25):
        for condition in range(15):
            score_rows.append({
                "participant_index": participant, "condition_id": condition + 1,
                "condition": CONDITIONS[condition], "pc1": float(archived_scores[participant, condition, 0]),
                "pc2": float(archived_scores[participant, condition, 1]),
                "paper_pca_pc1": float(paper_condition_scores[participant, condition, 0]),
                "paper_pca_pc2": float(paper_condition_scores[participant, condition, 1]),
                **{f"itpc_{int(h)}hz": float(condition_harmonics[participant, condition, index])
                   for index, h in enumerate(config.harmonics_hz)},
            })
    ordering = pd.DataFrame({
        "condition_id": np.arange(1, 16), "condition": CONDITIONS,
        "mean_pc1": archived_scores[:, :, 0].mean(axis=0),
        "mean_pc2": archived_scores[:, :, 1].mean(axis=0),
    }).sort_values("mean_pc1", ascending=False)

    paths = {
        "scores": context.table_dir / "published_condition_scores.csv",
        "ordering": context.table_dir / "published_condition_ordering.csv",
        "groups": context.table_dir / "published_group_comparisons.csv",
        "exact_acoustic": context.table_dir / "original_acoustic_correlations.csv",
        "reconstructed": context.table_dir / "reconstructed_acoustic_landmarks.csv",
        "reconstructed_conditions": context.table_dir / "reconstructed_acoustic_condition_means.csv",
        "reconstructed_correlations": context.table_dir / "reconstructed_acoustic_correlations.csv",
        "paper_pca": context.run_dir / "published_pca.json",
        "repository_pca": context.run_dir / "repository_score_pca.json",
        "report": context.run_dir / "published_results.json",
        "figure": context.figure_dir / "published_pca_conditions.png",
    }
    pd.DataFrame(score_rows).to_csv(paths["scores"], index=False)
    ordering.to_csv(paths["ordering"], index=False)
    pd.DataFrame(comparisons).to_csv(paths["groups"], index=False)
    pd.DataFrame(exact_acoustic_rows).to_csv(paths["exact_acoustic"], index=False)
    reconstructed_table.to_csv(paths["reconstructed"], index=False)
    reconstructed_conditions.to_csv(paths["reconstructed_conditions"], index=False)
    pd.DataFrame(reconstructed_correlations).to_csv(paths["reconstructed_correlations"], index=False)
    _save_pca_figure(archived_scores, paths["figure"])

    paper_pca_report = {
        "branch": "paper_reported_pca", "observations": 25,
        "mean": paper_pca.mean.tolist(), "loadings": paper_pca.loadings.tolist(),
        "explained_variance_ratio": paper_pca.explained_variance_ratio.tolist(),
        "method": "covariance PCA on participant harmonic means",
    }
    repository_pca_report = {
        "branch": "repository_score_pca", "observations": 1125,
        "mean": repository_pca.mean.tolist(), "loadings": repository_pca.loadings.tolist(),
        "explained_variance_ratio": repository_pca.explained_variance_ratio.tolist(),
        "archived_score_maximum_errors": archived_errors,
        "method": "MATLAB-order reshape of 45 streams x 25 participants",
    }
    atomic_write_json(paths["paper_pca"], paper_pca_report)
    atomic_write_json(paths["repository_pca"], repository_pca_report)
    report = {
        "source": "stored_matlab_and_original_repository",
        "itc_for_pca_gate": {"passed": True, "maximum_absolute_error": stream_gate_error},
        "paper_reported_pca": paper_pca_report,
        "repository_score_pca": repository_pca_report,
        "pca_provenance_discrepancy": True,
        "pca_gate_passed": paper_gate and max(archived_errors) < config.exact_tolerance,
        "pca_explained_variance_ratio": paper_pca.explained_variance_ratio.tolist(),
        "three_group_anova": group_anova,
        "voiced_minus_unvoiced_stops": {
            "mean_difference": float(np.mean(voiced - unvoiced)),
            "t": float(voicing_test.statistic), "p_two_sided": float(voicing_test.pvalue),
        },
        "archived_r_input_relationship": {
            "correlation": archived_r_correlation,
            "slope": float(archived_r_fit[0]), "offset": float(archived_r_fit[1]),
            "interpretation": "R CSV inputs are authoritative for published tests; archived score MAT is the exact repository-script score.",
        },
        "original_acoustic": {
            "source": "original_repository_stored_acoustic_matrix", "raw_75_syllable_wavs_available": False,
            "maximum_derivative_latency_vs_itpc4": {
                "pearson_r": float(derivative_relation.statistic), "pearson_p": float(derivative_relation.pvalue),
                "spearman_rho": float(derivative_spearman.statistic), "spearman_p": float(derivative_spearman.pvalue),
            },
            "pca_explained_variance_ratio": acoustic_pca.explained_variance_ratio.tolist(),
            "acoustic_pc1_vs_archived_itpc1_abs_r": abs(float(acoustic_pc_relation.statistic)),
            "acoustic_pc1_vs_archived_itpc1_p": float(acoustic_pc_relation.pvalue),
            "syllable_rows": int(sound_ordered.shape[0]),
        },
        "independent_stream_reconstruction": {
            "source": "45_continuous_stream_wavs", "fallback_streams": int(reconstructed_table["fallback"].sum()),
            "interpretation": "robustness extension; not an exact reconstruction of the original 75-syllable marker analysis",
        },
        "acoustic_provenance": "original_repository_stored_acoustic_matrix",
        "acoustic_fallback_streams": int(reconstructed_table["fallback"].sum()),
    }
    atomic_write_json(paths["report"], report)
    outputs = list(paths.values())
    context.finish_stage(stage, outputs, report)
    return outputs, report
