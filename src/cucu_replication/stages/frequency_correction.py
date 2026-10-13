from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from ..config import CONDITIONS
from ..io import discover_epoch_files, load_stream_trials
from ..runtime import RunContext, atomic_save_npz, atomic_write_json
from ..signal import hann_scalloping_loss, squared_itpc, weighted_sinusoid_coefficients
from ..statistics import PCAResult, project_pca, rm_anova_oneway
from .published_results import GROUPS


def _joint_regression_test(x: np.ndarray, y: np.ndarray, iterations: int, seed: int) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    design = np.column_stack([np.ones(x.shape[0]), x])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = design @ beta
    denominator = np.sum((y - y.mean()) ** 2)
    r_squared = float(1.0 - np.sum((y - fitted) ** 2) / denominator)
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(iterations):
        permuted = rng.permutation(y)
        permuted_fit = design @ np.linalg.lstsq(design, permuted, rcond=None)[0]
        permuted_r2 = 1.0 - np.sum((permuted - permuted_fit) ** 2) / np.sum((permuted - permuted.mean()) ** 2)
        if permuted_r2 >= r_squared:
            extreme += 1
    bootstraps = np.empty((5000, beta.size), dtype=float)
    for index in range(bootstraps.shape[0]):
        sample = rng.integers(0, y.size, y.size)
        bootstraps[index] = np.linalg.lstsq(design[sample], y[sample], rcond=None)[0]
    return {
        "coefficients": beta.tolist(),
        "coefficient_95_ci": np.quantile(bootstraps, [0.025, 0.975], axis=0).T.tolist(),
        "r_squared": r_squared,
        "p_permutation": (extreme + 1) / (iterations + 1),
    }


def run(context: RunContext, force: bool = False) -> tuple[list, dict]:
    stage = "06_frequency_correction"
    config = context.config
    epochs = discover_epoch_files(config)
    total = 45 + len(epochs)
    context.start_stage(stage, total)
    acoustic_checkpoint = context.checkpoint_dir / "03_published_results" / "acoustics"
    frequency_rows = []
    fitted_frequencies = np.empty(45, dtype=float)
    completed = 0
    for stream_id in range(1, 46):
        checkpoint = acoustic_checkpoint / f"stream_{stream_id:02d}.npz"
        if not context.checkpoint_valid(checkpoint):
            raise FileNotFoundError(f"Missing acoustic checkpoint {checkpoint}; run 03_published_results")
        with np.load(checkpoint) as archive:
            names = [str(name) for name in archive["feature_names"]]
            features = dict(zip(names, archive["feature_values"].astype(float)))
            frequency = float(features["fitted_frequency_hz"])
            fitted_frequencies[stream_id - 1] = frequency
            delta = frequency - 4.0
            row = {
                "stream_id": stream_id,
                "condition_id": int(archive["condition_id"]),
                "condition": str(archive["condition"]),
                "fitted_frequency_hz": frequency,
                "delta_from_4hz": delta,
                "mean_ioi_seconds": float(features["mean_ioi_seconds"]),
                "sd_ioi_seconds": float(features["sd_ioi_seconds"]),
                "fallback": bool(archive["fallback"]),
            }
            for h_index, harmonic in enumerate(config.harmonics_hz, start=1):
                row[f"predicted_hann_power_ratio_h{h_index}"] = float(
                    hann_scalloping_loss(h_index * delta, 4.5)
                )
            frequency_rows.append(row)
        completed += 1
        context.progress(stage, completed, total, f"frequency stream {stream_id}")

    checkpoint_dir = context.checkpoint_dir / stage / "eeg"
    nominal_all = np.empty((25, 45, 32, 4), dtype=np.float32)
    corrected_all = np.empty_like(nominal_all)
    crop_samples = int(round(config.crop_start_seconds * config.sampling_rate))
    for p_index, epoch in enumerate(epochs):
        checkpoint = checkpoint_dir / f"{epoch.participant}.npz"
        if context.checkpoint_valid(checkpoint) and not force:
            with np.load(checkpoint) as archive:
                nominal = archive["nominal"]
                corrected = archive["corrected"]
        else:
            nominal = np.empty((45, 32, 4), dtype=np.float32)
            corrected = np.empty_like(nominal)
            for stream_id in range(1, 46):
                trials = load_stream_trials(epoch, stream_id)
                true_fundamental = fitted_frequencies[stream_id - 1]
                for h_index in range(4):
                    harmonic_number = h_index + 1
                    nominal_coefficients = weighted_sinusoid_coefficients(
                        trials, config.sampling_rate, harmonic_number * 4.0, crop_samples
                    )
                    corrected_coefficients = weighted_sinusoid_coefficients(
                        trials, config.sampling_rate, harmonic_number * true_fundamental, crop_samples
                    )
                    nominal[stream_id - 1, :, h_index] = squared_itpc(nominal_coefficients, axis=0)
                    corrected[stream_id - 1, :, h_index] = squared_itpc(corrected_coefficients, axis=0)
            atomic_save_npz(checkpoint, nominal=nominal, corrected=corrected)
            context.mark_checkpoint(checkpoint, {"participant": epoch.participant})
        nominal_all[p_index] = nominal
        corrected_all[p_index] = corrected
        completed += 1
        context.progress(stage, completed, total, epoch.participant)

    nominal_condition = nominal_all.mean(axis=2).reshape(25, 15, 3, 4).mean(axis=2)
    corrected_condition = corrected_all.mean(axis=2).reshape(25, 15, 3, 4).mean(axis=2)
    pca_json = json.loads((context.run_dir / "published_pca.json").read_text(encoding="utf-8"))
    pca = PCAResult(
        mean=np.asarray(pca_json["mean"]),
        loadings=np.asarray(pca_json["loadings"]),
        scores=np.empty((0, 4)),
        explained_variance_ratio=np.asarray(pca_json["explained_variance_ratio"]),
    )
    nominal_scores = project_pca(nominal_condition, pca)
    corrected_scores = project_pca(corrected_condition, pca)
    nominal_groups = np.column_stack([
        nominal_scores[:, np.asarray(ids) - 1, 0].mean(axis=1) for ids in GROUPS.values()
    ])
    corrected_groups = np.column_stack([
        corrected_scores[:, np.asarray(ids) - 1, 0].mean(axis=1) for ids in GROUPS.values()
    ])
    frequency_table = pd.DataFrame(frequency_rows)
    frequency_conditions = frequency_table.groupby(["condition_id", "condition"], as_index=False).mean(numeric_only=True)
    published = pd.read_csv(context.table_dir / "published_condition_scores.csv")
    published_pc1 = published.groupby("condition_id")["pc1"].mean().sort_index().to_numpy()
    predictors = frequency_conditions.sort_values("condition_id")[[
        "delta_from_4hz", "predicted_hann_power_ratio_h1"
    ]].to_numpy()
    regression = _joint_regression_test(
        predictors,
        published_pc1,
        config.permutation_iterations,
        config.seed,
    )
    ordering_correlation = stats.spearmanr(
        nominal_scores.mean(axis=0)[:, 0], corrected_scores.mean(axis=0)[:, 0]
    )
    score_rows = []
    for participant in range(25):
        for condition in range(15):
            score_rows.append({
                "participant_index": participant,
                "condition_id": condition + 1,
                "condition": CONDITIONS[condition],
                "nominal_pc1": float(nominal_scores[participant, condition, 0]),
                "corrected_pc1": float(corrected_scores[participant, condition, 0]),
                "pc1_change": float(corrected_scores[participant, condition, 0] - nominal_scores[participant, condition, 0]),
                **{
                    f"nominal_itpc_{int(h)}hz": float(nominal_condition[participant, condition, i])
                    for i, h in enumerate(config.harmonics_hz)
                },
                **{
                    f"corrected_itpc_{int(h)}hz": float(corrected_condition[participant, condition, i])
                    for i, h in enumerate(config.harmonics_hz)
                },
            })
    frequency_path = context.table_dir / "stream_frequency_estimates.csv"
    score_path = context.table_dir / "frequency_corrected_condition_scores.csv"
    report_path = context.run_dir / "frequency_correction.json"
    figure_path = context.figure_dir / "frequency_correction.png"
    frequency_table.to_csv(frequency_path, index=False)
    pd.DataFrame(score_rows).to_csv(score_path, index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    axes[0].bar(frequency_conditions["condition"], frequency_conditions["delta_from_4hz"])
    axes[0].tick_params(axis="x", rotation=60)
    axes[0].set_ylabel("Estimated frequency minus 4 Hz")
    axes[1].scatter(nominal_scores.mean(0)[:, 0], corrected_scores.mean(0)[:, 0])
    axes[1].set_xlabel("Nominal-frequency PC1")
    axes[1].set_ylabel("Corrected-frequency PC1")
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    report = {
        "audio_source": "target_stream_wavs_1_to_45",
        "eeg_source": "public_set_fdt",
        "mean_fitted_frequency_hz": float(fitted_frequencies.mean()),
        "frequency_range_hz": [float(fitted_frequencies.min()), float(fitted_frequencies.max())],
        "fallback_streams": int(frequency_table["fallback"].sum()),
        "published_pc1_frequency_regression": regression,
        "nominal_group_anova": rm_anova_oneway(nominal_groups),
        "corrected_group_anova": rm_anova_oneway(corrected_groups),
        "nominal_corrected_condition_order_spearman": float(ordering_correlation.statistic),
        "nominal_corrected_condition_order_p": float(ordering_correlation.pvalue),
        "mean_pc1_change": float(np.mean(corrected_scores - nominal_scores, axis=(0, 1))[0]),
        "interpretation": (
            "frequency_offset_associated_with_published_condition_pattern"
            if regression["p_permutation"] < 0.05 else
            "published_condition_pattern_robust_to_frequency_offset_test"
        ),
    }
    atomic_write_json(report_path, report)
    outputs = [frequency_path, score_path, report_path, figure_path]
    context.finish_stage(stage, outputs, report)
    return outputs, report
