from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from ..config import CONDITIONS
from ..io import discover_epoch_files, load_stream_trials
from ..runtime import RunContext, atomic_save_npz, atomic_write_json
from ..signal import hilbert_itpc
from ..statistics import PCAResult, project_pca, rm_anova_oneway
from .published_results import GROUPS


def run(context: RunContext, force: bool = False) -> tuple[list, dict]:
    stage = "04_hilbert"
    config = context.config
    epochs = discover_epoch_files(config)
    context.start_stage(stage, len(epochs))
    ordinary_all = np.empty((25, 45, 32, 4), dtype=np.float32)
    squared_all = np.empty_like(ordinary_all)
    trace_sum = None
    checkpoint_dir = context.checkpoint_dir / stage
    for p_index, epoch in enumerate(epochs):
        checkpoint = checkpoint_dir / f"{epoch.participant}.npz"
        if context.checkpoint_valid(checkpoint) and not force:
            with np.load(checkpoint) as archive:
                ordinary = archive["ordinary"]
                squared = archive["squared"]
                traces = archive["traces"]
        else:
            ordinary = np.empty((45, 32, 4), dtype=np.float32)
            squared = np.empty_like(ordinary)
            traces = None
            for stream_id in range(1, 46):
                trials = load_stream_trials(epoch, stream_id)
                stream_traces = []
                for h_index, harmonic in enumerate(config.harmonics_hz):
                    mean_r, mean_r_squared, time_trace = hilbert_itpc(
                        trials,
                        config.sampling_rate,
                        harmonic,
                        config.hilbert_half_bandwidth_hz,
                        config.hilbert_filter_order,
                        config.crop_start_seconds,
                    )
                    ordinary[stream_id - 1, :, h_index] = mean_r
                    squared[stream_id - 1, :, h_index] = mean_r_squared
                    stream_traces.append(time_trace.astype(np.float32))
                stacked_trace = np.stack(stream_traces)
                if traces is None:
                    traces = np.empty((45, 4, stacked_trace.shape[-1]), dtype=np.float32)
                traces[stream_id - 1] = stacked_trace
            atomic_save_npz(checkpoint, ordinary=ordinary, squared=squared, traces=traces)
            context.mark_checkpoint(checkpoint, {"participant": epoch.participant})
        ordinary_all[p_index] = ordinary
        squared_all[p_index] = squared
        trace_sum = traces.astype(np.float64) if trace_sum is None else trace_sum + traces
        context.progress(stage, p_index + 1, len(epochs), epoch.participant)

    ordinary_condition = ordinary_all.mean(axis=2).reshape(25, 15, 3, 4).mean(axis=2)
    squared_condition = squared_all.mean(axis=2).reshape(25, 15, 3, 4).mean(axis=2)
    with np.load(context.run_dir / "stored_fft_harmonics.npz") as archive:
        stored_stream = archive["harmonic_itpc"]
    stored_condition = stored_stream.reshape(25, 15, 3, 4).mean(axis=2)
    comparisons = []
    for h_index, harmonic in enumerate(config.harmonics_hz):
        fft = stored_condition[:, :, h_index]
        hilbert = squared_condition[:, :, h_index]
        comparisons.append({
            "harmonic_hz": harmonic,
            "fft_mean_squared_itpc": float(fft.mean()),
            "hilbert_mean_squared_itpc": float(hilbert.mean()),
            "participant_condition_pearson_r": float(stats.pearsonr(fft.ravel(), hilbert.ravel()).statistic),
            "participant_condition_spearman_rho": float(stats.spearmanr(fft.ravel(), hilbert.ravel()).statistic),
            "condition_mean_pearson_r": float(stats.pearsonr(fft.mean(0), hilbert.mean(0)).statistic),
            "condition_mean_spearman_rho": float(stats.spearmanr(fft.mean(0), hilbert.mean(0)).statistic),
        })

    pca_json = json.loads((context.run_dir / "published_pca.json").read_text(encoding="utf-8"))
    pca = PCAResult(
        mean=np.asarray(pca_json["mean"]),
        loadings=np.asarray(pca_json["loadings"]),
        scores=np.empty((0, 4)),
        explained_variance_ratio=np.asarray(pca_json["explained_variance_ratio"]),
    )
    hilbert_scores = project_pca(squared_condition, pca)
    group_values = np.column_stack([
        hilbert_scores[:, np.asarray(ids) - 1, 0].mean(axis=1) for ids in GROUPS.values()
    ])
    group_anova = rm_anova_oneway(group_values)
    source_gate = json.loads((context.run_dir / "source_provenance_gate.json").read_text(encoding="utf-8"))

    condition_rows = []
    for participant in range(25):
        for condition in range(15):
            condition_rows.append({
                "participant_index": participant,
                "condition_id": condition + 1,
                "condition": CONDITIONS[condition],
                "pc1_projected": float(hilbert_scores[participant, condition, 0]),
                "pc2_projected": float(hilbert_scores[participant, condition, 1]),
                **{
                    f"ordinary_itpc_{int(h)}hz": float(ordinary_condition[participant, condition, i])
                    for i, h in enumerate(config.harmonics_hz)
                },
                **{
                    f"mean_squared_itpc_{int(h)}hz": float(squared_condition[participant, condition, i])
                    for i, h in enumerate(config.harmonics_hz)
                },
            })
    trace = trace_sum / 25.0 / 45.0
    time = np.arange(trace.shape[-1]) / config.sampling_rate + config.crop_start_seconds
    trace_rows = []
    for index, second in enumerate(time):
        for h_index, harmonic in enumerate(config.harmonics_hz):
            trace_rows.append({
                "time_seconds": second,
                "harmonic_hz": harmonic,
                "mean_time_resolved_itpc": float(trace[:, h_index, index].mean()),
            })
    comparison_path = context.table_dir / "hilbert_fft_comparison.csv"
    condition_path = context.table_dir / "hilbert_condition_scores.csv"
    trace_path = context.table_dir / "hilbert_time_resolved_mean.csv"
    report_path = context.run_dir / "hilbert_robustness.json"
    figure_path = context.figure_dir / "hilbert_fft_condition_means.png"
    pd.DataFrame(comparisons).to_csv(comparison_path, index=False)
    pd.DataFrame(condition_rows).to_csv(condition_path, index=False)
    pd.DataFrame(trace_rows).to_csv(trace_path, index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(8, 7), constrained_layout=True)
    for h_index, (axis, harmonic) in enumerate(zip(axes.ravel(), config.harmonics_hz)):
        axis.scatter(stored_condition[:, :, h_index].mean(0), squared_condition[:, :, h_index].mean(0), s=25)
        axis.set_title(f"{harmonic:g} Hz")
        axis.set_xlabel("FFT squared ITPC")
        axis.set_ylabel("Hilbert mean squared ITPC")
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    report = {
        "source": "public_set_fdt",
        "comparison_source": "stored_matlab",
        "source_equivalence_passed": source_gate["passed_exact_equivalence"],
        "interpretation_level": (
            "numerical_and_group" if source_gate["passed_exact_equivalence"] else "group_level_only"
        ),
        "method": "zero-phase Butterworth, time-resolved trial phase coherence",
        "filter_order": config.hilbert_filter_order,
        "half_bandwidth_hz": config.hilbert_half_bandwidth_hz,
        "edge_crop_seconds": config.crop_start_seconds,
        "ordinary_mean_by_harmonic": ordinary_all.mean(axis=(0, 1, 2)).astype(float).tolist(),
        "mean_squared_by_harmonic": squared_all.mean(axis=(0, 1, 2)).astype(float).tolist(),
        "three_group_anova": group_anova,
    }
    atomic_write_json(report_path, report)
    outputs = [comparison_path, condition_path, trace_path, report_path, figure_path]
    context.finish_stage(stage, outputs, report)
    return outputs, report
