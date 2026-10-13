from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import signal

from ..io import discover_epoch_files, load_stream_trials
from ..runtime import RunContext, atomic_save_npz, atomic_write_json
from ..signal import hilbert_time_resolved_itpc, phase_randomize_trials
from ..statistics import wilson_interval


def _configurations(context: RunContext) -> list[tuple[float, int, float, float]]:
    config = context.config
    combinations = {
        (4.0, order, bandwidth, crop)
        for order, bandwidth, crop in itertools.product(
            config.null_orders,
            config.null_half_bandwidths_hz,
            config.null_edge_crops_seconds,
        )
    }
    combinations.update({
        (harmonic, config.hilbert_filter_order, config.hilbert_half_bandwidth_hz, config.crop_start_seconds)
        for harmonic in config.harmonics_hz
    })
    return sorted(combinations)


def _null_threshold(k_trials: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(1_000_000, k_trials))
    values = np.abs(np.mean(np.exp(1j * phases), axis=1)) ** 2
    return float(np.quantile(values, 0.95))


def run(context: RunContext, force: bool = False) -> tuple[list, dict]:
    stage = "05_filter_null"
    config = context.config
    epochs = discover_epoch_files(config)
    configurations = _configurations(context)
    unique_filters = sorted({item[:3] for item in configurations})
    total = len(epochs) * config.null_batches
    context.start_stage(stage, total)
    checkpoint_dir = context.checkpoint_dir / stage
    threshold = _null_threshold(config.trials_per_stream, config.seed)
    completed = 0
    records = []
    for p_index, epoch in enumerate(epochs):
        for batch in range(config.null_batches):
            checkpoint = checkpoint_dir / epoch.participant / f"batch_{batch:02d}.npz"
            if context.checkpoint_valid(checkpoint) and not force:
                with np.load(checkpoint) as archive:
                    mean_squared = archive["mean_squared"]
                    exceedances = archive["exceedances"].astype(np.int64)
                    pointwise_cells = archive["pointwise_cells"].astype(np.int64)
            else:
                squared_sums = np.zeros(len(configurations), dtype=np.float64)
                exceedances = np.zeros(len(configurations), dtype=np.int64)
                pointwise_cells = np.zeros(len(configurations), dtype=np.int64)
                seed = config.seed + p_index * 1_000_003 + batch * 10_007
                rng = np.random.default_rng(seed)
                for stream_id in range(1, 46):
                    trials = load_stream_trials(epoch, stream_id)
                    downsampled = signal.resample_poly(
                        trials,
                        config.null_sampling_rate,
                        config.sampling_rate,
                        axis=-1,
                    )
                    randomized = phase_randomize_trials(downsampled, rng)
                    filter_results = {
                        filter_key: hilbert_time_resolved_itpc(
                            randomized,
                            config.null_sampling_rate,
                            filter_key[0],
                            filter_key[2],
                            filter_key[1],
                            0.0,
                        )
                        for filter_key in unique_filters
                    }
                    for c_index, (frequency, order, bandwidth, crop) in enumerate(configurations):
                        time_itpc = filter_results[(frequency, order, bandwidth)]
                        crop_samples = int(round(crop * config.null_sampling_rate))
                        if crop_samples:
                            time_itpc = time_itpc[:, crop_samples:-crop_samples]
                        squared = np.square(time_itpc)
                        squared_sums[c_index] += float(squared.sum())
                        pointwise_cells[c_index] += int(squared.size)
                        exceedances[c_index] += int(np.count_nonzero(squared >= threshold))
                mean_squared = squared_sums / pointwise_cells
                atomic_save_npz(
                    checkpoint,
                    mean_squared=mean_squared,
                    exceedances=exceedances,
                    pointwise_cells=pointwise_cells,
                )
                context.mark_checkpoint(checkpoint, {
                    "participant": epoch.participant,
                    "batch": batch,
                    "seed": seed,
                    "sampling_rate": config.null_sampling_rate,
                })
            for c_index, (frequency, order, bandwidth, crop) in enumerate(configurations):
                successes = int(exceedances[c_index])
                cells = int(pointwise_cells[c_index])
                low, high = wilson_interval(successes, cells)
                records.append({
                    "participant": epoch.participant,
                    "participant_index": p_index,
                    "frequency_hz": frequency,
                    "filter_order": order,
                    "half_bandwidth_hz": bandwidth,
                    "edge_crop_seconds": crop,
                    "batch": batch,
                    "mean_squared_itpc": float(mean_squared[c_index]),
                    "chance_mean_squared_itpc": 1.0 / config.trials_per_stream,
                    "threshold_95": threshold,
                    "false_positive_rate": successes / cells,
                    "fpr_ci_low": low,
                    "fpr_ci_high": high,
                    "flag_ci_excludes_0p05": bool(high < 0.05 or low > 0.05),
                    "cells": cells,
                })
            completed += 1
            context.progress(stage, completed, total, f"{epoch.participant} batch {batch + 1}")

    rows = pd.DataFrame(records)
    keys = ["frequency_hz", "filter_order", "half_bandwidth_hz", "edge_crop_seconds"]
    rows["exceedance_count"] = np.rint(rows["false_positive_rate"] * rows["cells"]).astype(np.int64)
    summary = rows.groupby(keys, as_index=False).agg(
        mean_squared_itpc=("mean_squared_itpc", "mean"),
        exceedances=("exceedance_count", "sum"),
        batches=("batch", "count"),
        cells=("cells", "sum"),
    )
    summary["false_positive_rate"] = summary["exceedances"] / summary["cells"]
    intervals = [wilson_interval(int(row.exceedances), int(row.cells)) for row in summary.itertuples(index=False)]
    summary["fpr_ci_low"] = [interval[0] for interval in intervals]
    summary["fpr_ci_high"] = [interval[1] for interval in intervals]
    summary["flag_ci_excludes_0p05"] = (summary["fpr_ci_high"] < 0.05) | (summary["fpr_ci_low"] > 0.05)
    cells_path = context.table_dir / "filter_null_cells.csv"
    summary_path = context.table_dir / "filter_null_summary.csv"
    report_path = context.run_dir / "filter_null.json"
    figure_path = context.figure_dir / "filter_null_fpr.png"
    rows.to_csv(cells_path, index=False)
    summary.to_csv(summary_path, index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    four_hz = summary[summary["frequency_hz"] == 4.0].copy()
    labels = [
        f"o{int(row.filter_order)} bw{row.half_bandwidth_hz:g} c{row.edge_crop_seconds:g}"
        for row in four_hz.itertuples(index=False)
    ]
    figure, axis = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    axis.plot(np.arange(len(four_hz)), four_hz["false_positive_rate"], "o")
    axis.axhline(0.05, color="black", linestyle="--", linewidth=1)
    axis.set_xticks(np.arange(len(four_hz)), labels, rotation=60, ha="right")
    axis.set_ylabel("Pointwise empirical false-positive rate")
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    report = {
        "source": "public_set_fdt_phase_randomized",
        "phase_randomization": "independent per trial and channel after anti-aliased downsampling",
        "sampling_rate": config.null_sampling_rate,
        "chance_mean_squared_itpc": 1.0 / config.trials_per_stream,
        "pointwise_empirical_95_percent_threshold": threshold,
        "configurations": len(configurations),
        "unique_filters": len(unique_filters),
        "randomization_batches": config.null_batches,
        "flagged_configurations": int(summary["flag_ci_excludes_0p05"].sum()),
    }
    atomic_write_json(report_path, report)
    outputs = [cells_path, summary_path, report_path, figure_path]
    context.finish_stage(stage, outputs, report)
    return outputs, report
