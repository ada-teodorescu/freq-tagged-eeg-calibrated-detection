from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.io import loadmat

from ..runtime import RunContext, atomic_save_npz, atomic_write_json
from ..signal import squared_itpc


def run(context: RunContext, force: bool = False) -> tuple[list, dict]:
    stage = "01_stored_fft"
    config = context.config
    context.start_stage(stage, config.expected_participants)
    matlab = config.data_root / "Matlab variables"
    coefficients_all = loadmat(
        matlab / "Data_fft_all.mat", variable_names=["Data_fft"],
        verify_compressed_data_integrity=True,
    )["Data_fft"]
    reference_all = loadmat(
        matlab / "Itc_per_stream.mat", variable_names=["itc"],
        verify_compressed_data_integrity=True,
    )["itc"]
    statistics = loadmat(matlab / "Itc_for_stats.mat", squeeze_me=True)
    expected_fft_shape = (25, 32, 222, 45, 10)
    expected_itpc_shape = (25, 32, 222, 45)
    if coefficients_all.shape != expected_fft_shape or reference_all.shape != expected_itpc_shape:
        raise ValueError(f"Unexpected stored shapes: {coefficients_all.shape}, {reference_all.shape}")
    frequencies = np.arange(4, 226, dtype=float) * config.sampling_rate / 4500.0
    harmonic_indices = np.asarray([
        int(np.argmin(np.abs(frequencies - frequency))) for frequency in config.harmonics_hz
    ])
    harmonic_itpc = np.empty((25, 45, 4), dtype=np.float64)
    rows = []
    checkpoint_dir = context.checkpoint_dir / stage
    for participant in range(25):
        checkpoint = checkpoint_dir / f"participant_{participant:02d}.npz"
        if context.checkpoint_valid(checkpoint) and not force:
            with np.load(checkpoint) as archive:
                participant_harmonics = archive["harmonic_itpc"]
                max_difference = float(archive["max_difference"])
                rmse = float(archive["rmse"])
        else:
            reconstructed = squared_itpc(coefficients_all[participant], axis=-1)
            difference = reconstructed - reference_all[participant]
            max_difference = float(np.nanmax(np.abs(difference)))
            rmse = float(np.sqrt(np.nanmean(difference ** 2)))
            selected = reconstructed[:, harmonic_indices, :].mean(axis=0).T
            participant_harmonics = selected
            atomic_save_npz(
                checkpoint,
                harmonic_itpc=participant_harmonics,
                max_difference=max_difference,
                rmse=rmse,
            )
            context.mark_checkpoint(checkpoint, {"participant_index": participant})
        harmonic_itpc[participant] = participant_harmonics
        rows.append({
            "participant_index": participant,
            "max_abs_difference": max_difference,
            "rmse": rmse,
        })
        context.progress(stage, participant + 1, 25, f"stored participant {participant + 1}")

    stats_differences = {}
    condition_differences = {}
    for h_index, harmonic in enumerate(config.harmonics_hz):
        stored_stream = np.asarray(statistics[f"Itc{int(harmonic)}"], dtype=float).T
        stats_differences[str(int(harmonic))] = float(
            np.max(np.abs(harmonic_itpc[:, :, h_index] - stored_stream))
        )
        condition = harmonic_itpc[:, :, h_index].reshape(25, 15, 3).mean(axis=2)
        stored_condition = np.asarray(statistics[f"meanitc{int(harmonic)}"], dtype=float).T
        condition_differences[str(int(harmonic))] = float(np.max(np.abs(condition - stored_condition)))
    table_path = context.table_dir / "stored_fft_agreement.csv"
    arrays_path = context.run_dir / "stored_fft_harmonics.npz"
    report_path = context.run_dir / "stored_fft_gate.json"
    pd.DataFrame(rows).to_csv(table_path, index=False)
    atomic_save_npz(
        arrays_path,
        harmonic_itpc=harmonic_itpc,
        frequencies_hz=frequencies,
        harmonic_indices=harmonic_indices,
    )
    maximum = max(
        max(row["max_abs_difference"] for row in rows),
        max(stats_differences.values()),
        max(condition_differences.values()),
    )
    report = {
        "source": "stored_matlab",
        "passed": bool(maximum < config.exact_tolerance),
        "tolerance": config.exact_tolerance,
        "maximum_absolute_error": maximum,
        "data_fft_to_itpc_maximum": max(row["max_abs_difference"] for row in rows),
        "itpc_for_stats_maximum_by_harmonic": stats_differences,
        "condition_maximum_by_harmonic": condition_differences,
        "harmonic_frequencies_hz": [float(frequencies[index]) for index in harmonic_indices],
    }
    atomic_write_json(report_path, report)
    if not report["passed"]:
        raise AssertionError(f"Stored FFT replication gate failed: maximum error {maximum}")
    outputs = [table_path, arrays_path, report_path]
    context.finish_stage(stage, outputs, report)
    return outputs, report
