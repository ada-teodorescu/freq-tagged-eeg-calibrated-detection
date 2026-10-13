from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.optimize import linear_sum_assignment

from ..io import discover_epoch_files, load_stream_trials
from ..runtime import RunContext, atomic_save_npz, atomic_write_json, ordered_parallel_map
from ..signal import published_fft_itpc


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float).reshape(-1)
    right = np.asarray(right, dtype=float).reshape(-1)
    if left.std() == 0 or right.std() == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def run(context: RunContext, force: bool = False) -> tuple[list, dict]:
    stage = "02_source_audit"
    config = context.config
    epochs = discover_epoch_files(config)
    context.start_stage(stage, len(epochs))
    checkpoint_dir = context.checkpoint_dir / stage
    reconstructed_all = np.empty((len(epochs), 32, 45, 4), dtype=np.float64)

    def reconstruct_participant(item):
        p_index, epoch = item
        checkpoint = checkpoint_dir / f"{epoch.participant}.npz"
        if context.checkpoint_valid(checkpoint) and not force:
            with np.load(checkpoint) as archive:
                reconstructed = archive["itpc"]
        else:
            reconstructed = np.empty((32, 45, 4), dtype=np.float64)
            for stream_id in range(1, 46):
                trials = load_stream_trials(epoch, stream_id)
                reconstructed[:, stream_id - 1, :] = published_fft_itpc(
                    trials,
                    config.sampling_rate,
                    config.harmonics_hz,
                    crop_samples=int(round(config.crop_start_seconds * config.sampling_rate)),
                )
            atomic_save_npz(checkpoint, itpc=reconstructed)
            context.mark_checkpoint(checkpoint, {"participant": epoch.participant})
        return p_index, epoch.participant, reconstructed

    results = ordered_parallel_map(reconstruct_participant, enumerate(epochs), context.workers)
    for completed, (p_index, participant, reconstructed) in enumerate(results, start=1):
        reconstructed_all[p_index] = reconstructed
        context.progress(stage, completed, len(epochs), participant)

    stored_all = loadmat(
        config.data_root / "Matlab variables" / "Itc_per_stream.mat",
        variable_names=["itc"], verify_compressed_data_integrity=True,
    )["itc"]
    frequencies = np.arange(4, 226, dtype=float) * config.sampling_rate / 4500.0
    indices = np.asarray([int(np.argmin(np.abs(frequencies - h))) for h in config.harmonics_hz])
    stored = np.transpose(stored_all[:, :, indices, :], (0, 1, 3, 2))
    correlations = np.empty((25, 25), dtype=float)
    for public_index in range(25):
        for stored_index in range(25):
            correlations[public_index, stored_index] = _safe_correlation(
                reconstructed_all[public_index], stored[stored_index]
            )
    public_assignment, stored_assignment = linear_sum_assignment(-correlations)
    assignments = []
    assigned_max_errors = []
    for public_index, stored_index in zip(public_assignment, stored_assignment):
        difference = reconstructed_all[public_index] - stored[stored_index]
        max_error = float(np.max(np.abs(difference)))
        assigned_max_errors.append(max_error)
        assignments.append({
            "public_participant": epochs[public_index].participant,
            "public_index": int(public_index),
            "stored_index": int(stored_index),
            "correlation": float(correlations[public_index, stored_index]),
            "max_abs_difference": max_error,
        })
    identity = np.diag(correlations)
    optimal = correlations[public_assignment, stored_assignment]
    passed = bool(max(assigned_max_errors) < config.exact_tolerance)
    assignment_path = context.table_dir / "source_participant_assignment.csv"
    correlation_path = context.table_dir / "source_participant_correlation_matrix.csv"
    arrays_path = context.run_dir / "public_set_fft_harmonics.npz"
    report_path = context.run_dir / "source_provenance_gate.json"
    pd.DataFrame(assignments).to_csv(assignment_path, index=False)
    pd.DataFrame(
        correlations,
        index=[epoch.participant for epoch in epochs],
        columns=[f"stored_{index:02d}" for index in range(25)],
    ).to_csv(correlation_path)
    atomic_save_npz(arrays_path, harmonic_itpc=reconstructed_all)
    report = {
        "source": "public_set_fdt",
        "reference_source": "stored_matlab",
        "passed_exact_equivalence": passed,
        "tolerance": config.exact_tolerance,
        "identity_mean_correlation": float(identity.mean()),
        "identity_correlation_range": [float(identity.min()), float(identity.max())],
        "optimal_mean_correlation": float(optimal.mean()),
        "optimal_correlation_range": [float(optimal.min()), float(optimal.max())],
        "optimal_maximum_absolute_error": float(max(assigned_max_errors)),
        "original_script_audit": {
            "event_sorting_script": str(config.original_repo_root / "EEG_sound analyses" / "sort_data_from_eeglab.m"),
            "itpc_script": str(config.original_repo_root / "EEG_sound analyses" / "pow_itc_e2_no_ica.m"),
            "neighbor_bin_script": str(config.original_repo_root / "EEG_sound analyses" / "itc_48_no_ica.m"),
            "reproduced_operations": [
                "remove samples 1:500",
                "sort epochs by numeric S1..S45 event code",
                "reshape each stream into 10 trials",
                "apply symmetric 4500-sample Hann window",
                "use FFT bins 5:226 and squared phase-vector magnitude",
            ],
            "participant_order_risk": (
                "The MATLAB script used unsorted dir('*.set') order; the audit therefore tests every "
                "public participant against every stored participant."
            ),
        },
        "ica_metadata": {
            "sets_with_ica_weights": int(sum(epoch.has_ica_weights for epoch in epochs)),
            "sets_without_ica_weights": int(sum(not epoch.has_ica_weights for epoch in epochs)),
            "component_counts": {epoch.participant: epoch.ica_components for epoch in epochs},
            "interpretation": (
                "ICA fields describe the supplied preprocessed SET files; raw BrainVision/ICA recreation "
                "is unavailable and is not inferred from the no_ica analysis filename."
            ),
        },
        "interpretation": (
            "Sources may be treated as equivalent." if passed else
            "Stored MATLAB and public SET/FDT results remain separate provenance branches."
        ),
    }
    atomic_write_json(report_path, report)
    outputs = [assignment_path, correlation_path, arrays_path, report_path]
    context.finish_stage(stage, outputs, report)
    return outputs, report
