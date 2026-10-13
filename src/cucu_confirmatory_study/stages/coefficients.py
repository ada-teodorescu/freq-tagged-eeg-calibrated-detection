from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from cucu_accuracy_study.features import exact_harmonic_coefficients
from cucu_article.runtime import atomic_save_npz, atomic_write_json
from cucu_detection.estimators import hilbert_coefficients
from cucu_replication.io import inspect_set, load_stream_trials, natural_subject_key

from ..runtime import RunContext


def feature_path(context: RunContext, participant: str) -> Path:
    return context.checkpoint_dir / "01_coefficients" / f"{participant}.npz"


def _participant(context: RunContext, set_path: Path, force: bool) -> tuple[str, Path]:
    config = context.config
    epoch = inspect_set(set_path)
    destination = feature_path(context, epoch.participant)
    if context.checkpoint_valid(destination) and not force:
        return epoch.participant, destination

    n_harmonics = len(config.harmonics_hz)
    hann_matched = np.empty((config.streams, config.trials_per_stream, config.channels, n_harmonics), dtype=np.complex64)
    hann_full = np.empty_like(hann_matched)
    hilbert = np.empty_like(hann_matched)
    matched_duration = config.matched_window_end_seconds - config.matched_window_start_seconds
    full_duration = config.full_window_end_seconds - config.full_window_start_seconds
    harmonics = np.asarray(config.harmonics_hz, dtype=float)

    for stream in range(1, config.streams + 1):
        raw = load_stream_trials(epoch, stream, dtype=np.float32)
        cropped = raw[..., config.onset_crop_samples:]
        # H1: fair-window Hann, matching Hilbert's [0.5, 4.5]s retained window exactly.
        hann_matched[stream - 1] = exact_harmonic_coefficients(
            cropped, config.sampling_rate, config.harmonics_hz, 0.0, matched_duration,
        )
        # H3: full-window Hann, [0.5, 5.0]s -- the historical S2/S4/C4 convention.
        hann_full[stream - 1] = exact_harmonic_coefficients(
            cropped, config.sampling_rate, config.harmonics_hz, 0.0, full_duration,
        )
        # H2: corrected Hilbert on the FULL uncropped epoch -- hilbert_coefficients does its
        # own zero-phase filtfilt and symmetric edge crop, and needs untruncated data on both
        # sides to avoid filtfilt edge artifacts.
        hilbert[stream - 1] = hilbert_coefficients(
            raw, config.sampling_rate, harmonics,
            config.hilbert_order, config.hilbert_half_bandwidth_hz, config.hilbert_edge_crop_seconds,
        )

    atomic_save_npz(
        destination, hann_matched=hann_matched, hann_full=hann_full, hilbert=hilbert,
        participant=np.asarray(epoch.participant), channel_labels=np.asarray(epoch.channel_labels),
        harmonics_hz=harmonics,
    )
    context.mark_checkpoint(destination, {"participant": epoch.participant, "natural_trial_order": True})
    return epoch.participant, destination


def run(context: RunContext, force: bool = False) -> tuple[list[Path], dict]:
    stage, config = "01_coefficients", context.config
    paths = sorted((config.data_root / "Pre-processed EEGLAB datasets").glob("*.set"), key=natural_subject_key)
    context.start_stage(stage, len(paths))
    outputs: list[Path] = []
    if context.workers == 1:
        for index, path in enumerate(paths, 1):
            participant, output = _participant(context, path, force)
            outputs.append(output); context.progress(stage, index, len(paths), participant)
    else:
        with ThreadPoolExecutor(max_workers=context.workers) as executor:
            futures = [executor.submit(_participant, context, path, force) for path in paths]
            for index, future in enumerate(as_completed(futures), 1):
                participant, output = future.result()
                outputs.append(output); context.progress(stage, index, len(paths), participant)
    summary = {"participants": len(outputs),
               "windows": {"matched": [config.matched_window_start_seconds, config.matched_window_end_seconds],
                           "full": [config.full_window_start_seconds, config.full_window_end_seconds]}}
    report = context.run_dir / "coefficient_features.json"
    atomic_write_json(report, summary); outputs.append(report)
    context.finish_stage(stage, outputs, summary)
    return outputs, summary
