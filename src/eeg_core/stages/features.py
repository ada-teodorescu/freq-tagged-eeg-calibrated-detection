from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np

from eeg_detection.detectors import deterministic_subsets
from eeg_replication.io import inspect_set, load_stream_trials, natural_subject_key
from ..runtime import RunContext, atomic_save_npz, atomic_write_json
from ..spectral import hann_coefficients, score_bundle


def feature_path(context: RunContext, participant: str, duration: float, k: int) -> Path:
    token = str(duration).replace(".", "p")
    return context.checkpoint_dir / "02_duration_features" / participant / f"duration_{token}_k{k:02d}.npz"


def _participant(context: RunContext, set_path: Path, force: bool) -> tuple[str, list[Path]]:
    config = context.config
    epoch = inspect_set(set_path)
    paths = [feature_path(context, epoch.participant, duration, k)
             for duration in config.durations_seconds for k in config.k_values]
    if not force and all(context.checkpoint_valid(path) for path in paths):
        return epoch.participant, paths
    subsets = {k: deterministic_subsets(config.trials_per_stream, k, config.subset_cap, config.seed)
               for k in config.k_values}
    buffers: dict[tuple[float, int], dict[str, np.ndarray]] = {}
    for duration in config.durations_seconds:
        for k in config.k_values:
            draws = len(subsets[k])
            buffers[(duration, k)] = {
                "positive_phase": np.empty((config.streams, draws, 4), np.float32),
                "null_phase": np.empty((config.streams, draws, config.neighbor_bins_per_harmonic, 4), np.float32),
                "positive_hotelling": np.empty((config.streams, draws), np.float32),
                "null_hotelling": np.empty((config.streams, draws, config.neighbor_bins_per_harmonic), np.float32),
                "positive_joint": np.empty((config.streams, draws), np.float32),
                "null_joint": np.empty((config.streams, draws, config.neighbor_bins_per_harmonic), np.float32),
            }
    for stream in range(1, config.streams + 1):
        trials = load_stream_trials(epoch, stream)[..., config.onset_crop_samples:]
        for duration in config.durations_seconds:
            target, neighbors = hann_coefficients(
                trials, config.sampling_rate, duration, config.fft_pad_samples,
                config.harmonics_hz, config.neighbor_bins_per_harmonic,
            )
            for k in config.k_values:
                bundle = score_bundle(target, neighbors, subsets[k])
                for name, values in bundle.items(): buffers[(duration, k)][name][stream - 1] = values
    for (duration, k), arrays in buffers.items():
        path = feature_path(context, epoch.participant, duration, k)
        atomic_save_npz(path, **arrays, subsets=subsets[k], duration_seconds=np.asarray(duration), k=np.asarray(k))
        context.mark_checkpoint(path, {
            "participant": epoch.participant, "duration_seconds": duration, "k": k,
            "source": str(epoch.set_path), "zero_padding_samples": config.fft_pad_samples,
            "window": "numpy.hanning", "channel_operation": "statistic_then_mean",
        })
    return epoch.participant, paths


def run(context: RunContext, force: bool = False) -> tuple[list, dict]:
    stage = "02_duration_features"
    paths = sorted((context.config.data_root / "Pre-processed EEGLAB datasets").glob("*.set"), key=natural_subject_key)
    context.start_stage(stage, len(paths))
    outputs: list[Path] = []
    if context.workers == 1:
        for index, set_path in enumerate(paths, 1):
            participant, participant_paths = _participant(context, set_path, force)
            outputs.extend(participant_paths)
            context.progress(stage, index, len(paths), participant)
    else:
        with ThreadPoolExecutor(max_workers=context.workers) as executor:
            futures = {executor.submit(_participant, context, path, force): path for path in paths}
            for index, future in enumerate(as_completed(futures), 1):
                participant, participant_paths = future.result()
                outputs.extend(participant_paths)
                context.progress(stage, index, len(paths), participant)
    summary = {
        "participants": len(paths), "durations_seconds": list(context.config.durations_seconds),
        "k_values": list(context.config.k_values), "feature_files": len(outputs),
        "short_duration_frequency_grid": "Hann window zero-padded to the fixed 4500-sample grid",
        "neighbor_calibration": "18 fixed neighboring bins per harmonic",
    }
    report_path = context.run_dir / "duration_features.json"
    atomic_write_json(report_path, summary)
    outputs.append(report_path)
    context.finish_stage(stage, outputs, summary)
    return outputs, summary
