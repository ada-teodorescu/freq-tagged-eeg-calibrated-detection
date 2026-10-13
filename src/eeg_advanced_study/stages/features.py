from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json

import numpy as np

from eeg_accuracy_study.features import exact_harmonic_coefficients
from eeg_core.runtime import atomic_save_npz, atomic_write_json
from eeg_replication.io import inspect_set, load_stream_trials, natural_subject_key

from ..runtime import RunContext


def feature_path(context: RunContext, participant: str) -> Path:
    return context.checkpoint_dir / "01_coefficients" / f"{participant}.npz"


def _participant(context: RunContext, set_path: Path, force: bool) -> tuple[str, Path]:
    config = context.config
    epoch = inspect_set(set_path)
    destination = feature_path(context, epoch.participant)
    if context.checkpoint_valid(destination) and not force:
        return epoch.participant, destination
    target = np.empty(
        (len(config.durations_seconds), config.streams, config.trials_per_stream,
         config.channels, len(config.harmonics_hz)), dtype=np.complex64,
    )
    off_target = np.empty(
        (len(config.off_target_bases_hz), config.streams, config.trials_per_stream,
         config.channels, 2), dtype=np.complex64,
    )
    required = int(round(config.primary_duration_seconds * config.sampling_rate))
    for stream in range(1, config.streams + 1):
        trials = load_stream_trials(epoch, stream, dtype=np.float32)[..., config.onset_crop_samples:]
        if trials.shape[-1] < required:
            raise ValueError(f"{epoch.participant} stream {stream}: insufficient retained samples")
        for duration_index, duration in enumerate(config.durations_seconds):
            target[duration_index, stream - 1] = exact_harmonic_coefficients(
                trials, config.sampling_rate, config.harmonics_hz, 0.0, duration,
            )
        for family_index, base in enumerate(config.off_target_bases_hz):
            off_target[family_index, stream - 1] = exact_harmonic_coefficients(
                trials, config.sampling_rate, (base, 2 * base), 0.0, config.primary_duration_seconds,
            )
    atomic_save_npz(
        destination, target=target, off_target=off_target,
        participant=np.asarray(epoch.participant), channel_labels=np.asarray(epoch.channel_labels),
        durations_seconds=np.asarray(config.durations_seconds), harmonics_hz=np.asarray(config.harmonics_hz),
        off_target_bases_hz=np.asarray(config.off_target_bases_hz),
    )
    context.mark_checkpoint(destination, {"participant": epoch.participant, "natural_trial_order": True})
    return epoch.participant, destination


def run(context: RunContext, force: bool = False) -> tuple[list[Path], dict]:
    stage, config = "01_features", context.config
    lock = json.loads((context.run_dir / "design_lock.json").read_text(encoding="utf-8"))
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
    if sorted(path.stem for path in outputs) != sorted(lock["participants"]):
        raise AssertionError("feature participants differ from locked cohort")
    summary = {"participants": len(outputs), "durations": len(config.durations_seconds),
               "off_target_families": len(config.off_target_bases_hz), "natural_order_preserved": True}
    report = context.run_dir / "coefficient_features.json"
    atomic_write_json(report, summary); outputs.append(report)
    context.finish_stage(stage, outputs, summary)
    return outputs, summary
