from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import numpy as np

from eeg_replication.io import inspect_set, load_stream_trials, natural_subject_key
from ..features import condition_feature_bundle, exact_harmonic_coefficients, region_indices
from ..runtime import RunContext, atomic_save_npz, atomic_write_json


def feature_path(context: RunContext, participant: str) -> Path:
    return context.checkpoint_dir / "01_condition_features" / f"{participant}.npz"


def _participant(context: RunContext, set_path: Path, participant_index: int, force: bool) -> tuple[str, Path]:
    config = context.config
    epoch = inspect_set(set_path)
    destination = feature_path(context, epoch.participant)
    if context.checkpoint_valid(destination) and not force:
        return epoch.participant, destination
    regions = region_indices(epoch.channel_labels, config.regions)
    full = np.empty((config.streams, config.trials_per_stream, config.channels, len(config.harmonics_hz)), np.complex64)
    sliding = np.empty((len(config.onset_starts_seconds), *full.shape), np.complex64)
    retained_samples = int(round(config.primary_duration_seconds * config.sampling_rate))
    for stream in range(1, config.streams + 1):
        trials = load_stream_trials(epoch, stream, dtype=np.float32)[..., config.onset_crop_samples:]
        if trials.shape[-1] != retained_samples:
            raise ValueError(f"{epoch.participant} stream {stream}: retained samples={trials.shape[-1]}")
        full[stream - 1] = exact_harmonic_coefficients(
            trials, config.sampling_rate, config.harmonics_hz, 0.0, config.primary_duration_seconds,
        )
        for window_index, start in enumerate(config.onset_starts_seconds):
            sliding[window_index, stream - 1] = exact_harmonic_coefficients(
                trials, config.sampling_rate, config.harmonics_hz, start, config.onset_window_seconds,
            )
    quantiles = (config.reliability_lower_quantile, config.reliability_upper_quantile)
    primary = condition_feature_bundle(
        full, regions, config.surrogate_seeds, config.surrogates_per_seed,
        config.reliability_harmonic_indices, quantiles, seed_salt=participant_index,
    )
    onset_positive = []
    onset_null = []
    onset_positive_weighted = []
    onset_null_weighted = []
    for window_index in range(len(config.onset_starts_seconds)):
        bundle = condition_feature_bundle(
            sliding[window_index], regions, (config.seed + 9000 + window_index,), config.onset_surrogates,
            config.reliability_harmonic_indices, quantiles, seed_salt=participant_index,
        )
        onset_positive.append(bundle["positive_global"])
        onset_null.append(bundle["null_global"])
        onset_positive_weighted.append(bundle["positive_weighted_global"])
        onset_null_weighted.append(bundle["null_weighted_global"])
    atomic_save_npz(
        destination,
        **primary,
        onset_positive_global=np.asarray(onset_positive, dtype=np.float32),
        onset_null_global=np.asarray(onset_null, dtype=np.float32),
        onset_positive_weighted_global=np.asarray(onset_positive_weighted, dtype=np.float32),
        onset_null_weighted_global=np.asarray(onset_null_weighted, dtype=np.float32),
        onset_starts_seconds=np.asarray(config.onset_starts_seconds),
        channel_labels=np.asarray(epoch.channel_labels), region_names=np.asarray(config.region_names),
        surrogate_seeds=np.asarray(config.surrogate_seeds),
        surrogates_per_seed=np.asarray(config.surrogates_per_seed),
    )
    context.mark_checkpoint(destination, {
        "participant": epoch.participant, "primary_unit": "condition_from_three_streams",
        "primary_surrogates": config.primary_surrogates, "seed_salt": participant_index,
    })
    return epoch.participant, destination


def run(context: RunContext, force: bool = False) -> tuple[list[Path], dict]:
    stage = "01_condition_features"
    config = context.config
    lock = json.loads((context.run_dir / "design_lock.json").read_text(encoding="utf-8"))
    paths = sorted((config.data_root / "Pre-processed EEGLAB datasets").glob("*.set"), key=natural_subject_key)
    context.start_stage(stage, len(paths))
    outputs: list[Path] = []
    if context.workers == 1:
        for index, path in enumerate(paths):
            participant, output = _participant(context, path, index, force)
            outputs.append(output)
            context.progress(stage, index + 1, len(paths), participant)
    else:
        with ThreadPoolExecutor(max_workers=context.workers) as executor:
            futures = {executor.submit(_participant, context, path, index, force): path
                       for index, path in enumerate(paths)}
            for completed, future in enumerate(as_completed(futures), 1):
                participant, output = future.result()
                outputs.append(output)
                context.progress(stage, completed, len(paths), participant)
    if sorted(path.stem for path in outputs) != sorted(lock["participants"]):
        raise AssertionError("feature participants differ from the locked cohort")
    summary = {
        "participants": len(outputs), "conditions_per_participant": config.conditions,
        "primary_surrogates_per_condition": config.primary_surrogates,
        "regional_features": len(config.regions) * len(config.harmonics_hz),
        "onset_windows": len(config.onset_starts_seconds),
    }
    report = context.run_dir / "condition_features.json"
    atomic_write_json(report, summary)
    outputs.append(report)
    context.finish_stage(stage, outputs, summary)
    return outputs, summary
