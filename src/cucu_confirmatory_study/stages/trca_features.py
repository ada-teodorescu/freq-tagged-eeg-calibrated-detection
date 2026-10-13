from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from cucu_accuracy_study.features import exact_harmonic_coefficients
from cucu_article.runtime import atomic_save_npz, atomic_write_json
from cucu_detection.spatial import apply_spatial
from cucu_replication.io import inspect_set, load_stream_trials, natural_subject_key

from ..runtime import RunContext
from ..trca import fit_trca


def _condition_streams(condition: int) -> range:
    return range(3 * condition + 1, 3 * condition + 4)


def trca_path(context: RunContext, participant: str) -> Path:
    return context.checkpoint_dir / "04_trca_features" / f"{participant}.npz"


def _participant(context: RunContext, set_path: Path, force: bool) -> tuple[str, Path]:
    config = context.config
    epoch = inspect_set(set_path)
    destination = trca_path(context, epoch.participant)
    if context.checkpoint_valid(destination) and not force:
        return epoch.participant, destination

    full_duration = config.full_window_end_seconds - config.full_window_start_seconds
    harmonics = tuple(config.harmonics_hz[:2])  # TRCA+S2 uses 4 and 8 Hz only, matching S2's formula
    raw_by_stream: dict[int, np.ndarray] = {}
    for stream in range(1, config.streams + 1):
        raw_by_stream[stream] = load_stream_trials(epoch, stream, dtype=np.float32)[..., config.onset_crop_samples:]

    # stream x trial x channel(=1, the TRCA-projected virtual channel) x harmonic --
    # this shape feeds directly into phase_trajectories' "stream x trial x channel x
    # harmonic" contract without reshaping downstream.
    projected = np.empty((config.streams, config.trials_per_stream, 1, len(harmonics)), dtype=np.complex64)
    filter_records = []
    for held_condition in range(config.conditions):
        training_conditions = [c for c in range(config.conditions) if c != held_condition]
        training_by_stream = [
            raw_by_stream[stream] for c in training_conditions for stream in _condition_streams(c)
        ]
        model = fit_trca(training_by_stream, config.trca_regularization[0])
        filter_records.append(model.weights)
        for stream in _condition_streams(held_condition):
            single_channel = apply_spatial(raw_by_stream[stream], model)[:, None, :]  # trial x 1 x time
            projected[stream - 1] = exact_harmonic_coefficients(
                single_channel, config.sampling_rate, harmonics, 0.0, full_duration,
            )

    atomic_save_npz(
        destination, trca_coefficients=projected, filter_weights=np.stack(filter_records),
        participant=np.asarray(epoch.participant), harmonics_hz=np.asarray(harmonics),
    )
    context.mark_checkpoint(destination, {
        "participant": epoch.participant, "unit": "leave_one_condition_out",
        "training_streams_per_fold": (config.conditions - 1) * config.streams_per_condition,
    })
    return epoch.participant, destination


def run(context: RunContext, force: bool = False) -> tuple[list[Path], dict]:
    stage, config = "04_trca_features", context.config
    paths = sorted((config.data_root / "Pre-processed EEGLAB datasets").glob("*.set"), key=natural_subject_key)
    context.start_stage(stage, len(paths))
    outputs: list[Path] = []
    for index, path in enumerate(paths, 1):
        participant, output = _participant(context, path, force)
        outputs.append(output); context.progress(stage, index, len(paths), participant)
    model_rows = []
    for output in outputs:
        with np.load(output, allow_pickle=False) as archive:
            participant = str(archive["participant"])
            weights = archive["filter_weights"]
        for held_condition, fold_weights in enumerate(weights, 1):
            training_conditions = [value for value in range(1, config.conditions + 1) if value != held_condition]
            training_streams = [
                stream for condition in training_conditions
                for stream in _condition_streams(condition - 1)
            ]
            model_rows.append({
                "participant": participant,
                "heldout_condition": held_condition,
                "training_conditions": json.dumps(training_conditions),
                "training_streams": json.dumps(training_streams),
                "training_stream_count": len(training_streams),
                "regularization": config.trca_regularization[0],
                "weights": json.dumps(fold_weights.tolist()),
            })
    model_path = context.table_dir / "trca_model_records.csv"
    pd.DataFrame(model_rows).to_csv(model_path, index=False)
    outputs.append(model_path)
    summary = {"participants": len(outputs), "unit": "leave_one_condition_out_per_participant",
               "training_streams_per_fold": (config.conditions - 1) * config.streams_per_condition,
               "regularization": config.trca_regularization[0]}
    report = context.run_dir / "trca_features.json"
    atomic_write_json(report, summary); outputs.append(report)
    context.finish_stage(stage, outputs, summary)
    return outputs, summary
