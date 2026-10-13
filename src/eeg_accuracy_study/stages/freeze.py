from __future__ import annotations

from pathlib import Path
import json

from eeg_replication.io import inspect_set, natural_subject_key
from ..runtime import RunContext, atomic_write_json


METHODS = (
    "C1_ConditionITPC4",
    "C2_Condition4plus8",
    "C3_Reliability4plus8",
    "C4_MonotoneHarmonic",
    "C5_RegionalMonotone",
)


def run(context: RunContext, force: bool = False) -> tuple[list[Path], dict]:
    del force
    stage = "00_freeze"
    config = context.config
    context.start_stage(stage, 3)
    set_paths = sorted((config.data_root / "Pre-processed EEGLAB datasets").glob("*.set"), key=natural_subject_key)
    if len(set_paths) != config.participants_expected:
        raise ValueError(f"expected {config.participants_expected} SET files, found {len(set_paths)}")
    epochs = [inspect_set(path) for path in set_paths]
    participants = [epoch.participant for epoch in epochs]
    labels = epochs[0].channel_labels
    if any(epoch.channel_labels != labels for epoch in epochs):
        raise ValueError("channel ordering differs across participants")
    context.progress(stage, 1, 3, "public cohort and channel order")
    configured = [channel for region in config.regions for channel in region]
    if set(configured) != set(labels):
        raise ValueError("configured spatial regions do not match public EEG channels")
    baseline = config.baseline_algorithm_run / "algorithm_conclusions.json"
    if not baseline.exists():
        raise FileNotFoundError(f"baseline repaired study missing: {baseline}")
    context.progress(stage, 2, 3, "baseline and regions")
    lock = {
        "participants": participants,
        "channel_labels": list(labels),
        "methods": list(METHODS),
        "primary_unit": "participant_condition_from_three_streams",
        "primary_endpoint": {
            "trials_per_stream": config.trials_per_stream,
            "streams_per_condition": config.streams_per_condition,
            "duration_seconds": config.primary_duration_seconds,
            "target_sensitivity": config.target_sensitivity,
            "maximum_fpr": config.target_fpr,
        },
        "validation": "outer leave-one-participant-out; grouped inner CV; training-only thresholds",
        "null": "same-frequency independent trial-phase rotations shared across channels",
        "surrogate_seeds": list(config.surrogate_seeds),
        "surrogates_per_seed": config.surrogates_per_seed,
        "outcome_dependent_tuning_allowed": False,
        "manuscript_outputs": False,
        "baseline_algorithm_conclusions": json.loads(baseline.read_text(encoding="utf-8")),
    }
    path = context.run_dir / "design_lock.json"
    atomic_write_json(path, lock)
    context.progress(stage, 3, 3, "design locked")
    context.finish_stage(stage, [path], lock)
    return [path], lock
