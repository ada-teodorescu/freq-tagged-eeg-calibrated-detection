from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from eeg_core.runtime import atomic_write_json
from eeg_replication.io import inspect_set, natural_subject_key

from ..runtime import RunContext


HILBERT_METHODS = ("H1_Hann4s_Regional4H", "H2_Hilbert4s_Regional4H", "H3_Hann4p5s_Regional4H")
TRCA_METHODS = ("S2_Equal4plus8", "S4_RegionalMonotone4plus8", "TRCA_S2_Grouped")
CONFIRMATORY_FAMILY = (
    {"candidate": "H2_Hilbert4s_Regional4H", "comparator": "H1_Hann4s_Regional4H",
     "unit": "participant_condition", "endpoint": "static_k10"},
    {"candidate": "TRCA_S2_Grouped", "comparator": "S2_Equal4plus8",
     "unit": "participant_stream", "endpoint": "sequential_population"},
    {"candidate": "TRCA_S2_Grouped", "comparator": "S4_RegionalMonotone4plus8",
     "unit": "participant_stream", "endpoint": "sequential_population"},
)
SECONDARY_COMPARISONS = (
    {"candidate": "H2_Hilbert4s_Regional4H", "comparator": "H3_Hann4p5s_Regional4H",
     "note": "duration-confounded (H3 has 0.5s more data)"},
    {"candidate": "TRCA_S2_Grouped", "comparator": "S4_RegionalMonotone4plus8_shrinkage",
     "note": "participant-adapted calibration sensitivity comparison; model information differs"},
)


def _fingerprint(path: Path) -> dict:
    stat = path.stat()
    digest = sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(1024 * 1024))
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "first_megabyte_sha256": digest.hexdigest()}


def run(context: RunContext, force: bool = False) -> tuple[list[Path], dict]:
    del force
    stage, config = "00_freeze", context.config
    context.start_stage(stage, 3)

    paths = sorted((config.data_root / "Pre-processed EEGLAB datasets").glob("*.set"), key=natural_subject_key)
    if len(paths) != config.participants_expected:
        raise ValueError(f"expected {config.participants_expected} SET files, found {len(paths)}")
    epochs = [inspect_set(path) for path in paths]
    participants = [epoch.participant for epoch in epochs]
    labels = epochs[0].channel_labels
    if any(epoch.channel_labels != labels for epoch in epochs):
        raise ValueError("channel ordering differs across participants")
    if set(channel for region in config.regions for channel in region) != set(labels):
        raise ValueError("configured regions do not match the public EEG channels")
    context.progress(stage, 1, 3, "cohort and channel topology")

    baseline = config.baseline_advanced_run / "advanced_conclusions.json"
    if not baseline.exists():
        raise FileNotFoundError(
            f"locked eeg_advanced_study baseline is missing: {baseline}. "
            "This study compares against its S2/S4/C4 numbers and must not proceed without it."
        )
    context.progress(stage, 2, 3, "advanced_study baseline present")

    lock = {
        "schema_version": 1,
        "analysis_id": context.analysis_id,
        "participants": participants,
        "channel_labels": list(labels),
        "hilbert_methods": list(HILBERT_METHODS),
        "trca_methods": list(TRCA_METHODS),
        "confirmatory_family": list(CONFIRMATORY_FAMILY),
        "secondary_comparisons": list(SECONDARY_COMPARISONS),
        "locked_c": config.locked_c,
        "matched_window_seconds": [config.matched_window_start_seconds, config.matched_window_end_seconds],
        "full_window_seconds": [config.full_window_start_seconds, config.full_window_end_seconds],
        "outcome_dependent_retuning_allowed": False,
        "manuscript_outputs_mutated": False,
        "existing_studies_mutated": False,
        "stop_and_report_gate": "If neither H2 nor TRCA_S2 beats its locked comparator with a Holm-adjusted "
                                 "p < .05 confirmatory CI, report both as null results and do not retune.",
        "baseline_advanced_run": str(config.baseline_advanced_run),
    }
    design_path = context.run_dir / "design_lock.json"
    atomic_write_json(design_path, lock)
    context.progress(stage, 3, 3, "design locked")

    manifest = {
        "analysis_id": context.analysis_id,
        "inputs": [_fingerprint(path) for pair in zip(paths, [epoch.fdt_path for epoch in epochs]) for path in pair],
        "baseline": _fingerprint(baseline),
    }
    manifest_path = context.run_dir / "input_fingerprints.json"
    atomic_write_json(manifest_path, manifest)

    summary = {"participants": len(participants)}
    context.finish_stage(stage, [design_path, manifest_path], summary)
    return [design_path, manifest_path], summary
