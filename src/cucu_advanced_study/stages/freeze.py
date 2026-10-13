from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import json

from cucu_article.runtime import atomic_write_json
from cucu_replication.io import inspect_set, natural_subject_key

from ..runtime import RunContext


STREAM_METHODS = (
    "S1_ITPC4",
    "S2_Equal4plus8",
    "S3_RidgeComplex4plus8",
    "S4_RegionalMonotone4plus8",
)
CONDITION_METHODS = (
    "C1_ConditionITPC4",
    "C2_ConditionEqual4plus8",
    "C3_RegionalMonotone4plus8",
    "C4_PreviousRegionalMonotone4H",
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
    context.start_stage(stage, 4)
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
    context.progress(stage, 1, 4, "cohort and channel topology")

    baseline_paths = (
        config.baseline_algorithm_run / "algorithm_conclusions.json",
        config.baseline_accuracy_run / "accuracy_conclusions.json",
    )
    if any(not path.exists() for path in baseline_paths):
        raise FileNotFoundError("locked baseline conclusions are missing")
    context.progress(stage, 2, 4, "prior immutable studies")

    external_files = sorted(config.external_negative_root.glob("*.npz")) if config.external_negative_root.exists() else []
    lock = {
        "schema_version": 1,
        "analysis_id": context.analysis_id,
        "participants": participants,
        "channel_labels": list(labels),
        "stream_methods": list(STREAM_METHODS),
        "condition_methods": list(CONDITION_METHODS),
        "primary_endpoint": {
            "unit": "stream", "method": "S2_Equal4plus8", "comparator": "S1_ITPC4",
            "duration_seconds": config.primary_duration_seconds, "natural_trial_order": True,
            "looks": list(config.k_values), "familywise_fpr": config.target_fpr,
        },
        "secondary_endpoints": ["S3 versus S2", "S4 versus S2", "condition-level regional fusion"],
        "null": {
            "primary": "one random temporal shift per trial; exp(i*h*phi), shared across channels and looks",
            "sensitivity": "independent same-frequency harmonic phase rotations, shared across channels",
            "surrogates": config.surrogates, "seeds": list(config.surrogate_seeds),
        },
        "validation": "outer participant LOPO; participant-grouped inner CV; training-only calibration",
        "shrinkage": "held participant's other 14 conditions; null-only pseudo-LOPO; evaluated condition excluded",
        "off_target_families_hz": [[base, 2 * base] for base in config.off_target_bases_hz],
        "off_target_label": "off-target controls (not genuine sham recordings)",
        "external_negative_status": "available" if external_files else "external negatives unavailable",
        "external_negative_files": [path.name for path in external_files],
        "outcome_dependent_retuning_allowed": False,
        "manuscript_outputs_mutated": False,
        "baseline_conclusions": [json.loads(path.read_text(encoding="utf-8")) for path in baseline_paths],
    }
    design_path = context.run_dir / "design_lock.json"
    atomic_write_json(design_path, lock)
    context.progress(stage, 3, 4, "design locked")
    manifest = {
        "analysis_id": context.analysis_id,
        "inputs": [_fingerprint(path) for pair in zip(paths, [epoch.fdt_path for epoch in epochs]) for path in pair],
        "baselines": [_fingerprint(path) for path in baseline_paths],
        "external_negatives": [_fingerprint(path) for path in external_files],
    }
    manifest_path = context.run_dir / "input_fingerprints.json"
    atomic_write_json(manifest_path, manifest)
    context.progress(stage, 4, 4, "input fingerprints")
    summary = {"participants": len(participants), "external_negative_files": len(external_files)}
    context.finish_stage(stage, [design_path, manifest_path], summary)
    return [design_path, manifest_path], summary
