from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from eeg_core.runtime import atomic_write_json
from ..runtime import RunContext


def _fingerprint(path: Path) -> dict:
    stat = path.stat(); digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return {"path": str(path), "bytes": stat.st_size, "sha256": digest.hexdigest()}


def run(context: RunContext, force: bool = False) -> tuple[list[Path], dict]:
    del force
    stage, config = "00_freeze", context.config
    context.start_stage(stage, 3)
    required = {
        "conclusions": config.baseline_confirmatory_run / "confirmatory_conclusions.json",
        "design": config.baseline_confirmatory_run / "design_lock.json",
        "static_features": config.baseline_confirmatory_run / "checkpoints" / "02_hilbert_condition" / "condition_features.npz",
        "sequential_features": config.baseline_confirmatory_run / "checkpoints" / "03_hilbert_sequential" / "trajectories.npz",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing: raise FileNotFoundError("corrected confirmatory baseline incomplete: " + ", ".join(missing))
    context.progress(stage, 1, 3, "corrected confirmatory baseline")
    lock = {
        "schema_version": 1, "analysis_id": context.analysis_id,
        "dataset": {"id": config.dataset_id, "version": config.dataset_version,
                    "primary_task": config.primary_task, "secondary_tasks": list(config.secondary_tasks)},
        "baseline_confirmatory_run": str(config.baseline_confirmatory_run),
        "primary_unit": "participant_condition_from_three_pseudostreams",
        "target_channels": list(config.target_channels), "regions": [list(value) for value in config.regions],
        "structural_interpolation_targets": list(config.structural_interpolation_targets),
        "structural_interpolation_reason": "ds007615 BioSemi-64 lacks TP9, TP10, PO9, and PO10",
        "preprocessing": {"l_freq": config.l_freq, "h_freq": config.h_freq,
                          "sampling_rate": config.sampling_rate, "average_reference": config.average_reference,
                          "maximum_additional_bad_target_channels": config.maximum_bad_target_channels},
        "model": {"method": "H2_Hilbert4s_Regional4H", "locked_c": config.locked_c,
                  "harmonics_hz": list(config.harmonics_hz), "k_values": list(config.k_values)},
        "submission_gate": {"endpoints": ["static", "sequential"], "task": config.primary_task,
                            "one_sided_confidence": config.one_sided_confidence,
                            "upper_fpr_limit": config.external_fpr_upper_limit,
                            "both_must_pass": True},
        "external_data_used_for_training_or_calibration": False,
        "outcome_dependent_retuning_allowed": False,
    }
    design_path = context.run_dir / "design_lock.json"; atomic_write_json(design_path, lock)
    context.progress(stage, 2, 3, "external design locked")
    fingerprints = {name: _fingerprint(path) for name, path in required.items()}
    fingerprint_path = context.run_dir / "baseline_fingerprints.json"; atomic_write_json(fingerprint_path, fingerprints)
    context.progress(stage, 3, 3, "baseline fingerprinted")
    outputs = [design_path, fingerprint_path]
    context.finish_stage(stage, outputs, {"baseline": str(config.baseline_confirmatory_run)})
    return outputs, lock
