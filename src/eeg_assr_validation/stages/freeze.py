from __future__ import annotations

import subprocess

from ..runtime import RunContext, atomic_write_json


def _git_value(root, *arguments: str) -> str:
    result = subprocess.run(["git", "-C", str(root), *arguments], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def run(context: RunContext, force: bool = False) -> tuple[list, dict]:
    del force
    stage, config = "00_freeze", context.config
    context.start_stage(stage, 1)
    design = {
        "analysis_id": context.analysis_id,
        "status": "prospectively_locked_external_positive_estimator_validation",
        "dataset": {"id": config.dataset_id, "version": config.dataset_version,
                    "git_commit": _git_value(config.dataset_root, "rev-parse", "HEAD")},
        "cohort": {"group": config.group, "minimum_participants": config.minimum_participants,
                   "known_missing_assr": ["sub-10003", "sub-10124", "sub-10508"],
                   "known_incomplete": ["sub-10593"]},
        "trial_rules": {"40_hz_standards": 170, "27_hz_standards": 160,
                        "trials_per_block": config.trials_per_block, "chronological": True,
                        "excluded_event_types": ["oddballs", "responses", "block markers", "surplus standards"]},
        "preprocessing": {"continuous_filter_hz": [config.l_freq, config.h_freq],
                          "notch_hz": config.notch_hz, "average_reference": config.average_reference,
                          "retained_interval_seconds": [config.epoch_start_seconds, config.epoch_end_seconds],
                          "primary_artifact_rejection": False},
        "estimators": {"methods": ["hann", "rectangular_ls", "hilbert"],
                       "hilbert_order": config.hilbert_order,
                       "hilbert_half_bandwidth_hz": config.hilbert_half_bandwidth_hz,
                       "channel_weighting": "equal_64_eeg"},
        "null": {"type": "trial-wise phase rotations shared across channels and estimators",
                 "surrogates_per_block": config.surrogates_per_block, "seed": config.seed},
        "primary_endpoint": "40-Hz Hilbert minus Hann K=10 detection rate",
        "primary_gate": {"ci95_low_strictly_positive": True,
                         "hilbert_heldout_fpr_max": config.fpr_acceptance_high,
                         "minimum_participants": config.minimum_participants},
        "interpretation_guardrail": "Estimator-level ASSR validation; not validation of syllable classification or a clinical detector.",
    }
    path = context.run_dir / "design_lock.json"
    atomic_write_json(path, design)
    context.progress(stage, 1, 1, "design locked")
    context.finish_stage(stage, [path], {"locked": True})
    return [path], design
