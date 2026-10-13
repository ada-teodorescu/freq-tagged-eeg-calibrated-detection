from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

from eeg_core.stages.benchmark import auc_fast, empirical_threshold_fast
from ..runtime import RunContext, atomic_write_json
from .features import feature_path


def _load_onset(context: RunContext, participants: list[str]) -> dict[str, np.ndarray]:
    names = (
        "onset_positive_global", "onset_null_global",
        "onset_positive_weighted_global", "onset_null_weighted_global",
    )
    values: dict[str, list[np.ndarray]] = {name: [] for name in names}
    starts = None
    for participant in participants:
        with np.load(feature_path(context, participant), allow_pickle=False) as archive:
            for name in names:
                values[name].append(archive[name])
            current = archive["onset_starts_seconds"]
            if starts is None:
                starts = current
            elif not np.array_equal(starts, current):
                raise AssertionError("onset windows differ across participants")
    result = {name: np.stack(items) for name, items in values.items()}
    result["starts"] = np.asarray(starts)
    return result


def run(context: RunContext, force: bool = False) -> tuple[list[Path], dict]:
    del force
    stage = "03_robustness"
    config = context.config
    lock = json.loads((context.run_dir / "design_lock.json").read_text(encoding="utf-8"))
    participants = list(lock["participants"])
    participant_numbers = np.arange(len(participants))
    onset = _load_onset(context, participants)
    context.start_stage(stage, len(config.onset_starts_seconds) + 1)
    scale = 2 * config.trials_per_stream
    rows = []
    for window_index, start in enumerate(onset["starts"]):
        fixed = {
            "C1_ConditionITPC4": (
                scale * onset["onset_positive_global"][:, window_index, :, 0],
                scale * onset["onset_null_global"][:, window_index, :, :, 0],
            ),
            "C2_Condition4plus8": (
                scale * onset["onset_positive_global"][:, window_index, :, :2].sum(axis=-1),
                scale * onset["onset_null_global"][:, window_index, :, :, :2].sum(axis=-1),
            ),
            "C3_Reliability4plus8": (
                scale * onset["onset_positive_weighted_global"][:, window_index, :, :2].sum(axis=-1),
                scale * onset["onset_null_weighted_global"][:, window_index, :, :, :2].sum(axis=-1),
            ),
        }
        for heldout, participant in enumerate(participants):
            training = participant_numbers != heldout
            for algorithm, (positive, null) in fixed.items():
                threshold = empirical_threshold_fast(null[training], config.target_fpr)
                for condition in range(config.conditions):
                    rows.append({
                        "participant": participant, "participant_index": heldout, "condition": condition,
                        "algorithm": algorithm, "window_start_seconds_after_crop": float(start),
                        "window_start_seconds_after_stimulus": float(start + config.onset_crop_samples / config.sampling_rate),
                        "window_seconds": config.onset_window_seconds,
                        "detected": int(positive[heldout, condition] > threshold),
                        "heldout_fpr": float(np.mean(null[heldout, condition] > threshold)),
                        "auc": auc_fast(positive[heldout, condition:condition + 1], null[heldout, condition]),
                    })
        context.progress(stage, window_index + 1, len(config.onset_starts_seconds) + 1, f"start={start:g}s")
    onset_path = context.table_dir / "sliding_window_scores.csv"
    pd.DataFrame(rows).to_csv(onset_path, index=False)

    primary = pd.read_csv(context.table_dir / "condition_scores.csv")
    seed_rows = primary[(primary.target_fpr == config.target_fpr) & (primary.surrogate_block != "pooled")]
    seed_summary = seed_rows.groupby(["algorithm", "surrogate_block"], as_index=False).agg(
        sensitivity=("detected", "mean"), heldout_fpr=("heldout_fpr", "mean"), auc=("auc", "mean"),
    )
    seed_path = context.table_dir / "surrogate_seed_stability.csv"
    seed_summary.to_csv(seed_path, index=False)
    summary = {
        "equal_length_windows": len(config.onset_starts_seconds),
        "window_seconds": config.onset_window_seconds,
        "surrogate_seed_blocks": len(config.surrogate_seeds),
        "seed_sensitivity_ranges": {
            algorithm: [float(group.sensitivity.min()), float(group.sensitivity.max())]
            for algorithm, group in seed_summary.groupby("algorithm")
        },
    }
    report = context.run_dir / "robustness.json"
    atomic_write_json(report, summary)
    context.progress(stage, len(config.onset_starts_seconds) + 1,
                     len(config.onset_starts_seconds) + 1, "surrogate stability")
    context.finish_stage(stage, [onset_path, seed_path, report], summary)
    return [onset_path, seed_path, report], summary
