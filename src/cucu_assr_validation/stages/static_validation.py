from __future__ import annotations

import numpy as np
import pandas as pd

from ..algorithms import METHODS, heldout_static
from ..runtime import RunContext
from .trajectories import trajectory_path


def load_trajectories(context: RunContext, participants: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    positive, null = [], []
    for participant in participants:
        with np.load(trajectory_path(context, participant), allow_pickle=False) as archive:
            positive.append(archive["positive"]); null.append(archive["null"])
    return np.stack(positive), np.stack(null)


def run(context: RunContext, force: bool = False) -> tuple[list, dict]:
    del force
    stage, config = "04_static_validation", context.config
    participants = tuple(pd.read_csv(context.table_dir / "recording_inventory.csv")["participant"].astype(str))
    positive, null = load_trajectories(context, participants)
    context.start_stage(stage, len(METHODS) * len(config.frequencies_hz))
    rows, completed = [], 0
    for m_index, method in enumerate(METHODS):
        for f_index, frequency in enumerate(config.frequencies_hz):
            blocks = config.standard_limit(frequency) // config.trials_per_block
            local = heldout_static(positive[:, m_index, f_index, :blocks, -1],
                                   null[:, m_index, f_index, :blocks, :, -1], participants, config.target_fpr)
            for row in local:
                row.update(method=method, frequency_hz=frequency, k=config.k_values[-1])
            rows += local; completed += 1
            context.progress(stage, completed, len(METHODS) * len(config.frequencies_hz), f"{method} {frequency:g} Hz")
    frame = pd.DataFrame(rows)
    scores_path = context.table_dir / "static_scores.csv"
    frame.to_csv(scores_path, index=False)
    metrics = (frame.groupby(["participant", "method", "frequency_hz"], as_index=False)
               .agg(detection=("detected", "mean"), heldout_fpr=("heldout_fpr", "first"),
                    threshold=("threshold", "first"), blocks=("block", "size")))
    metrics_path = context.table_dir / "static_participant_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    thresholds_path = context.table_dir / "heldout_thresholds.csv"
    metrics[["participant", "method", "frequency_hz", "threshold", "heldout_fpr"]].to_csv(thresholds_path, index=False)
    outputs = [scores_path, metrics_path, thresholds_path]
    summary = {"participants": len(participants), "rows": len(frame), "target_fpr": config.target_fpr}
    context.finish_stage(stage, outputs, summary)
    return outputs, summary
