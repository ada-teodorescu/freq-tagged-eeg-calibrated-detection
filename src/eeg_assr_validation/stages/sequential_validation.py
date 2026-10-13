from __future__ import annotations

import pandas as pd

from ..algorithms import METHODS, heldout_sequential
from ..runtime import RunContext
from .static_validation import load_trajectories


def run(context: RunContext, force: bool = False) -> tuple[list, dict]:
    del force
    stage, config = "05_sequential_validation", context.config
    participants = tuple(pd.read_csv(context.table_dir / "recording_inventory.csv")["participant"].astype(str))
    positive, null = load_trajectories(context, participants)
    total = len(METHODS) * len(config.frequencies_hz)
    context.start_stage(stage, total)
    score_rows, metric_rows, completed = [], [], 0
    for m_index, method in enumerate(METHODS):
        for f_index, frequency in enumerate(config.frequencies_hz):
            blocks = config.standard_limit(frequency) // config.trials_per_block
            rows, metrics = heldout_sequential(positive[:, m_index, f_index, :blocks],
                                               null[:, m_index, f_index, :blocks], participants,
                                               config.k_values, config.target_fpr)
            for row in rows:
                row.update(method=method, frequency_hz=frequency)
            for row in metrics:
                row.update(method=method, frequency_hz=frequency)
            score_rows += rows; metric_rows += metrics; completed += 1
            context.progress(stage, completed, total, f"{method} {frequency:g} Hz")
    score_path = context.table_dir / "sequential_scores.csv"
    metrics_path = context.table_dir / "sequential_participant_metrics.csv"
    pd.DataFrame(score_rows).to_csv(score_path, index=False)
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    outputs = [score_path, metrics_path]
    summary = {"participants": len(participants), "looks": list(config.k_values), "target_familywise_fpr": config.target_fpr}
    context.finish_stage(stage, outputs, summary)
    return outputs, summary
