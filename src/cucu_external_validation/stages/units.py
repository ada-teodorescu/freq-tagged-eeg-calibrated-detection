from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..dataset import greedy_nonoverlapping, group_condition_units, rest_windows
from ..runtime import RunContext


def run(context: RunContext, force: bool = False) -> tuple[list[Path], dict]:
    del force
    stage, config = "03_negative_units", context.config
    inventory = pd.read_csv(context.table_dir / "recording_inventory.csv")
    eligible = pd.read_csv(context.table_dir / "preprocessing_eligibility.csv")
    merged = inventory.merge(eligible[["participant", "task", "eligible", "reason"]], on=["participant", "task"])
    context.start_stage(stage, len(merged))
    rows, exclusions = [], []
    for index, record in enumerate(merged.itertuples(index=False), 1):
        if not bool(record.eligible):
            exclusions.append({"participant": record.participant, "task": record.task, "reason": record.reason})
            context.progress(stage, index, len(merged), f"{record.participant} excluded"); continue
        try:
            if record.task == config.primary_task:
                events = pd.read_csv(Path(record.event_path), sep="\t")
                starts = greedy_nonoverlapping(events["onset"].to_numpy(float), config.epoch_seconds,
                                               float(record.duration_seconds), config.auditory_windows_per_participant)
            else:
                starts = rest_windows(float(record.duration_seconds), config.epoch_seconds,
                                      config.rest_windows_per_participant)
            grouped = group_condition_units(starts, config.streams_per_unit, config.trials_per_stream)
            for unit in range(grouped.shape[0]):
                for stream in range(grouped.shape[1]):
                    for trial in range(grouped.shape[2]):
                        start = float(grouped[unit, stream, trial])
                        rows.append({"participant": record.participant, "task": record.task, "unit": unit + 1,
                                     "stream": stream + 1, "trial": trial + 1, "start_seconds": start,
                                     "end_seconds": start + config.epoch_seconds, "vhdr_path": record.vhdr_path})
        except (OSError, ValueError, KeyError) as error:
            exclusions.append({"participant": record.participant, "task": record.task, "reason": str(error)})
        context.progress(stage, index, len(merged), f"{record.participant} {record.task}")
    membership = pd.DataFrame(rows)
    membership_path = context.table_dir / "negative_unit_membership.csv"; membership.to_csv(membership_path, index=False)
    exclusion_path = context.table_dir / "external_exclusions.csv"; pd.DataFrame(exclusions, columns=("participant", "task", "reason")).to_csv(exclusion_path, index=False)
    summary = {"windows": len(membership), "units": int(membership[["participant", "task", "unit"]].drop_duplicates().shape[0]),
               "exclusions": len(exclusions)}
    outputs = [membership_path, exclusion_path]; context.finish_stage(stage, outputs, summary); return outputs, summary
