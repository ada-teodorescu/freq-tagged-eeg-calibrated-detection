from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

from ..dataset import read_channel_status
from ..runtime import RunContext


def eligibility(source_channels: tuple[str, ...], bad_channels: tuple[str, ...], target_channels: tuple[str, ...],
                structural: tuple[str, ...], maximum_bad_targets: int) -> tuple[bool, str, list[str]]:
    source = set(source_channels); structural_set = set(structural)
    undeclared_missing = sorted(set(target_channels) - source - structural_set)
    if undeclared_missing: return False, "undeclared missing target channels", undeclared_missing
    additional_bad = sorted(set(bad_channels) & set(target_channels) - structural_set)
    if len(additional_bad) > maximum_bad_targets:
        return False, "too many dataset-marked bad target channels", additional_bad
    return True, "eligible", additional_bad


def run(context: RunContext, force: bool = False) -> tuple[list[Path], dict]:
    del force
    stage, config = "02_preprocess", context.config
    inventory = pd.read_csv(context.table_dir / "recording_inventory.csv")
    context.start_stage(stage, len(inventory))
    rows = []
    for index, row in enumerate(inventory.itertuples(index=False), 1):
        channels, bad = read_channel_status(Path(row.channel_path))
        eligible, reason, bad_targets = eligibility(channels, bad, config.target_channels,
                                                    config.structural_interpolation_targets,
                                                    config.maximum_bad_target_channels)
        rows.append({"participant": row.participant, "task": row.task, "eligible": eligible, "reason": reason,
                     "source_sampling_rate": row.sampling_rate, "duration_seconds": row.duration_seconds,
                     "structural_interpolation_targets": json.dumps(config.structural_interpolation_targets),
                     "dataset_bad_target_channels": json.dumps(bad_targets),
                     "undeclared_missing_channels": json.dumps(bad_targets if not eligible and reason.startswith("undeclared") else [])})
        context.progress(stage, index, len(inventory), f"{row.participant} {row.task}")
    path = context.table_dir / "preprocessing_eligibility.csv"; pd.DataFrame(rows).to_csv(path, index=False)
    summary = {"eligible": int(sum(row["eligible"] for row in rows)), "excluded": int(sum(not row["eligible"] for row in rows)),
               "structural_interpolation_targets": list(config.structural_interpolation_targets)}
    context.finish_stage(stage, [path], summary); return [path], summary
