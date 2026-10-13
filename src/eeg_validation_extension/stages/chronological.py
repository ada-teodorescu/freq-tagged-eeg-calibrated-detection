from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from eeg_core.runtime import atomic_write_json
from ..runtime import RunContext


def completion_event(condition_events: pd.DataFrame, k: int) -> pd.Series:
    candidates = []
    for _, stream_events in condition_events.groupby("stream_id", sort=True):
        ordered = stream_events.sort_values("trial_index")
        if len(ordered) < k: raise ValueError(f"stream has fewer than {k} events")
        candidates.append(ordered.iloc[k - 1])
    if len(candidates) != 3: raise ValueError("condition does not contain exactly three streams")
    return max(candidates, key=lambda row: int(row["trial_index"]))


def run(context: RunContext, force: bool = False) -> tuple[list[Path], dict]:
    del force
    stage, config = "04_chronological", context.config
    events = pd.read_csv(config.raw_events); scores = pd.read_csv(context.table_dir / "sequential_scores.csv")
    groups = list(scores.groupby(["participant", "condition", "method", "detector"], sort=True))
    context.start_stage(stage, len(groups)); rows = []
    for index, ((participant, condition, method, detector), score_rows) in enumerate(groups, 1):
        score = score_rows.iloc[0]
        participant_events = events[events["participant"] == participant].sort_values("trial_index")
        condition_events = participant_events[participant_events["condition_id"] == condition]
        first_k = int(score["first_detection_k"]); detected = first_k <= config.trials_per_stream
        if detected:
            event = completion_event(condition_events, first_k)
            presentation = int(event["trial_index"]); absolute_time = float(event["time_seconds"])
            condition_presentations = 3 * first_k
        else:
            event = participant_events.iloc[-1]; presentation = int(event["trial_index"])
            absolute_time = float(event["time_seconds"]); condition_presentations = len(condition_events)
        rows.append({"participant": participant, "condition": int(condition), "method": method, "detector": detector,
                     "detected": detected, "first_detection_k": first_k,
                     "condition_presentations": int(condition_presentations),
                     "global_presentation_index": presentation, "absolute_recording_time_seconds": absolute_time,
                     "elapsed_from_first_target_seconds": absolute_time - float(participant_events.iloc[0]["time_seconds"]),
                     "censored_at_session_end": not detected})
        context.progress(stage, index, len(groups), f"{participant} C{condition} {method} {detector}")
    path = context.table_dir / "chronological_stopping.csv"; pd.DataFrame(rows).to_csv(path, index=False)
    summary = {"rows": len(rows), "censor_presentation": 450,
               "rule": "the kth update occurs when the kth trial of all three condition streams has appeared"}
    report = context.run_dir / "chronological_summary.json"; atomic_write_json(report, summary)
    outputs = [path, report]; context.finish_stage(stage, outputs, summary); return outputs, summary
