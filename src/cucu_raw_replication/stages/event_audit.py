from __future__ import annotations

import pandas as pd

from ..events import discover_headers, target_events, validate_events
from ..runtime import RunContext, atomic_write_json


def run(context: RunContext, force: bool = False) -> tuple[list, dict]:
    stage = "01_events"
    config = context.config
    headers = discover_headers(config.paths.raw_root)
    context.start_stage(stage, len(context.participants))
    rows = []
    summaries = []
    for completed, participant in enumerate(context.participants, start=1):
        checkpoint = context.checkpoint_dir / stage / f"{participant}.json"
        events = target_events(headers[participant].vmrk_path)
        counts = validate_events(events, config.streams, config.trials_per_stream)
        for trial_index, (sample, _, stream) in enumerate(events, start=1):
            rows.append({
                "participant": participant, "trial_index": trial_index,
                "sample_zero_based": int(sample), "time_seconds": sample / config.sampling_rate,
                "stream_id": int(stream), "condition_id": (int(stream) - 1) // 3 + 1,
            })
        summary = {
            "participant": participant, "target_events": len(events),
            "first_sample": int(events[0, 0]), "last_sample": int(events[-1, 0]),
            "counts": {str(key): value for key, value in counts.items()},
        }
        atomic_write_json(checkpoint, summary)
        context.mark_checkpoint(checkpoint, {"participant": participant})
        summaries.append(summary)
        context.progress(stage, completed, len(context.participants), participant)
    events_path = context.table_dir / "events.csv"
    report_path = context.run_dir / "events.json"
    pd.DataFrame(rows).to_csv(events_path, index=False)
    report = {"passed": True, "participant_summaries": summaries}
    atomic_write_json(report_path, report)
    outputs = [events_path, report_path]
    context.finish_stage(stage, outputs, {"passed": True, "target_events": len(rows)})
    return outputs, report
