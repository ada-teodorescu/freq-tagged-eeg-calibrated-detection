from __future__ import annotations

import numpy as np
import pandas as pd

from ..reference import discover_references, reference_as_trials
from ..runtime import RunContext, atomic_write_json
from ..statistics import agreement_metrics
from .common import epoch_values, load_branch_epochs


def _stream_aligned(values: np.ndarray, stream_ids: np.ndarray) -> np.ndarray:
    indices = np.concatenate([np.flatnonzero(stream_ids == stream) for stream in range(1, 46)])
    return values[indices]


def run(context: RunContext, force: bool = False) -> tuple[list, dict]:
    stage = "03_epoch_comparison"
    references = discover_references(context.config.paths.preprocessed_root)
    total = len(context.participants) * len(context.branches)
    context.start_stage(stage, total)
    rows = []
    completed = 0
    for participant in context.participants:
        reference = references[participant]
        reference_values = _stream_aligned(reference_as_trials(reference), reference.stream_ids)
        for branch in context.branches:
            epochs = load_branch_epochs(context, participant, branch)
            values, stream_ids = epoch_values(epochs)
            candidate = _stream_aligned(values * 1e6, stream_ids)  # MNE volts -> EEGLAB microvolts
            if [name.casefold() for name in epochs.ch_names] != [name.casefold() for name in reference.channels]:
                raise ValueError(f"{participant}: channel order differs during epoch comparison")
            metrics = agreement_metrics(candidate, reference_values)
            rows.append({"participant": participant, "branch": branch, **metrics})
            completed += 1
            context.progress(stage, completed, total, f"{participant} {branch}")
            del epochs, values, candidate
    table_path = context.table_dir / "epoch_agreement.csv"
    report_path = context.run_dir / "epoch_comparison.json"
    pd.DataFrame(rows).to_csv(table_path, index=False)
    report = {
        "same_id_is_primary": True,
        "unit_conversion": "MNE volts multiplied by 1e6 for EEGLAB microvolt comparison",
        "rows": rows,
    }
    atomic_write_json(report_path, report)
    outputs = [table_path, report_path]
    context.finish_stage(stage, outputs, {"comparisons": len(rows)})
    return outputs, report
