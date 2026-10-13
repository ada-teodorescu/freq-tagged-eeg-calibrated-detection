from __future__ import annotations

import numpy as np
import pandas as pd

from ..runtime import RunContext, atomic_save_npz, atomic_write_json
from ..spectral import analyse_epochs
from .common import epoch_values, load_branch_epochs


def run(context: RunContext, force: bool = False) -> tuple[list, dict]:
    stage = "04_spectral"
    config = context.config
    total = len(context.participants) * len(context.branches)
    context.start_stage(stage, total)
    rows = []
    outputs = []
    mean_all = np.empty((len(context.branches), len(context.participants), config.streams, len(config.harmonics_hz)), dtype=np.float32)
    channel_all = np.empty((len(context.branches), len(context.participants), config.streams, config.channels, len(config.harmonics_hz)), dtype=np.float32)
    completed = 0
    for branch_index, branch in enumerate(context.branches):
        for participant_index, participant in enumerate(context.participants):
            checkpoint = context.checkpoint_dir / stage / branch / f"{participant}.npz"
            if not force and context.checkpoint_valid(checkpoint):
                with np.load(checkpoint, allow_pickle=False) as archive:
                    channel_itpc = archive["channel_itpc"]
                    mean_itpc = archive["mean_itpc"]
            else:
                epochs = load_branch_epochs(context, participant, branch)
                values, stream_ids = epoch_values(epochs)
                result = analyse_epochs(
                    values, stream_ids, config.sampling_rate, config.harmonics_hz,
                    int(round(config.crop_start_seconds * config.sampling_rate)),
                    config.streams, config.trials_per_stream,
                )
                channel_itpc, mean_itpc = result.channel_itpc, result.mean_itpc
                atomic_save_npz(
                    checkpoint, coefficients=result.coefficients,
                    channel_itpc=channel_itpc, mean_itpc=mean_itpc,
                    frequencies_hz=result.frequencies_hz, stream_ids=np.arange(1, config.streams + 1),
                )
                context.mark_checkpoint(checkpoint, {"participant": participant, "branch": branch})
                del epochs, values
            channel_all[branch_index, participant_index] = channel_itpc
            mean_all[branch_index, participant_index] = mean_itpc
            for stream_index in range(config.streams):
                for harmonic_index, frequency in enumerate(config.harmonics_hz):
                    rows.append({
                        "participant": participant, "branch": branch,
                        "stream_id": stream_index + 1, "condition_id": stream_index // 3 + 1,
                        "harmonic_hz": frequency, "mean_squared_itpc": float(mean_itpc[stream_index, harmonic_index]),
                    })
            outputs.append(checkpoint)
            completed += 1
            context.progress(stage, completed, total, f"{participant} {branch}")
    combined_path = context.run_dir / "harmonic_itpc.npz"
    atomic_save_npz(
        combined_path, mean_itpc=mean_all, channel_itpc=channel_all,
        participant_ids=np.asarray(context.participants), branch_ids=np.asarray(context.branches),
        stream_ids=np.arange(1, config.streams + 1), frequencies_hz=np.asarray(config.harmonics_hz),
    )
    table_path = context.table_dir / "harmonic_itpc.csv"
    report_path = context.run_dir / "spectral.json"
    pd.DataFrame(rows).to_csv(table_path, index=False)
    report = {
        "passed": True, "branches": list(context.branches), "participants": list(context.participants),
        "method": "first 500 ms removed; 4.5-s symmetric Hann; amplitude-normalized rFFT; squared ITPC",
        "shape": list(mean_all.shape),
    }
    atomic_write_json(report_path, report)
    outputs.extend([combined_path, table_path, report_path])
    context.finish_stage(stage, outputs, {"passed": True, "products": total})
    return outputs, report
