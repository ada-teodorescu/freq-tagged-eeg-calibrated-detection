from __future__ import annotations

import numpy as np
from scipy.io import loadmat

from ..runtime import RunContext, atomic_save_npz, atomic_write_json


def run(context: RunContext, force: bool = False) -> tuple[list, dict]:
    stage = "01_stored_coefficients"
    config = context.config
    context.start_stage(stage, 25)
    source = config.data_root / "Matlab variables" / "Data_fft_all.mat"
    data = loadmat(source, variable_names=["Data_fft"], verify_compressed_data_integrity=True)["Data_fft"]
    if data.shape != (25, 32, 222, 45, 10):
        raise ValueError(f"Unexpected Data_fft shape {data.shape}")
    frequency_axis = np.arange(4, 226, dtype=float) * config.sampling_rate / 4500.0
    targets = np.asarray([np.argmin(np.abs(frequency_axis - value)) for value in config.harmonics_hz])
    half = config.neighbor_bins_per_harmonic // 2
    offsets = np.concatenate((np.arange(-half, 0), np.arange(1, half + 1)))
    neighbor_indices = targets[:, None] + offsets[None, :]
    if neighbor_indices.min() < 0 or neighbor_indices.max() >= frequency_axis.size:
        raise ValueError("Neighbor-bin configuration exceeds stored spectrum")
    outputs = []
    for participant in range(25):
        path = context.cache_dir / "stored_matlab" / f"participant_{participant:02d}.npz"
        if not context.checkpoint_valid(path) or force:
            # stream x trial x channel x harmonic[/offset]
            participant_data = data[participant]
            signal = np.transpose(np.take(participant_data, targets, axis=1), (2, 3, 0, 1))
            null = np.empty((45, 10, 32, 4, len(offsets)), dtype=np.complex128)
            for harmonic, indices in enumerate(neighbor_indices):
                null[..., harmonic, :] = np.transpose(
                    np.take(participant_data, indices, axis=1), (2, 3, 0, 1)
                )
            atomic_save_npz(path, signal=signal, null=null)
            context.mark_checkpoint(path, {"branch": "stored_matlab", "participant": participant})
        outputs.append(path)
        context.progress(stage, participant + 1, 25, f"participant {participant + 1}")
    report = {
        "branch": "stored_matlab", "source_shape": list(data.shape),
        "target_frequencies_hz": frequency_axis[targets].tolist(),
        "neighbor_offsets_bins": offsets.tolist(),
        "cache_dtype": "complex128", "participants": 25,
    }
    report_path = context.run_dir / "stored_coefficient_cache.json"
    atomic_write_json(report_path, report)
    outputs.append(report_path)
    context.finish_stage(stage, outputs, report)
    return outputs, report
