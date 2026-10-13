from __future__ import annotations

import time
import numpy as np
from scipy import signal

from eeg_detection.estimators import hilbert_coefficients, thomson_line_from_spectra
from eeg_replication.io import inspect_set, load_stream_trials, natural_subject_key
from ..runtime import RunContext, atomic_save_npz, atomic_write_json


def _frequency_indices(config, n_times: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target = np.rint(np.asarray(config.harmonics_hz) * n_times / config.sampling_rate).astype(int)
    half = config.neighbor_bins_per_harmonic // 2
    offsets = np.concatenate((np.arange(-half, 0), np.arange(1, half + 1)))
    return target, target[:, None] + offsets[None, :], offsets


def _spectral_coefficients(trials: np.ndarray, config) -> dict[str, np.ndarray]:
    values = np.asarray(trials, dtype=np.float64)[..., config.fft_crop_samples:]
    n_times = values.shape[-1]; target, neighbors, _ = _frequency_indices(config, n_times)
    hann_spectra = np.fft.rfft(values * np.hanning(n_times), axis=-1)
    f1_target = np.take(hann_spectra, target, axis=-1)
    f1_neighbors = np.take(hann_spectra, neighbors.reshape(-1), axis=-1).reshape(
        *values.shape[:-1], len(target), neighbors.shape[1]
    )
    windows, ratios = signal.windows.dpss(
        n_times, config.dpss_time_bandwidth, Kmax=config.dpss_tapers,
        sym=False, norm=2, return_ratios=True,
    )
    response = windows.sum(axis=-1)
    line_target = np.empty((*values.shape[:-1], len(target)), dtype=np.complex128)
    f_target = np.empty(values.shape[:-1] + (len(target),), dtype=float)
    line_neighbors = np.empty((*values.shape[:-1], len(target), neighbors.shape[1]), dtype=np.complex128)
    f_neighbors = np.empty(values.shape[:-1] + (len(target), neighbors.shape[1]), dtype=float)
    for trial in range(values.shape[0]):
        spectra = np.fft.rfft(values[trial, :, None, :] * windows, axis=-1)
        selected_target = np.swapaxes(np.take(spectra, target, axis=-1), -1, -2)
        line_target[trial], f_target[trial] = thomson_line_from_spectra(selected_target, response)
        selected_neighbors = np.take(spectra, neighbors.reshape(-1), axis=-1).reshape(
            values.shape[1], config.dpss_tapers, len(target), neighbors.shape[1]
        ).transpose(0, 2, 3, 1)
        line_neighbors[trial], f_neighbors[trial] = thomson_line_from_spectra(selected_neighbors, response)
    return {
        "F1_target": f1_target, "F1_neighbors": f1_neighbors,
        "F3_line_target": line_target, "F3_line_neighbors": line_neighbors,
        "F3_f_target": f_target, "F3_f_neighbors": f_neighbors,
        "dpss_ratios": np.asarray(ratios, dtype=float),
    }


def run(context: RunContext, force: bool = False) -> tuple[list, dict]:
    stage = "02_public_coefficients"; config = context.config
    set_paths = sorted((config.data_root / "Pre-processed EEGLAB datasets").glob("*.set"), key=natural_subject_key)
    epochs = [inspect_set(path) for path in set_paths]
    total = len(epochs) * 45; context.start_stage(stage, total)
    outputs = []; completed = 0; started = time.perf_counter()
    for participant, epoch in enumerate(epochs):
        for stream in range(1, 46):
            path = context.cache_dir / "public_set_fdt" / f"participant_{participant:02d}" / f"stream_{stream:02d}.npz"
            if not context.checkpoint_valid(path) or force:
                trials = load_stream_trials(epoch, stream)
                arrays = _spectral_coefficients(trials, config)
                arrays["F2_target"] = hilbert_coefficients(
                    trials, config.sampling_rate, np.asarray(config.harmonics_hz),
                    config.hilbert_order, config.hilbert_half_bandwidth_hz,
                    config.hilbert_edge_crop_seconds,
                )
                atomic_save_npz(path, **arrays)
                context.mark_checkpoint(path, {"participant": participant, "stream": stream,
                    "hilbert": "zero-phase bandpass, analytic signal, complex demodulation",
                    "multitaper": "DPSS tapered spectra, Thomson regression line and F statistic"})
            outputs.append(path); completed += 1
            context.progress(stage, completed, total, f"{epoch.participant} stream {stream}")
    _, _, offsets = _frequency_indices(config, 5000 - config.fft_crop_samples)
    report = {"source": "public_set_fdt", "participants": len(epochs), "streams": 45,
              "neighbor_bin_offsets": offsets.tolist(), "hilbert_demodulated": True,
              "dpss_symmetry": "periodic (sym=False)", "thomson_nominal_df": [2, 2 * config.dpss_tapers - 2],
              "runtime_seconds": time.perf_counter() - started, "cross_source_equivalence_claim": False}
    report_path = context.run_dir / "corrected_public_coefficients.json"; atomic_write_json(report_path, report)
    outputs.append(report_path); context.finish_stage(stage, outputs, report); return outputs, report
