from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from cucu_article.runtime import atomic_save_npz, atomic_write_json
from cucu_detection.estimators import hilbert_coefficients
from cucu_replication.io import inspect_set, load_stream_trials, natural_subject_key
from ..algorithms import (causal_demodulation_coefficients, causal_settling_samples, filtered_ls_coefficients, kernel_metrics,
                          kernel_response, linear_effective_kernel, rectangular_ls_coefficients)
from ..runtime import RunContext


PRIMARY_METHODS = ("H1_Hann4s", "H2_Hilbert4s", "M1_RectLS4s", "M2_FilteredLS4s")


def coefficient_path(context: RunContext, participant: str) -> Path:
    return context.checkpoint_dir / "01_estimators" / f"{participant}.npz"


def _baseline_path(context: RunContext, participant: str) -> Path:
    return context.config.baseline_confirmatory_run / "checkpoints" / "01_coefficients" / f"{participant}.npz"


def _participant(context: RunContext, set_path: Path, force: bool) -> tuple[str, Path]:
    config = context.config; epoch = inspect_set(set_path); destination = coefficient_path(context, epoch.participant)
    if context.checkpoint_valid(destination) and not force: return epoch.participant, destination
    with np.load(_baseline_path(context, epoch.participant), allow_pickle=False) as baseline:
        output = {"H1_Hann4s": baseline["hann_matched"], "H2_Hilbert4s": baseline["hilbert"]}
        channel_labels = baseline["channel_labels"]
    shape = output["H1_Hann4s"].shape
    for name in ("M1_RectLS4s", "M2_FilteredLS4s", "S_HilbertBW0p25", "S_HilbertBW1p0",
                 "S_HilbertCrop0p25", "S_HilbertCrop0p75", "S_CausalDemod"):
        output[name] = np.empty(shape, dtype=np.complex64)
    start = int(round(config.window_start_seconds * config.sampling_rate))
    stop = int(round(config.window_end_seconds * config.sampling_rate))
    frequencies = np.asarray(config.harmonics_hz)
    causal_settling = causal_settling_samples(config.sampling_rate, config.hilbert_order,
                                              config.hilbert_half_bandwidth_hz)
    if epoch.n_times - causal_settling < 2 * config.sampling_rate:
        raise ValueError("causal demodulation leaves fewer than two usable seconds")
    for stream in range(1, config.streams + 1):
        raw = load_stream_trials(epoch, stream, dtype=np.float32)
        output["M1_RectLS4s"][stream - 1] = rectangular_ls_coefficients(raw, config.sampling_rate, frequencies, start, stop)
        output["M2_FilteredLS4s"][stream - 1] = filtered_ls_coefficients(
            raw, config.sampling_rate, frequencies, start, stop, config.hilbert_order, config.hilbert_half_bandwidth_hz)
        for bandwidth in config.bandwidth_sensitivity_hz:
            key = f"S_HilbertBW{str(bandwidth).replace('.', 'p')}"
            output[key][stream - 1] = hilbert_coefficients(raw, config.sampling_rate, frequencies,
                config.hilbert_order, bandwidth, config.hilbert_edge_crop_seconds)
        for crop in config.crop_sensitivity_seconds:
            key = f"S_HilbertCrop{str(crop).replace('.', 'p')}"
            output[key][stream - 1] = hilbert_coefficients(raw, config.sampling_rate, frequencies,
                config.hilbert_order, config.hilbert_half_bandwidth_hz, crop)
        output["S_CausalDemod"][stream - 1] = causal_demodulation_coefficients(
            raw, config.sampling_rate, frequencies, config.hilbert_order,
            config.hilbert_half_bandwidth_hz, causal_settling)
    atomic_save_npz(destination, **output, participant=np.asarray(epoch.participant),
                    channel_labels=channel_labels, harmonics_hz=frequencies)
    context.mark_checkpoint(destination, {"participant": epoch.participant, "matched_support": [0.5, 4.5]})
    return epoch.participant, destination


def _kernel_outputs(context: RunContext) -> list[Path]:
    config = context.config; n_times = 5500; frequency = config.kernel_frequency_hz
    estimator = lambda values: hilbert_coefficients(values, config.sampling_rate, np.asarray([frequency]),
        config.hilbert_order, config.hilbert_half_bandwidth_hz, config.hilbert_edge_crop_seconds)[..., 0]
    kernel = linear_effective_kernel(estimator, n_times, config.kernel_chunk_size)
    start = int(round(config.window_start_seconds * config.sampling_rate)); stop = int(round(config.window_end_seconds * config.sampling_rate))
    metrics = kernel_metrics(kernel, config.sampling_rate, frequency, start, stop)
    settling = causal_settling_samples(config.sampling_rate, config.hilbert_order,
                                       config.hilbert_half_bandwidth_hz)
    metrics.update({"frequency_hz": frequency, "causal_settling_samples_99pct": settling,
                    "causal_settling_seconds_99pct": settling / config.sampling_rate,
                    "causal_feasible_two_seconds": bool(n_times - settling >= 2 * config.sampling_rate),
                    "characterization_revision": "signed_frequency_v2"})
    kernel_path = context.table_dir / "hilbert_effective_kernel.csv"
    pd.DataFrame({"sample": np.arange(n_times), "time_seconds": np.arange(n_times) / config.sampling_rate,
                  "real": kernel.real, "imag": kernel.imag, "magnitude": np.abs(kernel)}).to_csv(kernel_path, index=False)
    frequencies, response = kernel_response(kernel, config.sampling_rate)
    selected = np.abs(frequencies) <= 25
    response_path = context.table_dir / "hilbert_kernel_response.csv"
    pd.DataFrame({"frequency_hz": frequencies[selected], "magnitude": response[selected]}).to_csv(response_path, index=False)
    metrics_path = context.run_dir / "kernel_characterization.json"; atomic_write_json(metrics_path, metrics)
    return [kernel_path, response_path, metrics_path]


def run(context: RunContext, force: bool = False) -> tuple[list[Path], dict]:
    stage, config = "01_estimators", context.config
    paths = sorted((config.data_root / "Pre-processed EEGLAB datasets").glob("*.set"), key=natural_subject_key)
    context.start_stage(stage, len(paths) + 1); outputs = []
    if context.workers == 1:
        for index, path in enumerate(paths, 1):
            participant, result = _participant(context, path, force); outputs.append(result)
            context.progress(stage, index, len(paths) + 1, participant)
    else:
        with ThreadPoolExecutor(max_workers=context.workers) as executor:
            futures = [executor.submit(_participant, context, path, force) for path in paths]
            for index, future in enumerate(as_completed(futures), 1):
                participant, result = future.result(); outputs.append(result)
                context.progress(stage, index, len(paths) + 1, participant)
    outputs += _kernel_outputs(context); context.progress(stage, len(paths) + 1, len(paths) + 1, "effective kernel")
    summary = {"participants": len(paths), "primary_methods": list(PRIMARY_METHODS),
               "sensitivity_methods": ["S_HilbertBW0p25", "S_HilbertBW1p0", "S_HilbertCrop0p25", "S_HilbertCrop0p75", "S_CausalDemod"]}
    summary_path = context.run_dir / "estimator_features.json"; atomic_write_json(summary_path, summary); outputs.append(summary_path)
    context.finish_stage(stage, outputs, summary); return outputs, summary
