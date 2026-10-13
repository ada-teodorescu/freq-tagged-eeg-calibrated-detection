from __future__ import annotations

import json
from pathlib import Path
import time
import tracemalloc
import numpy as np
import pandas as pd

from cucu_replication.io import inspect_set, load_stream_trials, natural_subject_key

from ..estimators import (
    dpss_multitaper_coefficients,
    hilbert_coefficients,
    least_squares_coefficients,
    published_hann_dft,
)
from ..runtime import RunContext, atomic_save_npz, atomic_write_json


def _measured_frequencies(context: RunContext) -> tuple[np.ndarray, str]:
    latest = context.config.replication_output_root / "latest.json"
    if latest.exists():
        run_dir = json.loads(latest.read_text(encoding="utf-8"))["run_dir"]
        run_path = Path(run_dir)
        if not run_path.is_absolute():
            run_path = context.config.project_root / run_path
        table = pd.read_csv(run_path / "tables" / "stream_frequency_estimates.csv")
        values = table.sort_values("stream_id")["fitted_frequency_hz"].to_numpy(float)
        if values.shape == (45,): return values, str(latest)
    return np.full(45, 4.0), "nominal_fallback_missing_replication_frequency_table"


def run(context: RunContext, force: bool = False) -> tuple[list, dict]:
    stage = "04_public_estimators"
    config = context.config
    set_paths = sorted((config.data_root / "Pre-processed EEGLAB datasets").glob("*.set"), key=natural_subject_key)
    epochs = [inspect_set(path) for path in set_paths]
    measured, frequency_source = _measured_frequencies(context)
    total = len(epochs) * 45
    context.start_stage(stage, total)
    tracemalloc.start(); started = time.perf_counter()
    outputs = []; completed = 0
    nominal = np.asarray(config.harmonics_hz)
    for participant, epoch in enumerate(epochs):
        for stream in range(1, 46):
            path = context.cache_dir / "public_set_fdt" / f"participant_{participant:02d}" / f"stream_{stream:02d}.npz"
            if not context.checkpoint_valid(path) or force:
                trials = load_stream_trials(epoch, stream)
                measured_harmonics = measured[stream - 1] * np.arange(1, 5)
                arrays = {
                    "F1": published_hann_dft(trials, config.sampling_rate, nominal, config.fft_crop_samples),
                    "F2": hilbert_coefficients(trials, config.sampling_rate, nominal,
                        config.hilbert_order, config.hilbert_half_bandwidth_hz,
                        config.hilbert_edge_crop_seconds),
                    "F3": dpss_multitaper_coefficients(trials, config.sampling_rate, nominal,
                        config.fft_crop_samples, config.dpss_time_bandwidth, config.dpss_tapers),
                    "F4_nominal": least_squares_coefficients(trials, config.sampling_rate, nominal,
                        config.fft_crop_samples),
                    "F4_measured": least_squares_coefficients(trials, config.sampling_rate,
                        measured_harmonics, config.fft_crop_samples),
                }
                atomic_save_npz(path, **arrays)
                context.mark_checkpoint(path, {"branch": "public_set_fdt", "participant": participant,
                                               "stream": stream, "frequency_source": frequency_source})
            outputs.append(path); completed += 1
            context.progress(stage, completed, total, f"{epoch.participant} stream {stream}")
    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    report = {"branch": "public_set_fdt", "participants": len(epochs), "streams": 45,
              "estimators": list(config.estimators), "frequency_source": frequency_source,
              "runtime_seconds": time.perf_counter() - started,
              "peak_python_memory_bytes": peak,
              "cross_source_numerical_equivalence_claim": False}
    report_path = context.run_dir / "public_estimator_cache.json"
    atomic_write_json(report_path, report); outputs.append(report_path)
    context.finish_stage(stage, outputs, report)
    return outputs, report
