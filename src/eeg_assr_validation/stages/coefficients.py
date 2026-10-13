from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

from ..algorithms import hilbert_interval_coefficients, sinusoidal_ls_coefficients
from ..runtime import RunContext, atomic_save_npz


def coefficient_path(context: RunContext, participant: str) -> Path:
    return context.checkpoint_dir / "02_coefficients" / f"{participant}.npz"


def _extract(data: np.ndarray, samples: np.ndarray, start: int, stop: int) -> np.ndarray:
    epochs = [data[:, int(sample) + start:int(sample) + stop] for sample in samples]
    expected = stop - start
    if any(epoch.shape != (data.shape[0], expected) for epoch in epochs):
        raise ValueError("event produces a short retained epoch")
    dtype = np.complex64 if np.iscomplexobj(data) else np.float32
    return np.stack(epochs).astype(dtype)


def _participant(context: RunContext, row, membership: pd.DataFrame, force: bool) -> tuple[str, Path, list[dict]]:
    config, participant = context.config, str(row.participant)
    destination = coefficient_path(context, participant)
    if context.checkpoint_valid(destination) and not force:
        with np.load(destination, allow_pickle=False) as archive:
            artifacts = archive["peak_to_peak_uv"]
        artifact_rows = []
        for f_index, frequency in enumerate(config.frequencies_hz):
            for trial, value in enumerate(artifacts[f_index, :config.standard_limit(frequency)], 1):
                artifact_rows.append({"participant": participant, "frequency_hz": frequency, "trial": trial,
                                      "peak_to_peak_uv": float(value),
                                      "sensitivity_rejected": bool(value > config.artifact_peak_to_peak_uv)})
        return participant, destination, artifact_rows
    import mne
    raw = mne.io.read_raw_bdf(Path(row.bdf_path), preload=True, verbose="ERROR")
    channels = pd.read_csv(Path(row.channels_path), sep="\t")
    eeg_names = channels.loc[channels["type"].str.upper() == "EEG", "name"].astype(str).tolist()
    if len(eeg_names) != 64 or not set(eeg_names).issubset(raw.ch_names):
        raise ValueError(f"{participant}: BIDS EEG channel contract mismatch")
    raw.pick(eeg_names)
    raw.notch_filter([config.notch_hz], method="fir", phase="zero", verbose="ERROR")
    raw.filter(config.l_freq, config.h_freq, method="fir", phase="zero", fir_window="hamming",
               fir_design="firwin", verbose="ERROR")
    if config.average_reference:
        raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    if float(raw.info["sfreq"]) != config.sampling_rate:
        raise ValueError(f"{participant}: unexpected sampling rate after preprocessing")
    continuous = raw.get_data().astype(np.float64)
    start = int(round(config.epoch_start_seconds * config.sampling_rate))
    stop = int(round(config.epoch_end_seconds * config.sampling_rate))
    max_trials = max(config.standards_per_frequency)
    output = np.full((3, len(config.frequencies_hz), max_trials, 64), np.nan + 1j * np.nan, dtype=np.complex64)
    artifact = np.full((len(config.frequencies_hz), max_trials), np.nan, dtype=np.float32)
    artifact_rows = []
    for f_index, frequency in enumerate(config.frequencies_hz):
        rows = membership[(membership["participant"] == participant) & (membership["frequency_hz"] == frequency)].sort_values("trial")
        samples = rows["sample"].to_numpy(int)
        epochs = _extract(continuous, samples, start, stop)
        limit = len(samples)
        output[0, f_index, :limit] = sinusoidal_ls_coefficients(epochs, config.sampling_rate, frequency,
                                                               start_sample=start, hann=True)
        output[1, f_index, :limit] = sinusoidal_ls_coefficients(epochs, config.sampling_rate, frequency,
                                                               start_sample=start, hann=False)
        sos = signal.butter(config.hilbert_order,
                            [frequency - config.hilbert_half_bandwidth_hz, frequency + config.hilbert_half_bandwidth_hz],
                            btype="bandpass", fs=config.sampling_rate, output="sos")
        analytic = signal.hilbert(signal.sosfiltfilt(sos, continuous, axis=-1), axis=-1)
        analytic_epochs = _extract(analytic, samples, start, stop)
        output[2, f_index, :limit] = hilbert_interval_coefficients(analytic_epochs, config.sampling_rate,
                                                                   frequency, start_sample=start)
        peak = np.max(np.ptp(epochs, axis=-1), axis=1) * 1e6
        artifact[f_index, :limit] = peak
        for trial, value in enumerate(peak, 1):
            artifact_rows.append({"participant": participant, "frequency_hz": frequency, "trial": trial,
                                  "peak_to_peak_uv": float(value),
                                  "sensitivity_rejected": bool(value > config.artifact_peak_to_peak_uv)})
    atomic_save_npz(destination, coefficients=output, peak_to_peak_uv=artifact,
                    participant=np.asarray(participant), methods=np.asarray(("hann", "rectangular_ls", "hilbert")),
                    frequencies_hz=np.asarray(config.frequencies_hz), channel_labels=np.asarray(eeg_names),
                    retained_samples=np.asarray((start, stop)))
    context.mark_checkpoint(destination, {"participant": participant, "input_fingerprint": str(row.bdf_sha256),
                                          "shape": list(output.shape), "retained_interval": [start, stop],
                                          "parameters": {"filter": [config.l_freq, config.h_freq], "notch": config.notch_hz,
                                                         "reference": "average", "hilbert_order": config.hilbert_order,
                                                         "hilbert_half_bandwidth_hz": config.hilbert_half_bandwidth_hz}})
    return participant, destination, artifact_rows


def run(context: RunContext, force: bool = False) -> tuple[list, dict]:
    stage = "02_coefficients"
    inventory = pd.read_csv(context.table_dir / "recording_inventory.csv")
    membership = pd.read_csv(context.table_dir / "trial_membership.csv")
    rows = list(inventory.itertuples(index=False))
    context.start_stage(stage, len(rows))
    outputs, artifacts = [], []
    if context.workers == 1:
        for index, row in enumerate(rows, 1):
            participant, path, local = _participant(context, row, membership, force)
            outputs.append(path); artifacts += local
            context.progress(stage, index, len(rows), participant)
    else:
        with ThreadPoolExecutor(max_workers=context.workers) as executor:
            futures = [executor.submit(_participant, context, row, membership, force) for row in rows]
            for index, future in enumerate(as_completed(futures), 1):
                participant, path, local = future.result()
                outputs.append(path); artifacts += local
                context.progress(stage, index, len(rows), participant)
    artifact_path = context.table_dir / "trial_artifacts.csv"
    pd.DataFrame(artifacts).sort_values(["participant", "frequency_hz", "trial"]).to_csv(artifact_path, index=False)
    outputs.append(artifact_path)
    summary = {"participants": len(rows), "methods": ["hann", "rectangular_ls", "hilbert"],
               "primary_artifact_rejection": False}
    context.finish_stage(stage, outputs, summary)
    return outputs, summary
