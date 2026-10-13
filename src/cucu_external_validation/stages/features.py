from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from cucu_article.runtime import atomic_save_npz
from cucu_detection.estimators import hilbert_coefficients
from ..runtime import RunContext


def feature_path(context: RunContext, participant: str, task: str) -> Path:
    return context.checkpoint_dir / "04_hilbert_features" / f"{participant}_{task}.npz"


def _prepare_raw(vhdr_path: Path, config, dataset_bad_targets: tuple[str, ...]):
    import mne
    raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose="ERROR")
    eeg = [name for name in raw.ch_names if raw.get_channel_types(picks=[name])[0] == "eeg"]
    raw.pick(eeg)
    # BrainVision declares four external channels as EEG, but they have no scalp
    # coordinates. The design locks interpolation to the 64-channel scalp montage.
    standard = mne.channels.make_standard_montage("standard_1020")
    positioned = [name for name in raw.ch_names if name in set(standard.ch_names)]
    raw.pick(positioned)
    present = set(raw.ch_names)
    undeclared = sorted(set(config.target_channels) - present - set(config.structural_interpolation_targets))
    if undeclared: raise ValueError(f"undeclared missing target channels: {undeclared}")
    missing = [name for name in config.structural_interpolation_targets if name not in present]
    if missing:
        info = mne.create_info(missing, raw.info["sfreq"], ch_types="eeg")
        zeros = np.zeros((len(missing), raw.n_times), dtype=float)
        raw.add_channels([mne.io.RawArray(zeros, info, verbose="ERROR")], force_update_info=True)
    raw.set_montage(standard, on_missing="raise", verbose="ERROR")
    marked = sorted(set(missing) | (set(dataset_bad_targets) & set(raw.ch_names)))
    raw.info["bads"] = marked
    raw.filter(config.l_freq, config.h_freq, method="fir", phase="zero", fir_window="hamming",
               fir_design="firwin", verbose="ERROR")
    if marked: raw.interpolate_bads(reset_bads=True, verbose="ERROR")
    raw.resample(config.sampling_rate, npad="auto", verbose="ERROR")
    if config.average_reference: raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    raw.pick(list(config.target_channels))
    if tuple(raw.ch_names) != config.target_channels:
        raise ValueError("target channel order mismatch after preprocessing")
    return raw, missing, marked


def run(context: RunContext, force: bool = False) -> tuple[list[Path], dict]:
    stage, config = "04_hilbert_features", context.config
    membership = pd.read_csv(context.table_dir / "negative_unit_membership.csv")
    eligibility = pd.read_csv(context.table_dir / "preprocessing_eligibility.csv")
    groups = list(membership.groupby(["participant", "task"], sort=True))
    context.start_stage(stage, len(groups))
    outputs, artifact_rows = [], []
    for index, ((participant, task), rows) in enumerate(groups, 1):
        destination = feature_path(context, participant, task)
        if context.checkpoint_valid(destination) and not force:
            with np.load(destination) as cached:
                if "peak_to_peak_uv" in cached and "sensitivity_rejected" in cached:
                    ordered = rows.sort_values(["unit", "stream", "trial"])
                    peak_to_peak = cached["peak_to_peak_uv"].reshape(-1)
                    rejected = cached["sensitivity_rejected"].reshape(-1)
                    if len(ordered) != len(peak_to_peak):
                        raise ValueError(f"artifact checkpoint shape mismatch for {participant} {task}")
                    for row, value, reject in zip(ordered.itertuples(index=False), peak_to_peak, rejected):
                        artifact_rows.append({"participant": participant, "task": task, "unit": row.unit,
                                              "stream": row.stream, "trial": row.trial,
                                              "peak_to_peak_uv": float(value),
                                              "sensitivity_rejected": bool(reject)})
                    outputs.append(destination)
                    context.progress(stage, index, len(groups), f"{participant} {task}")
                    continue
        eligibility_row = eligibility[(eligibility["participant"] == participant) & (eligibility["task"] == task)].iloc[0]
        bad_targets = tuple(json.loads(eligibility_row["dataset_bad_target_channels"]))
        raw, structural_missing, interpolated = _prepare_raw(Path(rows.iloc[0]["vhdr_path"]), config, bad_targets)
        rows = rows.sort_values(["unit", "stream", "trial"])
        epoch_samples = int(round(config.epoch_seconds * config.sampling_rate))
        epoch_data, local_peak_to_peak = [], []
        for row in rows.itertuples(index=False):
            start = int(round(float(row.start_seconds) * config.sampling_rate)); stop = start + epoch_samples
            data = raw.get_data(start=start, stop=stop).astype(np.float32)
            if data.shape != (len(config.target_channels), epoch_samples):
                raise ValueError(f"short epoch for {participant} {task} unit {row.unit}")
            peak_to_peak_uv = float(np.max(np.ptp(data, axis=1)) * 1e6)
            local_peak_to_peak.append(peak_to_peak_uv)
            artifact_rows.append({"participant": participant, "task": task, "unit": row.unit,
                                  "stream": row.stream, "trial": row.trial, "peak_to_peak_uv": peak_to_peak_uv,
                                  "sensitivity_rejected": peak_to_peak_uv > config.peak_to_peak_sensitivity_uv})
            epoch_data.append(data)
        epochs = np.stack(epoch_data)
        coefficients = hilbert_coefficients(epochs, config.sampling_rate, np.asarray(config.harmonics_hz),
                                            order=4, half_bandwidth_hz=0.5, edge_crop_seconds=0.5)
        units = rows["unit"].nunique()
        coefficients = coefficients.reshape(units, config.streams_per_unit, config.trials_per_stream,
                                            len(config.target_channels), len(config.harmonics_hz))
        peak_to_peak = np.asarray(local_peak_to_peak, dtype=np.float32).reshape(
            units, config.streams_per_unit, config.trials_per_stream)
        sensitivity_rejected = peak_to_peak > config.peak_to_peak_sensitivity_uv
        atomic_save_npz(destination, coefficients=coefficients.astype(np.complex64),
                        participant=np.asarray(participant), task=np.asarray(task),
                        channel_labels=np.asarray(config.target_channels), harmonics_hz=np.asarray(config.harmonics_hz),
                        structural_missing=np.asarray(structural_missing), interpolated=np.asarray(interpolated),
                        peak_to_peak_uv=peak_to_peak, sensitivity_rejected=sensitivity_rejected)
        context.mark_checkpoint(destination, {"participant": participant, "task": task, "units": units})
        outputs.append(destination); context.progress(stage, index, len(groups), f"{participant} {task}")
    artifact_path = context.table_dir / "epoch_artifact_sensitivity.csv"
    pd.DataFrame(artifact_rows).to_csv(artifact_path, index=False); outputs.append(artifact_path)
    summary = {"recording_features": len(groups), "feature_contract": "unit x 3 streams x 10 trials x 32 channels x 4 harmonics"}
    context.finish_stage(stage, outputs, summary); return outputs, summary
