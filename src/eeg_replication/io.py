from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from scipy.io import loadmat

from .config import ReplicationConfig


SUBJECT_RE = re.compile(r"^(P\d+)\s+E2\s+epochs\.set$", re.IGNORECASE)


def _scalar(value):
    array = np.asarray(value)
    return array.reshape(-1)[0] if array.size else value


def trigger_to_stream(trigger: object) -> int | None:
    text = re.sub(r"\s+", "", str(_scalar(trigger))).upper()
    if not text.startswith("S1") or not text[2:].isdigit():
        return None
    stream = int(text[2:])
    return stream if 1 <= stream <= 45 else None


def natural_subject_key(path_or_name: str | Path) -> int:
    match = re.search(r"P(\d+)", str(path_or_name), re.IGNORECASE)
    if match is None:
        raise ValueError(f"No participant number in {path_or_name}")
    return int(match.group(1))


@dataclass(frozen=True)
class EpochFile:
    participant: str
    set_path: Path
    fdt_path: Path
    n_channels: int
    n_times: int
    n_trials: int
    sampling_rate: float
    stream_by_trial: np.ndarray
    channel_labels: tuple[str, ...]
    event_count: int
    has_ica_weights: bool = False
    ica_components: int = 0


def inspect_set(path: Path) -> EpochFile:
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    eeg = mat["EEG"]
    data_name = str(_scalar(eeg.data))
    fdt_path = path.with_name(data_name)
    if not fdt_path.exists():
        fdt_path = path.with_suffix(".fdt")
    if not fdt_path.exists():
        raise FileNotFoundError(f"Missing FDT for {path}")
    n_trials = int(_scalar(eeg.trials))
    stream_by_trial = np.zeros(n_trials, dtype=np.int16)
    events = np.atleast_1d(eeg.event).reshape(-1)
    for event in events:
        stream = trigger_to_stream(event.type)
        if stream is None or not hasattr(event, "epoch"):
            continue
        epoch = int(float(_scalar(event.epoch))) - 1
        if 0 <= epoch < n_trials:
            previous = int(stream_by_trial[epoch])
            if previous not in (0, stream):
                raise ValueError(f"Conflicting stream triggers in {path}, epoch {epoch + 1}")
            stream_by_trial[epoch] = stream
    labels = []
    for channel in np.atleast_1d(getattr(eeg, "chanlocs", [])).reshape(-1):
        label = str(_scalar(getattr(channel, "labels", ""))).strip()
        labels.append(label)
    match = SUBJECT_RE.match(path.name)
    if match is None:
        raise ValueError(f"Unexpected EEGLAB filename: {path.name}")
    icaweights = np.asarray(getattr(eeg, "icaweights", []))
    has_ica_weights = bool(icaweights.size)
    return EpochFile(
        participant=match.group(1).upper(),
        set_path=path,
        fdt_path=fdt_path,
        n_channels=int(_scalar(eeg.nbchan)),
        n_times=int(_scalar(eeg.pnts)),
        n_trials=n_trials,
        sampling_rate=float(_scalar(eeg.srate)),
        stream_by_trial=stream_by_trial,
        channel_labels=tuple(labels),
        event_count=len(events),
        has_ica_weights=has_ica_weights,
        ica_components=int(icaweights.shape[0]) if icaweights.ndim >= 2 else 0,
    )


def discover_epoch_files(config: ReplicationConfig) -> list[EpochFile]:
    folder = config.data_root / "Pre-processed EEGLAB datasets"
    paths = sorted(folder.glob("*.set"), key=natural_subject_key)
    return [inspect_set(path) for path in paths]


def validate_epoch_file(epoch: EpochFile, config: ReplicationConfig) -> list[str]:
    errors = []
    expected_trials = config.expected_streams * config.trials_per_stream
    checks = {
        "channels": (epoch.n_channels, config.expected_channels),
        "samples": (epoch.n_times, config.samples_per_epoch),
        "trials": (epoch.n_trials, expected_trials),
        "sampling rate": (int(round(epoch.sampling_rate)), config.sampling_rate),
    }
    for label, (actual, expected) in checks.items():
        if actual != expected:
            errors.append(f"{epoch.participant}: {label}={actual}, expected {expected}")
    counts = np.bincount(epoch.stream_by_trial, minlength=config.expected_streams + 1)[1:]
    if counts.size != config.expected_streams or not np.all(counts == config.trials_per_stream):
        bad = {index + 1: int(count) for index, count in enumerate(counts) if count != config.trials_per_stream}
        errors.append(f"{epoch.participant}: invalid stream trial counts {bad}")
    expected_bytes = epoch.n_channels * epoch.n_times * epoch.n_trials * 4
    if epoch.fdt_path.stat().st_size != expected_bytes:
        errors.append(
            f"{epoch.participant}: FDT bytes={epoch.fdt_path.stat().st_size}, expected {expected_bytes}"
        )
    return errors


def open_fdt(epoch: EpochFile) -> np.memmap:
    return np.memmap(
        epoch.fdt_path,
        dtype="<f4",
        mode="r",
        shape=(epoch.n_channels, epoch.n_times, epoch.n_trials),
        order="F",
    )


def load_stream_trials(epoch: EpochFile, stream_id: int, dtype=np.float64) -> np.ndarray:
    indices = np.flatnonzero(epoch.stream_by_trial == stream_id)
    if indices.size != 10:
        raise ValueError(f"{epoch.participant} stream {stream_id}: expected 10 trials")
    data = open_fdt(epoch)
    values = np.asarray(data[:, :, indices], dtype=dtype)
    return np.moveaxis(values, -1, 0)  # trial x channel x time
