from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from scipy.io import loadmat


SET_RE = re.compile(r"^(P\d+)\s+E2\s+epochs\.set$", re.IGNORECASE)


def _scalar(value):
    array = np.asarray(value)
    return array.reshape(-1)[0] if array.size else value


@dataclass(frozen=True)
class ReferenceEpochs:
    participant: str
    set_path: Path
    fdt_path: Path
    channels: tuple[str, ...]
    stream_ids: np.ndarray
    n_channels: int
    n_times: int
    n_trials: int
    sampling_rate: float


def _trigger_to_stream(value: object) -> int | None:
    text = re.sub(r"\s+", "", str(_scalar(value))).upper()
    if not text.startswith("S1") or not text[2:].isdigit():
        return None
    stream = int(text[2:])
    return stream if 1 <= stream <= 45 else None


def inspect_set(path: Path) -> ReferenceEpochs:
    eeg = loadmat(path, squeeze_me=True, struct_as_record=False)["EEG"]
    data_name = str(_scalar(eeg.data))
    fdt_path = path.with_name(data_name)
    if not fdt_path.exists():
        fdt_path = path.with_suffix(".fdt")
    if not fdt_path.exists():
        raise FileNotFoundError(f"Missing FDT for {path}")
    match = SET_RE.match(path.name)
    if match is None:
        raise ValueError(f"Unexpected EEGLAB filename: {path.name}")
    n_trials = int(_scalar(eeg.trials))
    stream_ids = np.zeros(n_trials, dtype=np.int16)
    for event in np.atleast_1d(eeg.event).reshape(-1):
        stream = _trigger_to_stream(event.type)
        if stream is None or not hasattr(event, "epoch"):
            continue
        epoch = int(float(_scalar(event.epoch))) - 1
        if 0 <= epoch < n_trials:
            if stream_ids[epoch] not in (0, stream):
                raise ValueError(f"Conflicting event in {path}, epoch {epoch + 1}")
            stream_ids[epoch] = stream
    channels = tuple(
        str(_scalar(getattr(channel, "labels", ""))).strip()
        for channel in np.atleast_1d(getattr(eeg, "chanlocs", [])).reshape(-1)
    )
    return ReferenceEpochs(
        participant=match.group(1).upper(), set_path=path, fdt_path=fdt_path,
        channels=channels, stream_ids=stream_ids,
        n_channels=int(_scalar(eeg.nbchan)), n_times=int(_scalar(eeg.pnts)),
        n_trials=n_trials, sampling_rate=float(_scalar(eeg.srate)),
    )


def discover_references(folder: Path) -> dict[str, ReferenceEpochs]:
    references = {}
    for path in folder.glob("*.set"):
        item = inspect_set(path)
        references[item.participant] = item
    return references


def open_fdt(reference: ReferenceEpochs) -> np.memmap:
    return np.memmap(
        reference.fdt_path, dtype="<f4", mode="r",
        shape=(reference.n_channels, reference.n_times, reference.n_trials), order="F",
    )


def reference_as_trials(reference: ReferenceEpochs, dtype=np.float32) -> np.ndarray:
    """Return trial x channel x time in EEGLAB's native microvolt scale."""
    return np.moveaxis(np.asarray(open_fdt(reference), dtype=dtype), -1, 0)


def load_released_harmonics(matlab_root: Path, harmonics_hz: tuple[float, ...]) -> np.ndarray:
    values = loadmat(matlab_root / "Itc_for_stats.mat", squeeze_me=True)
    stacked = np.stack([
        np.asarray(values[f"meanitc{int(frequency)}"], dtype=float)
        for frequency in harmonics_hz
    ], axis=-1)
    return np.transpose(stacked, (1, 0, 2))  # participant x condition x harmonic


def load_released_stream_itpc(original_repo_root: Path) -> np.ndarray:
    path = original_repo_root / "Data files" / "Itc_for_pca_oana.mat"
    values = np.asarray(loadmat(path, squeeze_me=True)["Itc_for_pca"], dtype=float)
    return values.transpose(1, 0, 2)  # participant x stream x harmonic
