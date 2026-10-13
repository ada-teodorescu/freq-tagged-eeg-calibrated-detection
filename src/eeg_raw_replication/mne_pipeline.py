from __future__ import annotations

from pathlib import Path
import os

import numpy as np

from .config import RawReplicationConfig
from .events import BrainVisionHeader, target_events, validate_events
from .runtime import _replace_with_retry


def require_mne():
    try:
        import mne
    except ImportError as error:
        raise RuntimeError(
            "MNE is required for raw preprocessing. Install the optional stack with "
            "`pip install -e .[raw]` or `uv sync --extra raw`."
        ) from error
    return mne


def frontal_power_scores(components: np.ndarray, channel_names: list[str], frontal: tuple[str, ...]) -> np.ndarray:
    maps = np.asarray(components, dtype=float)
    if maps.ndim != 2 or maps.shape[0] != len(channel_names):
        raise ValueError("components must have shape channel x component")
    lookup = {name.casefold(): index for index, name in enumerate(channel_names)}
    try:
        indices = [lookup[name.casefold()] for name in frontal]
    except KeyError as error:
        raise ValueError(f"Missing frontal ICA channel {error.args[0]!r}") from error
    power = np.square(maps)
    total = power.sum(axis=0)
    return np.divide(power[indices].sum(axis=0), total, out=np.zeros_like(total), where=total > 0)


def paper_component_exclusions(
    components: np.ndarray,
    channel_names: list[str],
    frontal: tuple[str, ...],
    threshold: float,
) -> tuple[np.ndarray, list[int]]:
    scores = frontal_power_scores(components, channel_names, frontal)
    return scores, np.flatnonzero(scores > threshold).astype(int).tolist()


def filter_diagnostics(config: RawReplicationConfig, lowpass_only: bool = False) -> dict[str, float | int | None | str]:
    """Return the realized locked-MNE FIR design used by :func:`make_epochs`."""
    mne = require_mne()
    l_freq = None if lowpass_only else config.l_freq
    taps = mne.filter.create_filter(
        data=None, sfreq=config.sampling_rate, l_freq=l_freq, h_freq=config.h_freq,
        method=config.filter_method, phase=config.filter_phase,
        fir_window=config.fir_window, fir_design=config.fir_design, verbose="ERROR",
    )
    low_transition = None if l_freq is None else min(max(0.25 * l_freq, 2.0), l_freq)
    high_transition = min(
        max(0.25 * config.h_freq, 2.0), config.sampling_rate / 2.0 - config.h_freq,
    )
    return {
        "l_freq_hz": l_freq, "h_freq_hz": config.h_freq,
        "low_transition_bandwidth_hz": low_transition,
        "high_transition_bandwidth_hz": high_transition,
        "filter_length_samples": int(len(taps)),
        "method": config.filter_method, "phase": config.filter_phase,
        "fir_window": config.fir_window, "fir_design": config.fir_design,
    }


def _filtered_raw(header: BrainVisionHeader, config: RawReplicationConfig, l_freq: float | None):
    mne = require_mne()
    raw = mne.io.read_raw_brainvision(header.vhdr_path, preload=True, verbose="ERROR")
    raw.set_montage(config.montage, on_missing="raise", verbose="ERROR")
    raw.filter(
        l_freq=l_freq, h_freq=config.h_freq, method=config.filter_method,
        phase=config.filter_phase, fir_window=config.fir_window, fir_design=config.fir_design,
        skip_by_annotation="edge", verbose="ERROR",
    )
    if config.average_reference:
        raw.set_eeg_reference(ref_channels="average", projection=False, verbose="ERROR")
    return raw


def make_epochs(header: BrainVisionHeader, config: RawReplicationConfig, lowpass_only: bool = False):
    mne = require_mne()
    events = target_events(header.vmrk_path)
    validate_events(events, config.streams, config.trials_per_stream)
    raw = _filtered_raw(header, config, None if lowpass_only else config.l_freq)
    epochs = mne.Epochs(
        raw, events, event_id={f"stream_{index:02d}": index for index in range(1, config.streams + 1)},
        tmin=0.0, tmax=(config.samples_per_epoch - 1) / config.sampling_rate,
        baseline=None, detrend=None, preload=True, reject_by_annotation=False,
        event_repeated="error", verbose="ERROR",
    )
    if len(epochs) != config.expected_trials or epochs.get_data(copy=False).shape != (
        config.expected_trials, config.channels, config.samples_per_epoch,
    ):
        raise ValueError(f"Unexpected epoch shape {epochs.get_data(copy=False).shape}")
    if not np.all(np.isfinite(epochs.get_data(copy=False))):
        raise ValueError(f"Non-finite samples in {header.participant}")
    return epochs


def fit_ica(epochs, config: RawReplicationConfig):
    mne = require_mne()
    rank = int(mne.compute_rank(epochs, rank=None, verbose="ERROR").get("eeg", 0))
    if rank != config.ica_components:
        raise ValueError(f"Expected EEG rank {config.ica_components}, found {rank}")
    ica = mne.preprocessing.ICA(
        n_components=config.ica_components, method="infomax",
        fit_params={"extended": True}, random_state=config.seed, max_iter="auto",
    )
    ica.fit(epochs, decim=config.ica_decimation, verbose="ERROR")
    return ica, rank


def iclabel_exclusions(epochs, ica, threshold: float) -> tuple[list[str], np.ndarray, list[int]]:
    try:
        from mne_icalabel import label_components
    except ImportError as error:
        raise RuntimeError(
            "The iclabel branch requires mne-icalabel and its ONNX runtime backend; "
            "install the project `raw` extra."
        ) from error
    labels = label_components(epochs, ica, method="iclabel")
    names = [str(value) for value in labels["labels"]]
    probabilities = np.asarray(labels["y_pred_proba"], dtype=float)
    excluded = [
        index for index, (name, probability) in enumerate(zip(names, probabilities, strict=True))
        if name.casefold() in {"eye blink", "eye"} and probability >= threshold
    ]
    return names, probabilities, excluded


def apply_ica(epochs, ica, excluded: list[int]):
    cleaned = epochs.copy()
    ica.apply(cleaned, exclude=excluded, verbose="ERROR")
    return cleaned


def save_epochs_atomic(epochs, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem.replace("-epo", "-temporary-epo") + path.suffix)
    epochs.save(temporary, fmt="single", overwrite=True, split_size="2GB", verbose="ERROR")
    _replace_with_retry(temporary, path)


def save_ica_atomic(ica, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem.replace("-ica", "-temporary-ica") + path.suffix)
    ica.save(temporary, overwrite=True, verbose="ERROR")
    _replace_with_retry(temporary, path)
