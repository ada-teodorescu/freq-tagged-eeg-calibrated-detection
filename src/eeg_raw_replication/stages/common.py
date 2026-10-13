from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from ..events import discover_headers
from ..mne_pipeline import apply_ica, make_epochs, require_mne
from ..runtime import RunContext


def preprocess_paths(context: RunContext, participant: str) -> dict[str, Path]:
    root = context.checkpoint_dir / "02_preprocess" / participant
    return {
        "base": root / f"{participant}-base-epo.fif",
        "paper12": root / f"{participant}-paper12-epo.fif",
        "ica": root / f"{participant}-ica.fif",
        "decisions": root / "component_decisions.json",
        "summary": root / "summary.json",
    }


def participant_header(context: RunContext, participant: str):
    headers = discover_headers(context.config.paths.raw_root)
    if participant not in headers:
        raise FileNotFoundError(f"Missing raw header for {participant}")
    return headers[participant]


def load_branch_epochs(context: RunContext, participant: str, branch: str):
    mne = require_mne()
    paths = preprocess_paths(context, participant)
    if branch == "paper12":
        return mne.read_epochs(paths["paper12"], preload=True, verbose="ERROR")
    if branch == "no_ica":
        return mne.read_epochs(paths["base"], preload=True, verbose="ERROR")
    if branch == "lowpass_no_ica":
        return make_epochs(participant_header(context, participant), context.config, lowpass_only=True)
    if branch == "iclabel":
        epochs = mne.read_epochs(paths["base"], preload=True, verbose="ERROR")
        ica = mne.preprocessing.read_ica(paths["ica"], verbose="ERROR")
        decisions = json.loads(paths["decisions"].read_text(encoding="utf-8"))
        return apply_ica(epochs, ica, [int(value) for value in decisions["iclabel"]["excluded_components"]])
    raise ValueError(f"Unknown branch {branch!r}")


def epoch_values(epochs) -> tuple[np.ndarray, np.ndarray]:
    values = epochs.get_data(copy=False)
    stream_ids = np.asarray(epochs.events[:, 2], dtype=np.int16)
    return values, stream_ids
