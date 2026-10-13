from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd


TASK_PATTERNS = {
    "ldaep_jittered_auditory": "*task-ldaep_eeg.vhdr",
    "rest_eyes_open": "*task-rest_acq-eo_eeg.vhdr",
    "rest_eyes_closed": "*task-rest_acq-ec_eeg.vhdr",
}


@dataclass(frozen=True)
class Recording:
    participant: str
    task: str
    vhdr_path: Path
    channel_path: Path
    event_path: Path | None


def _companion(path: Path, suffix: str) -> Path:
    stem = path.name.removesuffix("_eeg.vhdr")
    return path.with_name(stem + suffix)


def discover_recordings(root: Path) -> list[Recording]:
    rows = []
    for task, pattern in TASK_PATTERNS.items():
        for path in sorted(root.glob(f"sub-*/eeg/{pattern}")):
            match = re.search(r"sub-(\d+)", str(path))
            if match is None: continue
            event = _companion(path, "_events.tsv") if task == "ldaep_jittered_auditory" else None
            rows.append(Recording(f"sub-{int(match.group(1)):02d}", task, path,
                                  _companion(path, "_channels.tsv"), event))
    return rows


def read_channel_status(path: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    table = pd.read_csv(path, sep="\t")
    names = tuple(table["name"].astype(str))
    status = table["status"].astype(str).str.casefold() if "status" in table else pd.Series("good", index=table.index)
    bad = tuple(table.loc[status == "bad", "name"].astype(str))
    return names, bad


def greedy_nonoverlapping(onsets: np.ndarray, duration: float, recording_duration: float, count: int) -> np.ndarray:
    selected: list[float] = []
    boundary = -np.inf
    for onset in np.sort(np.asarray(onsets, dtype=float)):
        if onset < 0 or onset < boundary or onset + duration > recording_duration: continue
        selected.append(float(onset)); boundary = float(onset + duration)
        if len(selected) == count: break
    if len(selected) < count:
        raise ValueError(f"needed {count} non-overlapping windows, found {len(selected)}")
    return np.asarray(selected)


def rest_windows(recording_duration: float, duration: float, count: int, edge_seconds: float = 5.0) -> np.ndarray:
    starts = edge_seconds + np.arange(count, dtype=float) * duration
    if starts[-1] + duration > recording_duration - edge_seconds:
        raise ValueError(f"recording duration {recording_duration:g}s cannot provide {count} rest windows")
    return starts


def group_condition_units(starts: np.ndarray, streams: int = 3, trials: int = 10) -> np.ndarray:
    values = np.asarray(starts, dtype=float)
    per_unit = streams * trials
    if values.ndim != 1 or len(values) % per_unit:
        raise ValueError("window starts must form complete condition-like units")
    # Consecutive triplets represent the three streams at each natural repetition.
    return values.reshape(-1, trials, streams).transpose(0, 2, 1)
