from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np


PARTICIPANT_RE = re.compile(r"^(P\d+)\s+E2\s+Oana\.vhdr$", re.IGNORECASE)


@dataclass(frozen=True)
class BrainVisionHeader:
    participant: str
    vhdr_path: Path
    vmrk_path: Path
    eeg_path: Path
    channels: tuple[str, ...]
    sampling_rate: int
    binary_format: str


@dataclass(frozen=True)
class Marker:
    index: int
    marker_type: str
    description: str
    sample: int


def decode_target_description(description: object) -> int | None:
    """Decode Presentation's `1` prefix; practice/filler markers return None."""
    text = re.sub(r"\s+", "", str(description)).upper()
    if not text.startswith("S"):
        return None
    digits = text[1:]
    if len(digits) < 2 or not digits.startswith("1") or not digits.isdigit():
        return None
    stream = int(digits[1:])
    return stream if 1 <= stream <= 45 else None


def inspect_vhdr(path: Path) -> BrainVisionHeader:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    values: dict[str, str] = {}
    channels: list[tuple[int, str]] = []
    for line in text.splitlines():
        if "=" not in line or line.lstrip().startswith(";"):
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
        match = re.fullmatch(r"Ch(\d+)", key.strip(), re.IGNORECASE)
        if match:
            channels.append((int(match.group(1)), value.split(",", 1)[0].strip()))
    match = PARTICIPANT_RE.match(path.name)
    if match is None:
        raise ValueError(f"Unexpected raw header filename: {path.name}")
    interval_us = float(values["SamplingInterval"])
    sampling_rate = int(round(1_000_000.0 / interval_us))
    return BrainVisionHeader(
        participant=match.group(1).upper(),
        vhdr_path=path,
        vmrk_path=path.with_name(values["MarkerFile"]),
        eeg_path=path.with_name(values["DataFile"]),
        channels=tuple(name for _, name in sorted(channels)),
        sampling_rate=sampling_rate,
        binary_format=values.get("BinaryFormat", ""),
    )


def discover_headers(raw_root: Path) -> dict[str, BrainVisionHeader]:
    result: dict[str, BrainVisionHeader] = {}
    for path in raw_root.glob("*.vhdr"):
        header = inspect_vhdr(path)
        if header.participant in result:
            raise ValueError(f"Duplicate raw header for {header.participant}")
        result[header.participant] = header
    return result


def read_markers(path: Path) -> list[Marker]:
    markers: list[Marker] = []
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    for line in text.splitlines():
        match = re.match(r"Mk(\d+)=(.*)", line.strip(), re.IGNORECASE)
        if match is None:
            continue
        fields = match.group(2).split(",")
        if len(fields) < 3:
            raise ValueError(f"Malformed marker line in {path}: {line}")
        markers.append(Marker(
            index=int(match.group(1)),
            marker_type=fields[0].strip(),
            description=fields[1].strip(),
            sample=int(fields[2]),
        ))
    return markers


def target_events(path: Path) -> np.ndarray:
    rows = []
    for marker in read_markers(path):
        stream = decode_target_description(marker.description)
        if marker.marker_type.lower() == "stimulus" and stream is not None:
            rows.append((marker.sample - 1, 0, stream))
    return np.asarray(rows, dtype=np.int64).reshape(-1, 3)


def validate_events(events: np.ndarray, streams: int, trials_per_stream: int) -> dict[int, int]:
    events = np.asarray(events)
    expected = streams * trials_per_stream
    if events.shape != (expected, 3):
        raise ValueError(f"Expected {expected} target events, found {events.shape[0]}")
    if np.any(np.diff(events[:, 0]) <= 0):
        raise ValueError("Target events are not in strictly increasing sample order")
    counts = np.bincount(events[:, 2].astype(int), minlength=streams + 1)[1:]
    bad = {index + 1: int(value) for index, value in enumerate(counts) if value != trials_per_stream}
    if bad:
        raise ValueError(f"Invalid trials per stream: {bad}")
    return {index + 1: int(value) for index, value in enumerate(counts)}
