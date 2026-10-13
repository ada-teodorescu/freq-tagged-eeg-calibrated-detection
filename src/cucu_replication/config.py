from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


CONDITIONS = (
    "Vowels", "/b/", "/d/", "/g/", "/k/", "/p/", "/t/", "/m/",
    "/n/", "/s/", "/z/", "/l/", "/r/", "/f/", "/v/",
)


@dataclass(frozen=True)
class ReplicationConfig:
    config_path: Path
    project_root: Path
    data_root: Path
    original_repo_root: Path
    output_root: Path
    expected_participants: int
    expected_channels: int
    expected_streams: int
    expected_target_wavs: int
    expected_all_wavs: int
    trials_per_stream: int
    samples_per_epoch: int
    sampling_rate: int
    seed: int
    workers: int
    harmonics_hz: tuple[float, ...]
    crop_start_seconds: float
    crop_end_seconds: float
    exact_tolerance: float
    bootstrap_iterations: int
    permutation_iterations: int
    hilbert_filter_order: int
    hilbert_half_bandwidth_hz: float
    null_orders: tuple[int, ...]
    null_half_bandwidths_hz: tuple[float, ...]
    null_edge_crops_seconds: tuple[float, ...]
    null_batches: int
    null_sampling_rate: int
    expected_landmarks: int
    candidate_min_distance_ms: int
    ioi_target_ms: int
    ioi_soft_scale_ms: int
    envelope_lowpass_hz: float
    minimum_progress_interval_seconds: float


def load_config(path: str | Path = "replication_config.toml") -> ReplicationConfig:
    config_path = Path(path).resolve()
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    root = config_path.parent
    paths = raw["paths"]
    expected = raw["expected"]
    analysis = raw["analysis"]
    hilbert = raw["hilbert"]
    null = raw["filter_null"]
    acoustics = raw["acoustics"]
    runtime = raw["runtime"]
    workers = int(analysis.get("workers", 1))
    if not 1 <= workers <= 4:
        raise ValueError("analysis.workers must be in 1..4")
    return ReplicationConfig(
        config_path=config_path,
        project_root=root,
        data_root=(root / paths["data_root"]).resolve(),
        original_repo_root=(root / paths.get("original_repo_root", "cv_syllables-master")).resolve(),
        output_root=(root / paths["output_root"]).resolve(),
        expected_participants=int(expected["participants"]),
        expected_channels=int(expected["channels"]),
        expected_streams=int(expected["streams"]),
        expected_target_wavs=int(expected["target_wavs"]),
        expected_all_wavs=int(expected["all_wavs"]),
        trials_per_stream=int(expected["trials_per_stream"]),
        samples_per_epoch=int(expected["samples_per_epoch"]),
        sampling_rate=int(expected["sampling_rate"]),
        seed=int(analysis["seed"]),
        workers=workers,
        harmonics_hz=tuple(float(x) for x in analysis["harmonics_hz"]),
        crop_start_seconds=float(analysis["crop_start_seconds"]),
        crop_end_seconds=float(analysis["crop_end_seconds"]),
        exact_tolerance=float(analysis["exact_tolerance"]),
        bootstrap_iterations=int(analysis["bootstrap_iterations"]),
        permutation_iterations=int(analysis["permutation_iterations"]),
        hilbert_filter_order=int(hilbert["filter_order"]),
        hilbert_half_bandwidth_hz=float(hilbert["half_bandwidth_hz"]),
        null_orders=tuple(int(x) for x in null["orders"]),
        null_half_bandwidths_hz=tuple(float(x) for x in null["half_bandwidths_hz"]),
        null_edge_crops_seconds=tuple(float(x) for x in null["edge_crops_seconds"]),
        null_batches=int(null["randomization_batches"]),
        null_sampling_rate=int(null["sampling_rate"]),
        expected_landmarks=int(acoustics["expected_landmarks"]),
        candidate_min_distance_ms=int(acoustics["candidate_min_distance_ms"]),
        ioi_target_ms=int(acoustics["ioi_target_ms"]),
        ioi_soft_scale_ms=int(acoustics["ioi_soft_scale_ms"]),
        envelope_lowpass_hz=float(acoustics["envelope_lowpass_hz"]),
        minimum_progress_interval_seconds=float(runtime["minimum_progress_interval_seconds"]),
    )


def condition_for_stream(stream_id: int) -> tuple[int, str]:
    if not 1 <= stream_id <= 45:
        raise ValueError(f"stream_id must be in 1..45, got {stream_id}")
    condition_id = (stream_id - 1) // 3 + 1
    return condition_id, CONDITIONS[condition_id - 1]
