from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class DetectionConfig:
    config_path: Path
    project_root: Path
    data_root: Path
    replication_output_root: Path
    output_root: Path
    seed: int
    workers: int
    sampling_rate: int
    trials_per_stream: int
    harmonics_hz: tuple[float, ...]
    k_values: tuple[int, ...]
    subset_cap: int
    neighbor_bins_per_harmonic: int
    target_fpr: float
    target_detection: float
    surrogate_batches: int
    bootstrap_iterations: int
    permutation_iterations: int
    estimators: tuple[str, ...]
    detectors: tuple[str, ...]
    spatial_methods: tuple[str, ...]
    primary_baseline: str
    primary_candidate: str
    fft_crop_samples: int
    hilbert_order: int
    hilbert_half_bandwidth_hz: float
    hilbert_edge_crop_seconds: float
    dpss_time_bandwidth: float
    dpss_tapers: int
    spatial_regularization: float
    cca_lags_ms: tuple[int, ...]
    minimum_progress_interval_seconds: float


def load_config(path: str | Path = "detection_config.toml") -> DetectionConfig:
    config_path = Path(path).resolve()
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    root = config_path.parent
    paths, analysis, methods = raw["paths"], raw["analysis"], raw["methods"]
    estimators, spatial, runtime = raw["estimators"], raw["spatial"], raw["runtime"]
    workers = int(analysis.get("workers", 1))
    if not 1 <= workers <= 4:
        raise ValueError("analysis.workers must be in 1..4")
    k_values = tuple(int(value) for value in analysis["k_values"])
    trials = int(analysis["trials_per_stream"])
    if not k_values or min(k_values) < 2 or max(k_values) > trials:
        raise ValueError("analysis.k_values must lie in 2..trials_per_stream")
    lag_start = int(spatial["cca_lag_start_ms"])
    lag_stop = int(spatial["cca_lag_stop_ms"])
    lag_step = int(spatial["cca_lag_step_ms"])
    return DetectionConfig(
        config_path=config_path,
        project_root=root,
        data_root=(root / paths["data_root"]).resolve(),
        replication_output_root=(root / paths["replication_output_root"]).resolve(),
        output_root=(root / paths["output_root"]).resolve(),
        seed=int(analysis["seed"]), workers=workers,
        sampling_rate=int(analysis["sampling_rate"]),
        trials_per_stream=trials,
        harmonics_hz=tuple(float(value) for value in analysis["harmonics_hz"]),
        k_values=k_values, subset_cap=int(analysis["subset_cap"]),
        neighbor_bins_per_harmonic=int(analysis["neighbor_bins_per_harmonic"]),
        target_fpr=float(analysis["target_fpr"]),
        target_detection=float(analysis["target_detection"]),
        surrogate_batches=int(analysis["surrogate_batches"]),
        bootstrap_iterations=int(analysis["bootstrap_iterations"]),
        permutation_iterations=int(analysis["permutation_iterations"]),
        estimators=tuple(methods["estimators"]), detectors=tuple(methods["detectors"]),
        spatial_methods=tuple(methods["spatial_methods"]),
        primary_baseline=str(methods["primary_baseline"]),
        primary_candidate=str(methods["primary_candidate"]),
        fft_crop_samples=int(estimators["fft_crop_samples"]),
        hilbert_order=int(estimators["hilbert_order"]),
        hilbert_half_bandwidth_hz=float(estimators["hilbert_half_bandwidth_hz"]),
        hilbert_edge_crop_seconds=float(estimators["hilbert_edge_crop_seconds"]),
        dpss_time_bandwidth=float(estimators["dpss_time_bandwidth"]),
        dpss_tapers=int(estimators["dpss_tapers"]),
        spatial_regularization=float(spatial["regularization"]),
        cca_lags_ms=tuple(range(lag_start, lag_stop + 1, lag_step)),
        minimum_progress_interval_seconds=float(runtime["minimum_progress_interval_seconds"]),
    )
