from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class ValidationExtensionConfig:
    config_path: Path
    project_root: Path
    data_root: Path
    baseline_confirmatory_run: Path
    baseline_external_run: Path
    raw_events: Path
    external_data_root: Path
    output_root: Path
    export_root: Path
    seed: int
    workers: int
    sampling_rate: int
    trials_per_stream: int
    streams_per_condition: int
    conditions: int
    channels: int
    harmonics_hz: tuple[float, ...]
    k_values: tuple[int, ...]
    surrogate_seeds: tuple[int, ...]
    surrogates_per_seed: int
    window_start_seconds: float
    window_end_seconds: float
    hilbert_order: int
    hilbert_half_bandwidth_hz: float
    hilbert_edge_crop_seconds: float
    bandwidth_sensitivity_hz: tuple[float, ...]
    crop_sensitivity_seconds: tuple[float, ...]
    kernel_frequency_hz: float
    kernel_chunk_size: int
    locked_c: float
    maximum_iterations: int
    tolerance: float
    target_fpr: float
    fpr_acceptance_high: float
    simple_margin: float
    bootstrap_iterations: int
    sign_flip_iterations: int
    window_resamples: int
    one_sided_confidence: float
    shared_channels_excluded: tuple[str, ...]
    regions: tuple[tuple[str, ...], ...]
    region_names: tuple[str, ...]
    minimum_progress_interval_seconds: float
    minimum_free_disk_gb: float

    @property
    def streams(self) -> int:
        return self.conditions * self.streams_per_condition

    @property
    def surrogates(self) -> int:
        return len(self.surrogate_seeds) * self.surrogates_per_seed

    @property
    def target_channels(self) -> tuple[str, ...]:
        return tuple(channel for region in self.regions for channel in region)

    @property
    def shared_channels(self) -> tuple[str, ...]:
        excluded = set(self.shared_channels_excluded)
        return tuple(channel for channel in self.target_channels if channel not in excluded)


def load_config(path: str | Path = "validation_extension_config.toml") -> ValidationExtensionConfig:
    config_path = Path(path).resolve()
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    root = config_path.parent
    resolve = lambda value: (root / value).resolve()
    paths, analysis, estimators = raw["paths"], raw["analysis"], raw["estimators"]
    model, statistics, external = raw["model"], raw["statistics"], raw["external"]
    spatial, runtime = raw["spatial"], raw["runtime"]
    harmonics = tuple(float(value) for value in analysis["harmonics_hz"])
    k_values = tuple(int(value) for value in analysis["k_values"])
    trials = int(analysis["trials_per_stream"])
    regions = tuple(tuple(str(channel) for channel in group) for group in spatial["regions"])
    names = tuple(str(value) for value in spatial["region_names"])
    channels = tuple(channel for region in regions for channel in region)
    excluded = tuple(str(value) for value in external["shared_channels_excluded"])
    if harmonics != (4.0, 8.0, 12.0, 16.0):
        raise ValueError("the extension requires the locked four harmonics")
    if k_values != tuple(range(2, trials + 1)):
        raise ValueError("k_values must contain every prefix from 2 through 10")
    if len(channels) != int(analysis["channels"]) or len(set(channels)) != len(channels):
        raise ValueError("regions must partition the target channels")
    if len(regions) != len(names) or not set(excluded).issubset(channels):
        raise ValueError("invalid region names or shared-channel exclusion")
    workers = int(analysis.get("workers", 1))
    if not 1 <= workers <= 4:
        raise ValueError("workers must lie in 1..4")
    start, end = float(estimators["window_start_seconds"]), float(estimators["window_end_seconds"])
    if (start, end) != (0.5, 4.5):
        raise ValueError("the matched mechanism window is locked to [0.5, 4.5] seconds")
    return ValidationExtensionConfig(
        config_path=config_path, project_root=root, data_root=resolve(paths["data_root"]),
        baseline_confirmatory_run=resolve(paths["baseline_confirmatory_run"]),
        baseline_external_run=resolve(paths["baseline_external_run"]),
        raw_events=resolve(paths["raw_events"]), external_data_root=resolve(paths["external_data_root"]),
        output_root=resolve(paths["output_root"]), export_root=resolve(paths["export_root"]),
        seed=int(analysis["seed"]), workers=workers, sampling_rate=int(analysis["sampling_rate"]),
        trials_per_stream=trials, streams_per_condition=int(analysis["streams_per_condition"]),
        conditions=int(analysis["conditions"]), channels=int(analysis["channels"]), harmonics_hz=harmonics,
        k_values=k_values, surrogate_seeds=tuple(int(v) for v in analysis["surrogate_seeds"]),
        surrogates_per_seed=int(analysis["surrogates_per_seed"]), window_start_seconds=start,
        window_end_seconds=end, hilbert_order=int(estimators["hilbert_order"]),
        hilbert_half_bandwidth_hz=float(estimators["hilbert_half_bandwidth_hz"]),
        hilbert_edge_crop_seconds=float(estimators["hilbert_edge_crop_seconds"]),
        bandwidth_sensitivity_hz=tuple(float(v) for v in estimators["bandwidth_sensitivity_hz"]),
        crop_sensitivity_seconds=tuple(float(v) for v in estimators["crop_sensitivity_seconds"]),
        kernel_frequency_hz=float(estimators["kernel_frequency_hz"]),
        kernel_chunk_size=int(estimators["kernel_chunk_size"]), locked_c=float(model["locked_c"]),
        maximum_iterations=int(model["maximum_iterations"]), tolerance=float(model["tolerance"]),
        target_fpr=float(model["target_fpr"]), fpr_acceptance_high=float(model["fpr_acceptance_high"]),
        simple_margin=float(model["simple_margin"]), bootstrap_iterations=int(statistics["bootstrap_iterations"]),
        sign_flip_iterations=int(statistics["sign_flip_iterations"]),
        window_resamples=int(external["window_resamples"]),
        one_sided_confidence=float(external["one_sided_confidence"]), shared_channels_excluded=excluded,
        regions=regions, region_names=names,
        minimum_progress_interval_seconds=float(runtime["minimum_progress_interval_seconds"]),
        minimum_free_disk_gb=float(runtime["minimum_free_disk_gb"]),
    )
