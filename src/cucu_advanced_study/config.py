from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class AdvancedConfig:
    config_path: Path
    project_root: Path
    data_root: Path
    baseline_algorithm_run: Path
    baseline_accuracy_run: Path
    output_root: Path
    export_root: Path
    external_negative_root: Path
    seed: int
    workers: int
    sampling_rate: int
    onset_crop_samples: int
    trials_per_stream: int
    streams_per_condition: int
    conditions: int
    channels: int
    harmonics_hz: tuple[float, ...]
    durations_seconds: tuple[float, ...]
    k_values: tuple[int, ...]
    primary_duration_seconds: float
    target_fpr: float
    target_detection: float
    surrogate_seeds: tuple[int, ...]
    surrogates_per_seed: int
    order_permutations: int
    off_target_bases_hz: tuple[float, ...]
    shrinkage_weights: tuple[float, ...]
    fpr_acceptance_low: float
    fpr_acceptance_high: float
    sensitivity_noninferiority_margin: float
    c_grid: tuple[float, ...]
    ridge_grid: tuple[float, ...]
    inner_group_folds: int
    maximum_iterations: int
    tolerance: float
    training_nulls_per_stream: int
    regions: tuple[tuple[str, ...], ...]
    region_names: tuple[str, ...]
    bootstrap_iterations: int
    sign_flip_iterations: int
    minimum_progress_interval_seconds: float
    minimum_free_disk_gb: float

    @property
    def streams(self) -> int:
        return self.conditions * self.streams_per_condition

    @property
    def participants_expected(self) -> int:
        return 25

    @property
    def surrogates(self) -> int:
        return len(self.surrogate_seeds) * self.surrogates_per_seed


def load_config(path: str | Path = "advanced_study_config.toml") -> AdvancedConfig:
    config_path = Path(path).resolve()
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    root = config_path.parent
    paths, analysis = raw["paths"], raw["analysis"]
    calibration, machine = raw["calibration"], raw["machine_learning"]
    spatial, statistics, runtime = raw["spatial"], raw["statistics"], raw["runtime"]
    resolve = lambda value: (root / value).resolve()
    harmonics = tuple(float(value) for value in analysis["harmonics_hz"])
    if harmonics != (4.0, 8.0, 12.0, 16.0):
        raise ValueError("advanced study requires locked 4, 8, 12, and 16 Hz harmonics")
    durations = tuple(float(value) for value in analysis["durations_seconds"])
    primary_duration = float(analysis["primary_duration_seconds"])
    if primary_duration not in durations or tuple(sorted(set(durations))) != durations:
        raise ValueError("durations must be unique, sorted, and include the primary duration")
    k_values = tuple(int(value) for value in analysis["k_values"])
    trials = int(analysis["trials_per_stream"])
    if k_values != tuple(range(2, trials + 1)):
        raise ValueError("sequential K values must be every prefix from 2 through trials_per_stream")
    seeds = tuple(int(value) for value in analysis["surrogate_seeds"])
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("surrogate seeds must be unique and non-empty")
    weights = tuple(float(value) for value in calibration["shrinkage_weights"])
    if weights != tuple(sorted(set(weights))) or weights[0] != 0 or weights[-1] != 1:
        raise ValueError("shrinkage weights must be sorted, unique, and span 0..1")
    target_fpr = float(analysis["target_fpr"])
    low, high = float(calibration["fpr_acceptance_low"]), float(calibration["fpr_acceptance_high"])
    if not 0 < low <= target_fpr <= high < 1:
        raise ValueError("invalid FPR target or acceptance interval")
    bases = tuple(float(value) for value in analysis["off_target_bases_hz"])
    tagged = set(harmonics + (20.0, 24.0))
    if not bases or any(base in tagged or 2 * base in tagged for base in bases):
        raise ValueError("off-target tag families overlap a known response harmonic")
    regions = tuple(tuple(str(channel) for channel in group) for group in spatial["regions"])
    names = tuple(str(value) for value in spatial["region_names"])
    flattened = [channel for region in regions for channel in region]
    channels = int(analysis["channels"])
    if len(regions) != len(names) or len(flattened) != channels or len(set(flattened)) != channels:
        raise ValueError("regions must partition all configured channels")
    workers = int(analysis.get("workers", 1))
    if not 1 <= workers <= 4:
        raise ValueError("workers must lie in 1..4")
    return AdvancedConfig(
        config_path=config_path, project_root=root, data_root=resolve(paths["data_root"]),
        baseline_algorithm_run=resolve(paths["baseline_algorithm_run"]),
        baseline_accuracy_run=resolve(paths["baseline_accuracy_run"]),
        output_root=resolve(paths["output_root"]), export_root=resolve(paths["export_root"]),
        external_negative_root=resolve(paths["external_negative_root"]), seed=int(analysis["seed"]),
        workers=workers, sampling_rate=int(analysis["sampling_rate"]),
        onset_crop_samples=int(analysis["onset_crop_samples"]), trials_per_stream=trials,
        streams_per_condition=int(analysis["streams_per_condition"]), conditions=int(analysis["conditions"]),
        channels=channels, harmonics_hz=harmonics, durations_seconds=durations, k_values=k_values,
        primary_duration_seconds=primary_duration, target_fpr=target_fpr,
        target_detection=float(analysis["target_detection"]), surrogate_seeds=seeds,
        surrogates_per_seed=int(analysis["surrogates_per_seed"]),
        order_permutations=int(analysis["order_permutations"]), off_target_bases_hz=bases,
        shrinkage_weights=weights, fpr_acceptance_low=low, fpr_acceptance_high=high,
        sensitivity_noninferiority_margin=float(calibration["sensitivity_noninferiority_margin"]),
        c_grid=tuple(float(value) for value in machine["c_grid"]),
        ridge_grid=tuple(float(value) for value in machine["ridge_grid"]),
        inner_group_folds=int(machine["inner_group_folds"]),
        maximum_iterations=int(machine["maximum_iterations"]), tolerance=float(machine["tolerance"]),
        training_nulls_per_stream=int(machine["training_nulls_per_stream"]),
        regions=regions, region_names=names, bootstrap_iterations=int(statistics["bootstrap_iterations"]),
        sign_flip_iterations=int(statistics["sign_flip_iterations"]),
        minimum_progress_interval_seconds=float(runtime["minimum_progress_interval_seconds"]),
        minimum_free_disk_gb=float(runtime["minimum_free_disk_gb"]),
    )
