from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class ExternalValidationConfig:
    config_path: Path
    project_root: Path
    external_data_root: Path
    baseline_confirmatory_run: Path
    output_root: Path
    export_root: Path
    dataset_id: str
    dataset_version: str
    primary_task: str
    secondary_tasks: tuple[str, ...]
    minimum_primary_participants: int
    minimum_secondary_participants: int
    seed: int
    workers: int
    sampling_rate: int
    epoch_seconds: float
    trials_per_stream: int
    streams_per_unit: int
    k_values: tuple[int, ...]
    harmonics_hz: tuple[float, ...]
    auditory_windows_per_participant: int
    rest_windows_per_participant: int
    l_freq: float
    h_freq: float
    average_reference: bool
    maximum_bad_target_channels: int
    structural_interpolation_targets: tuple[str, ...]
    peak_to_peak_sensitivity_uv: float
    locked_c: float
    maximum_iterations: int
    tolerance: float
    target_fpr: float
    bootstrap_iterations: int
    one_sided_confidence: float
    external_fpr_upper_limit: float
    regions: tuple[tuple[str, ...], ...]
    region_names: tuple[str, ...]
    minimum_progress_interval_seconds: float
    minimum_free_disk_gb: float

    @property
    def target_channels(self) -> tuple[str, ...]:
        return tuple(channel for region in self.regions for channel in region)

    @property
    def windows_per_unit(self) -> int:
        return self.trials_per_stream * self.streams_per_unit


def load_config(path: str | Path = "external_validation_config.toml") -> ExternalValidationConfig:
    config_path = Path(path).resolve()
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    root = config_path.parent
    resolve = lambda value: (root / value).resolve()
    paths, dataset, analysis = raw["paths"], raw["dataset"], raw["analysis"]
    preprocessing, model = raw["preprocessing"], raw["model"]
    validation, spatial, runtime = raw["validation"], raw["spatial"], raw["runtime"]
    trials = int(analysis["trials_per_stream"])
    streams = int(analysis["streams_per_unit"])
    k_values = tuple(int(value) for value in analysis["k_values"])
    harmonics = tuple(float(value) for value in analysis["harmonics_hz"])
    regions = tuple(tuple(str(channel) for channel in values) for values in spatial["regions"])
    names = tuple(str(value) for value in spatial["region_names"])
    channels = [channel for region in regions for channel in region]
    structural = tuple(str(value) for value in preprocessing["structural_interpolation_targets"])
    if k_values != tuple(range(2, trials + 1)):
        raise ValueError("k_values must contain every prefix from 2 through trials_per_stream")
    if harmonics != (4.0, 8.0, 12.0, 16.0):
        raise ValueError("external validation locks harmonics to 4, 8, 12, and 16 Hz")
    if len(regions) != len(names) or len(channels) != 32 or len(set(channels)) != 32:
        raise ValueError("regions must partition exactly 32 unique target channels")
    if not set(structural).issubset(channels):
        raise ValueError("structural interpolation targets must be target channels")
    if int(analysis["auditory_windows_per_participant"]) % (trials * streams):
        raise ValueError("auditory window count must form complete condition-like units")
    if int(analysis["rest_windows_per_participant"]) != trials * streams:
        raise ValueError("each rest state must form exactly one condition-like unit")
    if not 0 < float(validation["one_sided_confidence"]) < 1:
        raise ValueError("invalid one-sided confidence")
    workers = int(analysis.get("workers", 1))
    if not 1 <= workers <= 4:
        raise ValueError("workers must lie in 1..4")
    return ExternalValidationConfig(
        config_path=config_path, project_root=root,
        external_data_root=resolve(paths["external_data_root"]),
        baseline_confirmatory_run=resolve(paths["baseline_confirmatory_run"]),
        output_root=resolve(paths["output_root"]), export_root=resolve(paths["export_root"]),
        dataset_id=str(dataset["dataset_id"]), dataset_version=str(dataset["dataset_version"]),
        primary_task=str(dataset["primary_task"]),
        secondary_tasks=tuple(str(value) for value in dataset["secondary_tasks"]),
        minimum_primary_participants=int(dataset["minimum_primary_participants"]),
        minimum_secondary_participants=int(dataset["minimum_secondary_participants"]),
        seed=int(analysis["seed"]), workers=workers, sampling_rate=int(analysis["sampling_rate"]),
        epoch_seconds=float(analysis["epoch_seconds"]), trials_per_stream=trials, streams_per_unit=streams,
        k_values=k_values, harmonics_hz=harmonics,
        auditory_windows_per_participant=int(analysis["auditory_windows_per_participant"]),
        rest_windows_per_participant=int(analysis["rest_windows_per_participant"]),
        l_freq=float(preprocessing["l_freq"]), h_freq=float(preprocessing["h_freq"]),
        average_reference=bool(preprocessing["average_reference"]),
        maximum_bad_target_channels=int(preprocessing["maximum_bad_target_channels"]),
        structural_interpolation_targets=structural,
        peak_to_peak_sensitivity_uv=float(preprocessing["peak_to_peak_sensitivity_uv"]),
        locked_c=float(model["locked_c"]), maximum_iterations=int(model["maximum_iterations"]),
        tolerance=float(model["tolerance"]), target_fpr=float(model["target_fpr"]),
        bootstrap_iterations=int(validation["bootstrap_iterations"]),
        one_sided_confidence=float(validation["one_sided_confidence"]),
        external_fpr_upper_limit=float(validation["external_fpr_upper_limit"]),
        regions=regions, region_names=names,
        minimum_progress_interval_seconds=float(runtime["minimum_progress_interval_seconds"]),
        minimum_free_disk_gb=float(runtime["minimum_free_disk_gb"]),
    )
