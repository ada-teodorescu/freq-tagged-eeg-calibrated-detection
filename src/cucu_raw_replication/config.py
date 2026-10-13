from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


CONDITIONS = (
    "Vowels", "/b/", "/d/", "/g/", "/k/", "/p/", "/t/", "/m/",
    "/n/", "/s/", "/z/", "/l/", "/r/", "/f/", "/v/",
)
BRANCHES = ("paper12", "no_ica", "iclabel", "lowpass_no_ica")
BRANCH_ALIASES = {"primary": "paper12"}


@dataclass(frozen=True)
class PathConfig:
    project_root: Path
    data_root: Path
    raw_root: Path
    preprocessed_root: Path
    matlab_root: Path
    original_repo_root: Path
    paper_paths: tuple[Path, ...]
    output_root: Path
    export_root: Path


@dataclass(frozen=True)
class RawReplicationConfig:
    config_path: Path
    paths: PathConfig
    participants: tuple[str, ...]
    channels: int
    streams: int
    trials_per_stream: int
    samples_per_epoch: int
    sampling_rate: int
    l_freq: float
    h_freq: float
    montage: str
    filter_method: str
    filter_phase: str
    fir_window: str
    fir_design: str
    average_reference: bool
    ica_components: int
    ica_decimation: int
    frontal_channels: tuple[str, ...]
    frontal_power_threshold: float
    iclabel_eye_threshold: float
    seed: int
    harmonics_hz: tuple[float, ...]
    crop_start_seconds: float
    workers: int
    maximum_checkpoint_gb: float
    minimum_progress_interval_seconds: float
    enabled_branches: tuple[str, ...]
    match_min_r: float
    match_max_nrmse: float
    approximate_min_r: float
    approximate_max_nrmse: float

    @property
    def expected_trials(self) -> int:
        return self.streams * self.trials_per_stream


def _resolve(root: Path, value: str) -> Path:
    return (root / value).resolve()


def normalize_participant(value: str) -> str:
    value = value.strip().upper()
    if not value.startswith("P") or not value[1:].isdigit():
        raise ValueError(f"Invalid participant ID {value!r}")
    return f"P{int(value[1:])}"


def normalize_branch(value: str) -> str:
    result = BRANCH_ALIASES.get(value.strip().lower(), value.strip().lower())
    if result not in BRANCHES:
        raise ValueError(f"Unknown branch {value!r}; choices: primary, {', '.join(BRANCHES[1:])}")
    return result


def load_config(path: str | Path = "raw_replication_config.toml") -> RawReplicationConfig:
    config_path = Path(path).resolve()
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    root = config_path.parent
    path_values = raw["paths"]
    data_root = _resolve(root, path_values["data_root"])
    paths = PathConfig(
        project_root=root,
        data_root=data_root,
        raw_root=data_root / path_values.get("raw_folder", "Raw participant data"),
        preprocessed_root=data_root / path_values.get("preprocessed_folder", "Pre-processed EEGLAB datasets"),
        matlab_root=data_root / path_values.get("matlab_folder", "Matlab variables"),
        original_repo_root=_resolve(root, path_values.get("original_repo_root", "cv_syllables-master")),
        paper_paths=tuple(_resolve(root, item) for item in path_values.get("papers", [])),
        output_root=_resolve(root, path_values["output_root"]),
        export_root=_resolve(root, path_values["export_root"]),
    )
    expected = raw["expected"]
    preprocessing = raw["preprocessing"]
    ica = raw["ica"]
    analysis = raw["analysis"]
    runtime = raw["runtime"]
    agreement = raw["agreement"]
    participants = tuple(normalize_participant(item) for item in raw["cohort"]["participants"])
    if len(participants) != len(set(participants)):
        raise ValueError("cohort.participants contains duplicates")
    branches = tuple(normalize_branch(item) for item in raw["branches"]["enabled"])
    workers = int(runtime.get("workers", 1))
    if not 1 <= workers <= 4:
        raise ValueError("runtime.workers must be in 1..4")
    if "paper12" not in branches:
        raise ValueError("branches.enabled must include the primary paper12 branch")
    return RawReplicationConfig(
        config_path=config_path,
        paths=paths,
        participants=participants,
        channels=int(expected["channels"]),
        streams=int(expected["streams"]),
        trials_per_stream=int(expected["trials_per_stream"]),
        samples_per_epoch=int(expected["samples_per_epoch"]),
        sampling_rate=int(expected["sampling_rate"]),
        l_freq=float(preprocessing["l_freq"]),
        h_freq=float(preprocessing["h_freq"]),
        montage=str(preprocessing["montage"]),
        filter_method=str(preprocessing["filter_method"]),
        filter_phase=str(preprocessing["filter_phase"]),
        fir_window=str(preprocessing["fir_window"]),
        fir_design=str(preprocessing["fir_design"]),
        average_reference=bool(preprocessing["average_reference"]),
        ica_components=int(ica["components"]),
        ica_decimation=int(ica["decimation"]),
        frontal_channels=tuple(str(x) for x in ica["frontal_channels"]),
        frontal_power_threshold=float(ica["frontal_power_threshold"]),
        iclabel_eye_threshold=float(ica["iclabel_eye_threshold"]),
        seed=int(analysis["seed"]),
        harmonics_hz=tuple(float(x) for x in analysis["harmonics_hz"]),
        crop_start_seconds=float(analysis["crop_start_seconds"]),
        workers=workers,
        maximum_checkpoint_gb=float(runtime["maximum_checkpoint_gb"]),
        minimum_progress_interval_seconds=float(runtime["minimum_progress_interval_seconds"]),
        enabled_branches=branches,
        match_min_r=float(agreement["match_min_r"]),
        match_max_nrmse=float(agreement["match_max_nrmse"]),
        approximate_min_r=float(agreement["approximate_min_r"]),
        approximate_max_nrmse=float(agreement["approximate_max_nrmse"]),
    )


def condition_for_stream(stream_id: int) -> tuple[int, str]:
    if not 1 <= stream_id <= 45:
        raise ValueError(f"stream_id must be in 1..45, got {stream_id}")
    condition_id = (stream_id - 1) // 3 + 1
    return condition_id, CONDITIONS[condition_id - 1]
