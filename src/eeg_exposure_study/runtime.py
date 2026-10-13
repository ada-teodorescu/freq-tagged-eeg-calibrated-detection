"""Content-addressed runs and atomic, integrity-checked unit checkpoints."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json
import tomllib

from eeg_core.runtime import atomic_save_npz, atomic_write_json, atomic_write_text
from eeg_confirmatory_study.config import ConfirmatoryConfig, load_config
from eeg_replication.runtime import package_versions, utc_now


def fingerprint(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Config:
    path: Path
    root: Path
    source_run: Path
    output_root: Path
    source: ConfirmatoryConfig
    participants: tuple[str, ...]
    channel_labels: tuple[str, ...]
    seed: int
    bootstrap_iterations: int
    maximum_iterations: int
    tolerance: float

    def coefficient_path(self, participant: str) -> Path:
        return self.source_run / "checkpoints" / "01_coefficients" / f"{participant}.npz"


def read_config(path: str) -> Config:
    location = Path(path).resolve()
    raw = tomllib.loads(location.read_text(encoding="utf-8"))
    root = location.parent
    source_run = (root / raw["paths"]["source_run"]).resolve()
    output_root = (root / raw["paths"]["output_root"]).resolve()
    if output_root == source_run or source_run in output_root.parents:
        raise ValueError("Outputs must be separate from the source study")
    source = load_config(source_run / "config_snapshot.toml")
    lock = json.loads((source_run / "design_lock.json").read_text(encoding="utf-8"))
    participants = tuple(lock["participants"])
    if len(participants) < 3 or len(set(participants)) != len(participants):
        raise ValueError("At least three unique source participants are required")
    if any(Path(p).name != p or p in (".", "..") for p in participants):
        raise ValueError("Invalid participant identifier")
    if source.trials_per_stream != 10 or source.streams_per_condition != 3:
        raise ValueError("This analysis requires ten repetitions and three streams per condition")
    labels = tuple(lock["channel_labels"])
    if len(labels) != source.channels or lock["locked_c"] != source.locked_c:
        raise ValueError("Source lock and configuration disagree")
    if lock["matched_window_seconds"] != [0.5, 4.5]:
        raise ValueError("Expected the matched four-second source comparison")
    statistics, machine = raw["statistics"], raw["machine_learning"]
    config = Config(location, root, source_run, output_root, source, participants, labels,
                    int(statistics["seed"]), int(statistics["bootstrap_iterations"]),
                    int(machine["maximum_iterations"]), float(machine["tolerance"]))
    if (config.seed < 0 or config.bootstrap_iterations < 100 or
            config.maximum_iterations < 1 or not 0 < config.tolerance < 1):
        raise ValueError("Invalid seed, bootstrap count, or optimizer settings")
    for participant in participants:
        path = config.coefficient_path(participant)
        metadata = json.loads(path.with_suffix(".npz.meta.json").read_text(encoding="utf-8"))
        if metadata.get("analysis_id") != lock["analysis_id"] or metadata.get("natural_trial_order") is not True:
            raise ValueError(f"Unverified natural-order source checkpoint: {path}")
        if not path.is_file():
            raise FileNotFoundError(path)
    return config


class Run:
    def __init__(self, config: Config):
        self.config = config
        inputs = [config.path, config.source_run / "design_lock.json",
                  config.source_run / "config_snapshot.toml"]
        for participant in config.participants:
            path = config.coefficient_path(participant)
            inputs.extend((path, path.with_suffix(".npz.meta.json")))
        # Include shared helpers as well as this package to invalidate stale results.
        inputs.extend(sorted((config.root / "src").rglob("*.py")))
        self.fingerprints = {str(p.relative_to(config.root)) if p.is_relative_to(config.root)
                             else str(p): fingerprint(p) for p in inputs}
        self.versions = package_versions()
        identity = {"files": self.fingerprints, "versions": self.versions}
        self.identifier = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
        self.path = config.output_root / self.identifier

    def initialize(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.path / "input_fingerprints.json", self.fingerprints)
        atomic_write_json(self.path / "design.json", {
            "analysis_id": self.identifier, "status": "exploratory",
            "participants": self.config.participants, "halves": {"early": [1, 5], "late": [6, 10]},
            "unit": "participant-condition-half; three streams, five trials each",
            "model": "LOPO nonnegative logistic; pooled training halves; shared threshold per fold/method",
            "locked_c": self.config.source.locked_c, "target_fpr": self.config.source.target_fpr,
            "versions": self.versions, "source_run": str(self.config.source_run),
        })
        atomic_write_text(self.path / "config_snapshot.toml", self.config.path.read_text(encoding="utf-8"))
        atomic_write_json(self.config.output_root / "latest.json", {
            "analysis_id": self.identifier, "run_dir": str(self.path),
        })

    def checkpoint(self, stage: str, unit: str) -> Path:
        return self.path / "checkpoints" / stage / f"{unit}.npz"

    def valid(self, path: Path) -> bool:
        try:
            meta = json.loads(path.with_suffix(".npz.meta.json").read_text(encoding="utf-8"))
            return meta["analysis_id"] == self.identifier and meta["sha256"] == fingerprint(path)
        except (OSError, ValueError, KeyError):
            return False

    def save(self, path: Path, **arrays) -> None:
        atomic_save_npz(path, **arrays)
        atomic_write_json(path.with_suffix(".npz.meta.json"), {
            "analysis_id": self.identifier, "sha256": fingerprint(path), "created_at": utc_now(),
        })

    def progress(self, stage: str, completed: int, total: int, unit: str,
                 status: str = "running") -> None:
        atomic_write_json(self.path / "state.json", {
            "stage": stage, "status": status, "completed": completed,
            "total": total, "unit": unit, "updated_at": utc_now(),
        })
        print(f"[{stage}] {completed}/{total} {unit} ({status})", flush=True)
