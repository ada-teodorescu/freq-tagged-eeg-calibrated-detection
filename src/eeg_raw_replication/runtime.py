from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar
import json
import os
import platform
import shutil
import time
import traceback

import numpy as np

from .config import RawReplicationConfig


T = TypeVar("T")
R = TypeVar("R")
STAGE_STATES = {"pending", "running", "completed", "failed", "blocked", "stale"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _replace_with_retry(temporary: Path, destination: Path) -> None:
    for attempt in range(10):
        try:
            os.replace(temporary, destination)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.05 * (attempt + 1))


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    _replace_with_retry(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic_save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    with np.load(temporary, allow_pickle=False) as archive:
        if set(archive.files) != set(arrays):
            raise OSError(f"Incomplete NPZ artifact {temporary}")
    _replace_with_retry(temporary, path)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> dict[str, str]:
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("numpy", "scipy", "pandas", "matplotlib", "mne", "mne-icalabel", "onnxruntime"):
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def _source_files(config: RawReplicationConfig) -> list[Path]:
    root = config.paths.project_root / "src" / "eeg_raw_replication"
    return sorted(root.rglob("*.py")) if root.exists() else []


def source_hash(config: RawReplicationConfig) -> str:
    digest = sha256()
    for path in _source_files(config):
        digest.update(str(path.relative_to(config.paths.project_root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def input_manifest_hash(config: RawReplicationConfig, participants: tuple[str, ...]) -> str:
    encoded = json.dumps(_input_manifest(config, participants), separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def _input_manifest(config: RawReplicationConfig, participants: tuple[str, ...]) -> list[tuple[str, int, int]]:
    participant_names = set(participants)
    records = []
    folders = (config.paths.raw_root, config.paths.preprocessed_root, config.paths.matlab_root)
    for folder in folders:
        if not folder.exists():
            records.append((str(folder), -1, -1))
            continue
        for path in sorted(item for item in folder.iterdir() if item.is_file()):
            if folder in (config.paths.raw_root, config.paths.preprocessed_root):
                prefix = path.name.split(" ", 1)[0].upper()
                if prefix.startswith("P") and prefix not in participant_names:
                    continue
            stat = path.stat()
            records.append((
                str(path.relative_to(config.paths.project_root)).replace("\\", "/"),
                int(stat.st_size), int(stat.st_mtime_ns),
            ))
    return records


def analysis_hash(
    config: RawReplicationConfig,
    participants: tuple[str, ...],
    branches: tuple[str, ...],
) -> str:
    digest = sha256(config.config_path.read_bytes())
    digest.update(source_hash(config).encode())
    digest.update(input_manifest_hash(config, participants).encode())
    digest.update(json.dumps({"participants": participants, "branches": branches}, sort_keys=True).encode())
    return digest.hexdigest()[:12]


def ordered_parallel_map(function: Callable[[T], R], items: Iterable[T], workers: int) -> list[R]:
    values = list(items)
    if workers == 1:
        return [function(item) for item in values]
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="eeg-raw") as executor:
        return list(executor.map(function, values))


@dataclass
class RunContext:
    config: RawReplicationConfig
    participants: tuple[str, ...]
    branches: tuple[str, ...]
    analysis_id: str
    run_dir: Path
    state_path: Path
    workers: int
    _last_progress_write: float = 0.0

    @classmethod
    def create(
        cls,
        config: RawReplicationConfig,
        participants: tuple[str, ...] | None = None,
        branches: tuple[str, ...] | None = None,
        workers: int | None = None,
    ) -> "RunContext":
        selected_participants = participants or config.participants
        selected_branches = branches or config.enabled_branches
        worker_count = config.workers if workers is None else int(workers)
        if not 1 <= worker_count <= 4:
            raise ValueError("workers must be in 1..4")
        identifier = analysis_hash(config, selected_participants, selected_branches)
        run_dir = config.paths.output_root / identifier
        run_dir.mkdir(parents=True, exist_ok=True)
        matplotlib_dir = run_dir / "matplotlib"
        matplotlib_dir.mkdir(exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_dir))
        context = cls(
            config, selected_participants, selected_branches, identifier,
            run_dir, run_dir / "state.json", worker_count,
        )
        if not context.state_path.exists():
            atomic_write_json(context.state_path, {
                "schema_version": 1, "analysis_id": identifier,
                "created_at": utc_now(), "updated_at": utc_now(),
                "config_path": str(config.config_path),
                "config_hash": sha256(config.config_path.read_bytes()).hexdigest(),
                "source_hash": source_hash(config),
                "input_manifest_hash": input_manifest_hash(config, selected_participants),
                "participants": list(selected_participants), "branches": list(selected_branches),
                "workers": worker_count, "versions": package_versions(), "stages": {},
            })
            atomic_write_text(run_dir / "config_snapshot.toml", config.config_path.read_text(encoding="utf-8"))
        atomic_write_json(config.paths.output_root / "latest.json", {
            "analysis_id": identifier, "run_dir": str(run_dir), "updated_at": utc_now(),
        })
        return context

    @property
    def checkpoint_dir(self) -> Path:
        path = self.run_dir / "checkpoints"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def table_dir(self) -> Path:
        path = self.run_dir / "tables"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def figure_dir(self) -> Path:
        path = self.run_dir / "figures"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def log_dir(self) -> Path:
        path = self.run_dir / "logs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def read_state(self) -> dict[str, Any]:
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def update_stage(self, stage: str, **changes: Any) -> None:
        state = self.read_state()
        record = state["stages"].setdefault(stage, {
            "status": "pending", "completed_units": 0, "total_units": 0,
            "current_unit": None, "outputs": [], "error": None,
        })
        if "status" in changes and changes["status"] not in STAGE_STATES:
            raise ValueError(f"Invalid stage status {changes['status']!r}")
        record.update(changes)
        record["updated_at"] = utc_now()
        state["updated_at"] = utc_now()
        atomic_write_json(self.state_path, state)

    def start_stage(self, stage: str, total_units: int) -> None:
        self.update_stage(
            stage, status="running", total_units=total_units, completed_units=0,
            current_unit=None, started_at=utc_now(), finished_at=None, error=None,
        )

    def progress(self, stage: str, completed: int, total: int, current: str | None = None) -> None:
        now = time.monotonic()
        if completed < total and now - self._last_progress_write < self.config.minimum_progress_interval_seconds:
            return
        self._last_progress_write = now
        self.update_stage(stage, completed_units=completed, total_units=total, current_unit=current)
        print(f"[{stage}] {completed}/{total} {current or ''}", flush=True)

    def finish_stage(self, stage: str, outputs: list[Path], summary: dict[str, Any]) -> None:
        total = int(self.read_state()["stages"][stage].get("total_units", 0))
        self.update_stage(
            stage, status="completed", completed_units=total, current_unit=None,
            outputs=[str(path) for path in outputs], summary=summary, finished_at=utc_now(),
        )

    def fail_stage(self, stage: str, error: BaseException) -> None:
        detail = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        self.update_stage(stage, status="failed", error=detail, finished_at=utc_now())
        atomic_write_text(self.log_dir / f"{stage}.log", detail)

    def stage_completed(self, stage: str) -> bool:
        return self.read_state().get("stages", {}).get(stage, {}).get("status") == "completed"

    def mark_stale(self, stages: set[str], reason: str) -> None:
        state = self.read_state()
        for stage in stages:
            record = state["stages"].get(stage)
            if record is not None and record.get("status") == "completed":
                record["status"] = "stale"
                record["stale_because"] = reason
                record["updated_at"] = utc_now()
        state["updated_at"] = utc_now()
        atomic_write_json(self.state_path, state)

    def checkpoint_valid(self, path: Path) -> bool:
        sidecar = path.with_name(path.name + ".meta.json")
        if not path.exists() or not sidecar.exists():
            return False
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            return (
                meta.get("analysis_id") == self.analysis_id
                and meta.get("size") == path.stat().st_size
                and meta.get("sha256") == sha256_file(path)
            )
        except (OSError, json.JSONDecodeError):
            return False

    def mark_checkpoint(self, path: Path, details: dict[str, Any] | None = None) -> None:
        state = self.read_state()
        atomic_write_json(path.with_name(path.name + ".meta.json"), {
            "analysis_id": self.analysis_id, "created_at": utc_now(),
            "size": path.stat().st_size, "sha256": sha256_file(path),
            "config_hash": state["config_hash"], "source_hash": state["source_hash"],
            "input_manifest_hash": state["input_manifest_hash"],
            "participants": list(self.participants), "branches": list(self.branches),
            "versions": package_versions(), **(details or {}),
        })

    def checkpoint_bytes(self) -> int:
        return sum(path.stat().st_size for path in self.checkpoint_dir.rglob("*") if path.is_file())

    def export_compact_results(self) -> Path:
        destination = self.config.paths.export_root / self.analysis_id
        destination.mkdir(parents=True, exist_ok=True)
        allowed = [self.run_dir / "report.md", self.run_dir / "claims.json", self.table_dir, self.figure_dir]
        for source in allowed:
            if not source.exists():
                continue
            target = destination / source.name
            if source.is_dir():
                shutil.copytree(source, target, dirs_exist_ok=True)
            else:
                shutil.copy2(source, target)
        atomic_write_json(destination / "export.json", {
            "analysis_id": self.analysis_id, "source_run": str(self.run_dir), "exported_at": utc_now(),
        })
        return destination
