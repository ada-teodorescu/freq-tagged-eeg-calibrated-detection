from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from importlib import metadata
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor
import json
import os
import platform
import time
import traceback

from .config import ReplicationConfig


STAGE_STATES = {"pending", "running", "completed", "failed", "blocked", "stale"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    _replace_with_retry(temporary, path)


def _replace_with_retry(temporary: Path, path: Path) -> None:
    """Atomically publish a validated temporary file, tolerating brief Windows locks."""
    for attempt in range(10):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.05 * (attempt + 1))


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic_save_npz(path: Path, **arrays: Any) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    with np.load(temporary, allow_pickle=False) as archive:
        if set(archive.files) != set(arrays):
            raise OSError(f"Incomplete NPZ checkpoint: {temporary}")
        for name, value in arrays.items():
            if archive[name].shape != np.asarray(value).shape:
                raise OSError(f"Invalid shape for {name!r} in {temporary}")
    _replace_with_retry(temporary, path)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _source_files(config: ReplicationConfig) -> list[Path]:
    source = config.project_root / "src" / "eeg_replication"
    return sorted(source.rglob("*.py")) if source.exists() else []


def _fast_input_manifest(config: ReplicationConfig) -> list[tuple[str, int, int]]:
    folders = [
        config.data_root / "Matlab variables",
        config.data_root / "Pre-processed EEGLAB datasets",
        config.data_root / "Stimuli",
        config.original_repo_root / "Data files",
        config.original_repo_root / "Edge markers",
        config.original_repo_root / "EEG_sound analyses",
        config.original_repo_root / "R Stats",
    ]
    records = []
    for folder in folders:
        if not folder.exists():
            records.append((str(folder.relative_to(config.project_root)), -1, -1))
            continue
        for path in sorted(p for p in folder.iterdir() if p.is_file()):
            stat = path.stat()
            records.append((
                str(path.relative_to(config.project_root)).replace("\\", "/"),
                int(stat.st_size),
                int(stat.st_mtime_ns),
            ))
    return records


def analysis_hash(config: ReplicationConfig) -> str:
    digest = sha256()
    digest.update(config.config_path.read_bytes())
    for path in _source_files(config):
        digest.update(str(path.relative_to(config.project_root)).encode())
        digest.update(path.read_bytes())
    digest.update(json.dumps(_fast_input_manifest(config), separators=(",", ":")).encode())
    return digest.hexdigest()[:12]


def package_versions() -> dict[str, str]:
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("numpy", "scipy", "pandas", "matplotlib", "rich"):
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def ordered_parallel_map(function, items, workers: int):
    """Map independent work units deterministically, using bounded threads when requested."""
    values = list(items)
    if workers == 1:
        return [function(value) for value in values]
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="eeg-replication") as executor:
        return list(executor.map(function, values))


@dataclass
class RunContext:
    config: ReplicationConfig
    analysis_id: str
    run_dir: Path
    state_path: Path
    workers: int = 1
    _last_progress_write: float = 0.0

    @classmethod
    def create(cls, config: ReplicationConfig, workers: int | None = None) -> "RunContext":
        worker_count = config.workers if workers is None else int(workers)
        if not 1 <= worker_count <= 4:
            raise ValueError("workers must be in 1..4")
        identifier = analysis_hash(config)
        run_dir = config.output_root / "runs" / identifier
        run_dir.mkdir(parents=True, exist_ok=True)
        matplotlib_dir = run_dir / "matplotlib"
        matplotlib_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_dir))
        latest_path = config.output_root / "latest.json"
        if latest_path.exists():
            try:
                previous = json.loads(latest_path.read_text(encoding="utf-8"))
                previous_state_path = Path(previous["run_dir"]) / "state.json"
                if previous.get("analysis_id") != identifier and previous_state_path.exists():
                    previous_state = json.loads(previous_state_path.read_text(encoding="utf-8"))
                    for record in previous_state.get("stages", {}).values():
                        if record.get("status") == "completed":
                            record["status"] = "stale"
                            record["stale_because"] = f"superseded by analysis {identifier}"
                    previous_state["updated_at"] = utc_now()
                    atomic_write_json(previous_state_path, previous_state)
            except (KeyError, json.JSONDecodeError, OSError):
                pass
        context = cls(config, identifier, run_dir, run_dir / "state.json", worker_count)
        if not context.state_path.exists():
            state = {
                "schema_version": 1,
                "analysis_id": identifier,
                "created_at": utc_now(),
                "updated_at": utc_now(),
                "config_path": str(config.config_path),
                "config_hash": sha256(config.config_path.read_bytes()).hexdigest(),
                "workers": worker_count,
                "versions": package_versions(),
                "stages": {},
            }
            atomic_write_json(context.state_path, state)
            atomic_write_text(run_dir / "config_snapshot.toml", config.config_path.read_text(encoding="utf-8"))
        atomic_write_json(latest_path, {
            "analysis_id": identifier,
            "run_dir": str(run_dir),
            "updated_at": utc_now(),
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
            raise ValueError(f"Invalid stage status: {changes['status']}")
        record.update(changes)
        record["updated_at"] = utc_now()
        state["updated_at"] = utc_now()
        state["workers"] = self.workers
        atomic_write_json(self.state_path, state)

    def start_stage(self, stage: str, total_units: int) -> None:
        self.update_stage(
            stage, status="running", total_units=int(total_units), completed_units=0,
            current_unit=None, started_at=utc_now(), finished_at=None, error=None,
        )
        self.log(stage, f"started ({total_units} work units)")

    def progress(self, stage: str, completed: int, total: int, current: str | None = None) -> None:
        now = time.monotonic()
        if completed < total and now - self._last_progress_write < self.config.minimum_progress_interval_seconds:
            return
        self._last_progress_write = now
        self.update_stage(
            stage, completed_units=int(completed), total_units=int(total), current_unit=current,
        )
        percent = 100.0 * completed / total if total else 100.0
        print(f"[{stage}] {completed}/{total} ({percent:5.1f}%) {current or ''}", flush=True)

    def finish_stage(self, stage: str, outputs: list[Path], summary: dict[str, Any] | None = None) -> None:
        state = self.read_state()
        record = state["stages"].get(stage, {})
        total = int(record.get("total_units", 0))
        self.update_stage(
            stage, status="completed", completed_units=total, current_unit=None,
            outputs=[str(path) for path in outputs], summary=summary or {}, finished_at=utc_now(),
        )
        self.log(stage, "completed")

    def fail_stage(self, stage: str, error: BaseException) -> None:
        detail = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        self.update_stage(stage, status="failed", error=detail, finished_at=utc_now())
        self.log(stage, f"failed: {error}\n{detail}")

    def log(self, stage: str, message: str) -> None:
        path = self.log_dir / f"{stage}.log"
        line = f"{utc_now()} {message}\n"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)

    def stage_completed(self, stage: str) -> bool:
        return self.read_state().get("stages", {}).get(stage, {}).get("status") == "completed"

    def checkpoint_valid(self, path: Path) -> bool:
        if not path.exists():
            return False
        sidecar = path.with_suffix(path.suffix + ".meta.json")
        if not sidecar.exists():
            return False
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False
        return meta.get("analysis_id") == self.analysis_id

    def mark_checkpoint(self, path: Path, metadata_value: dict[str, Any] | None = None) -> None:
        atomic_write_json(path.with_suffix(path.suffix + ".meta.json"), {
            "analysis_id": self.analysis_id,
            "created_at": utc_now(),
            **(metadata_value or {}),
        })
