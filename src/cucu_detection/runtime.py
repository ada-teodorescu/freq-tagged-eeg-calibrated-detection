from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
import json
import os
import time
import traceback

from cucu_replication.runtime import (
    STAGE_STATES,
    atomic_save_npz,
    atomic_write_json,
    atomic_write_text,
    package_versions,
    utc_now,
)

from .config import DetectionConfig


def _input_manifest(config: DetectionConfig) -> list[tuple[str, int, int]]:
    folders = (
        config.data_root / "Matlab variables",
        config.data_root / "Pre-processed EEGLAB datasets",
        config.data_root / "Stimuli",
    )
    records: list[tuple[str, int, int]] = []
    for folder in folders:
        if not folder.exists():
            records.append((str(folder), -1, -1))
            continue
        for path in sorted(item for item in folder.iterdir() if item.is_file()):
            stat = path.stat()
            records.append((str(path.relative_to(config.project_root)), stat.st_size, stat.st_mtime_ns))
    return records


def analysis_hash(config: DetectionConfig) -> str:
    digest = sha256(config.config_path.read_bytes())
    source = config.project_root / "src" / "cucu_detection"
    for path in sorted(source.rglob("*.py")):
        digest.update(str(path.relative_to(config.project_root)).encode())
        digest.update(path.read_bytes())
    digest.update(json.dumps(_input_manifest(config), separators=(",", ":")).encode())
    return digest.hexdigest()[:12]


@dataclass
class RunContext:
    config: DetectionConfig
    analysis_id: str
    run_dir: Path
    state_path: Path
    workers: int = 1
    _last_progress_write: float = 0.0

    @classmethod
    def create(cls, config: DetectionConfig, workers: int | None = None) -> "RunContext":
        worker_count = config.workers if workers is None else int(workers)
        if not 1 <= worker_count <= 4:
            raise ValueError("workers must be in 1..4")
        identifier = analysis_hash(config)
        run_dir = config.output_root / "runs" / identifier
        run_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(run_dir / "matplotlib"))
        (run_dir / "matplotlib").mkdir(exist_ok=True)
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
            atomic_write_json(context.state_path, {
                "schema_version": 1, "analysis_id": identifier,
                "created_at": utc_now(), "updated_at": utc_now(),
                "config_path": str(config.config_path),
                "config_hash": sha256(config.config_path.read_bytes()).hexdigest(),
                "workers": worker_count, "versions": package_versions(), "stages": {},
            })
            atomic_write_text(run_dir / "config_snapshot.toml", config.config_path.read_text(encoding="utf-8"))
        atomic_write_json(latest_path, {
            "analysis_id": identifier, "run_dir": str(run_dir), "updated_at": utc_now(),
        })
        return context

    def _directory(self, name: str) -> Path:
        path = self.run_dir / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def checkpoint_dir(self) -> Path: return self._directory("checkpoints")
    @property
    def table_dir(self) -> Path: return self._directory("tables")
    @property
    def figure_dir(self) -> Path: return self._directory("figures")
    @property
    def log_dir(self) -> Path: return self._directory("logs")
    @property
    def cache_dir(self) -> Path: return self._directory("cache")

    def read_state(self) -> dict[str, Any]:
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def update_stage(self, stage: str, **changes: Any) -> None:
        state = self.read_state()
        record = state["stages"].setdefault(stage, {
            "status": "pending", "completed_units": 0, "total_units": 0,
            "current_unit": None, "outputs": [], "error": None,
        })
        if "status" in changes and changes["status"] not in STAGE_STATES:
            raise ValueError(f"Invalid stage state: {changes['status']}")
        record.update(changes)
        record["updated_at"] = utc_now()
        state["updated_at"] = utc_now()
        state["workers"] = self.workers
        atomic_write_json(self.state_path, state)

    def start_stage(self, stage: str, total: int) -> None:
        self.update_stage(stage, status="running", total_units=int(total), completed_units=0,
                          current_unit=None, started_at=utc_now(), finished_at=None, error=None)
        self.log(stage, f"started ({total} work units)")

    def progress(self, stage: str, completed: int, total: int, current: str | None = None) -> None:
        now = time.monotonic()
        if completed < total and now - self._last_progress_write < self.config.minimum_progress_interval_seconds:
            return
        self._last_progress_write = now
        self.update_stage(stage, completed_units=int(completed), total_units=int(total), current_unit=current)
        print(f"[{stage}] {completed}/{total} ({100 * completed / total if total else 100:5.1f}%) {current or ''}", flush=True)

    def finish_stage(self, stage: str, outputs: list[Path], summary: dict | None = None) -> None:
        total = int(self.read_state()["stages"].get(stage, {}).get("total_units", 0))
        self.update_stage(stage, status="completed", completed_units=total, current_unit=None,
                          outputs=[str(path) for path in outputs], summary=summary or {}, finished_at=utc_now())
        self.log(stage, "completed")

    def fail_stage(self, stage: str, error: BaseException) -> None:
        detail = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        self.update_stage(stage, status="failed", error=detail, finished_at=utc_now())
        self.log(stage, f"failed: {error}\n{detail}")

    def log(self, stage: str, message: str) -> None:
        with (self.log_dir / f"{stage}.log").open("a", encoding="utf-8") as handle:
            handle.write(f"{utc_now()} {message}\n")

    def stage_completed(self, stage: str) -> bool:
        return self.read_state().get("stages", {}).get(stage, {}).get("status") == "completed"

    def checkpoint_valid(self, path: Path) -> bool:
        sidecar = path.with_suffix(path.suffix + ".meta.json")
        if not path.exists() or not sidecar.exists(): return False
        try: metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError): return False
        return metadata.get("analysis_id") == self.analysis_id

    def mark_checkpoint(self, path: Path, metadata: dict | None = None) -> None:
        atomic_write_json(path.with_suffix(path.suffix + ".meta.json"), {
            "analysis_id": self.analysis_id, "created_at": utc_now(), **(metadata or {}),
        })


__all__ = ["RunContext", "analysis_hash", "atomic_save_npz", "atomic_write_json", "atomic_write_text"]
