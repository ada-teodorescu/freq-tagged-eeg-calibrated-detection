from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
import json
import os
import time
import traceback

from cucu_article.runtime import atomic_save_npz, atomic_write_json, atomic_write_text
from cucu_replication.runtime import package_versions, utc_now
from .config import AdvancedConfig


def analysis_hash(config: AdvancedConfig) -> str:
    digest = sha256(config.config_path.read_bytes())
    for path in sorted((config.project_root / "src" / "cucu_advanced_study").rglob("*.py")):
        digest.update(str(path.relative_to(config.project_root)).encode())
        digest.update(path.read_bytes())
    for path in sorted((config.data_root / "Pre-processed EEGLAB datasets").glob("*")):
        if path.is_file():
            stat = path.stat()
            digest.update(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    for baseline in (
        config.baseline_algorithm_run / "algorithm_conclusions.json",
        config.baseline_accuracy_run / "accuracy_conclusions.json",
    ):
        if baseline.exists():
            digest.update(baseline.read_bytes())
    if config.external_negative_root.exists():
        for path in sorted(config.external_negative_root.glob("*.npz")):
            stat = path.stat()
            digest.update(f"external:{path.name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()[:12]


@dataclass
class RunContext:
    config: AdvancedConfig
    analysis_id: str
    run_dir: Path
    state_path: Path
    workers: int
    _last_progress: float = 0.0

    @classmethod
    def create(cls, config: AdvancedConfig, workers: int | None = None) -> "RunContext":
        worker_count = config.workers if workers is None else int(workers)
        if not 1 <= worker_count <= 4:
            raise ValueError("workers must lie in 1..4")
        identifier = analysis_hash(config)
        run_dir = config.output_root / identifier
        run_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(run_dir / "matplotlib"))
        (run_dir / "matplotlib").mkdir(exist_ok=True)
        context = cls(config, identifier, run_dir, run_dir / "state.json", worker_count)
        if not context.state_path.exists():
            atomic_write_json(context.state_path, {
                "schema_version": 1, "analysis_id": identifier, "created_at": utc_now(),
                "updated_at": utc_now(), "config_path": str(config.config_path),
                "config_hash": sha256(config.config_path.read_bytes()).hexdigest(),
                "workers": worker_count, "versions": package_versions(), "stages": {},
            })
            atomic_write_text(run_dir / "config_snapshot.toml", config.config_path.read_text(encoding="utf-8"))
        atomic_write_json(config.output_root / "latest.json", {
            "analysis_id": identifier, "run_dir": str(run_dir), "updated_at": utc_now(),
        })
        return context

    def directory(self, name: str) -> Path:
        path = self.run_dir / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def checkpoint_dir(self) -> Path: return self.directory("checkpoints")
    @property
    def table_dir(self) -> Path: return self.directory("tables")
    @property
    def figure_dir(self) -> Path: return self.directory("figures")

    def read_state(self) -> dict[str, Any]:
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def update_stage(self, stage: str, **changes: Any) -> None:
        state = self.read_state()
        record = state["stages"].setdefault(stage, {
            "status": "pending", "completed_units": 0, "total_units": 0, "outputs": [], "error": None,
        })
        record.update(changes); record["updated_at"] = utc_now(); state["updated_at"] = utc_now()
        atomic_write_json(self.state_path, state)

    def start_stage(self, stage: str, total: int) -> None:
        self.update_stage(stage, status="running", completed_units=0, total_units=total,
                          started_at=utc_now(), finished_at=None, error=None)

    def progress(self, stage: str, completed: int, total: int, current: str = "") -> None:
        now = time.monotonic()
        if completed < total and now - self._last_progress < self.config.minimum_progress_interval_seconds:
            return
        self._last_progress = now
        self.update_stage(stage, completed_units=completed, total_units=total, current_unit=current)
        print(f"[{stage}] {completed}/{total} ({100 * completed / total:5.1f}%) {current}", flush=True)

    def finish_stage(self, stage: str, outputs: list[Path], summary: dict | None = None) -> None:
        total = self.read_state()["stages"][stage]["total_units"]
        self.update_stage(stage, status="completed", completed_units=total, current_unit=None,
                          outputs=[str(path) for path in outputs], summary=summary or {}, finished_at=utc_now())

    def fail_stage(self, stage: str, error: BaseException) -> None:
        self.update_stage(stage, status="failed",
                          error="".join(traceback.format_exception(type(error), error, error.__traceback__)),
                          finished_at=utc_now())

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
