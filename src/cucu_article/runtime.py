from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
import json
import os
import time
import traceback
import math
import numpy as np

from cucu_replication.runtime import atomic_save_npz, atomic_write_json as _atomic_write_json, atomic_write_text, package_versions, utc_now
from .config import ArticleConfig


def json_ready(value: Any) -> Any:
    if isinstance(value, np.generic): return json_ready(value.item())
    if isinstance(value, Path): return str(value)
    if isinstance(value, float) and not math.isfinite(value): return None
    if isinstance(value, dict): return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [json_ready(item) for item in value]
    return value


def atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_json(path, json_ready(value))


def analysis_hash(config: ArticleConfig) -> str:
    digest = sha256(config.config_path.read_bytes())
    source = config.project_root / "src" / "cucu_article"
    for path in sorted(source.rglob("*.py")):
        digest.update(str(path.relative_to(config.project_root)).encode())
        digest.update(path.read_bytes())
    inputs = [
        config.replication_results / "tables" / "filter_null_summary.csv",
        config.detection_results / "benchmark_outcomes.json",
        config.improvement_results / "claim_status.json",
        config.raw_replication_run / "evidence.json",
    ]
    inputs.extend(sorted((config.data_root / "Pre-processed EEGLAB datasets").glob("*.set")))
    inputs.extend(sorted((config.data_root / "Pre-processed EEGLAB datasets").glob("*.fdt")))
    for path in inputs:
        if path.exists():
            stat = path.stat()
            digest.update(f"{path}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()[:12]


@dataclass
class RunContext:
    config: ArticleConfig
    analysis_id: str
    run_dir: Path
    state_path: Path
    workers: int = 1
    _last_progress_write: float = 0.0

    @classmethod
    def create(cls, config: ArticleConfig, workers: int | None = None) -> "RunContext":
        worker_count = config.workers if workers is None else int(workers)
        if not 1 <= worker_count <= 4:
            raise ValueError("workers must be in 1..4")
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

    def read_state(self) -> dict[str, Any]:
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def update_stage(self, stage: str, **changes: Any) -> None:
        state = self.read_state()
        record = state["stages"].setdefault(stage, {
            "status": "pending", "completed_units": 0, "total_units": 0,
            "current_unit": None, "outputs": [], "error": None,
        })
        record.update(changes)
        record["updated_at"] = utc_now()
        state["updated_at"] = utc_now()
        atomic_write_json(self.state_path, state)

    def start_stage(self, stage: str, total: int) -> None:
        self.update_stage(stage, status="running", total_units=total, completed_units=0,
                          current_unit=None, started_at=utc_now(), finished_at=None, error=None)

    def progress(self, stage: str, completed: int, total: int, current: str = "") -> None:
        now = time.monotonic()
        if completed < total and now - self._last_progress_write < self.config.minimum_progress_interval_seconds:
            return
        self._last_progress_write = now
        self.update_stage(stage, completed_units=completed, total_units=total, current_unit=current)
        print(f"[{stage}] {completed}/{total} ({100 * completed / total:5.1f}%) {current}", flush=True)

    def finish_stage(self, stage: str, outputs: list[Path], summary: dict | None = None) -> None:
        total = self.read_state()["stages"][stage]["total_units"]
        self.update_stage(stage, status="completed", completed_units=total, current_unit=None,
                          outputs=[str(path) for path in outputs], summary=summary or {}, finished_at=utc_now())

    def fail_stage(self, stage: str, error: BaseException) -> None:
        detail = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        self.update_stage(stage, status="failed", error=detail, finished_at=utc_now())

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


__all__ = ["RunContext", "analysis_hash", "atomic_save_npz", "atomic_write_json", "atomic_write_text", "json_ready"]
