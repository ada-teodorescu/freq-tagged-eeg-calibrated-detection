from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from cucu_replication.io import inspect_set, natural_subject_key
from ..runtime import RunContext, atomic_write_json


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def run(context: RunContext, force: bool = False) -> tuple[list, dict]:
    stage = "00_inventory"; context.start_stage(stage, 3)
    baseline = context.config.baseline_detection_run
    state_path = baseline / "state.json"
    problems = []
    if not state_path.exists(): problems.append(f"missing baseline state: {state_path}")
    else:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("stages", {}).get("01_stored_coefficients", {}).get("status") not in {"completed", "stale"}:
            problems.append("baseline stored coefficient stage was not completed")
    caches = sorted((baseline / "cache" / "stored_matlab").glob("participant_*.npz"))
    if len(caches) != 25: problems.append(f"expected 25 baseline participant caches, found {len(caches)}")
    context.progress(stage, 1, 3, "immutable stored baseline")
    set_paths = sorted((context.config.data_root / "Pre-processed EEGLAB datasets").glob("*.set"), key=natural_subject_key)
    if len(set_paths) != 25: problems.append(f"expected 25 SET files, found {len(set_paths)}")
    metadata = []
    for path in set_paths:
        epoch = inspect_set(path)
        if not epoch.fdt_path.exists(): problems.append(f"missing FDT for {path.name}: {epoch.fdt_path}")
        metadata.append({"set": path.name, "fdt": epoch.fdt_path.name, "participant": epoch.participant})
    context.progress(stage, 2, 3, "public SET/FDT source")
    if problems: raise FileNotFoundError("\n".join(problems))
    report = {
        "baseline_run": str(baseline), "baseline_state_sha256": _sha256(state_path),
        "baseline_cache_count": len(caches), "public_participant_count": len(set_paths),
        "public_files": metadata, "stored_methods": list(context.config.stored_methods),
        "public_estimators": list(context.config.public_estimators),
        "spatial_methods": list(context.config.spatial_methods),
        "branches_numerically_merged": False,
    }
    output = context.run_dir / "inventory.json"; atomic_write_json(output, report)
    context.progress(stage, 3, 3, "inventory recorded"); context.finish_stage(stage, [output], report)
    return [output], report
