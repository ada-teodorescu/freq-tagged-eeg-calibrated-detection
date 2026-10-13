from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pandas as pd

from eeg_core.runtime import atomic_write_json
from ..dataset import discover_recordings, read_channel_status
from ..runtime import RunContext


def _quick_fingerprint(path: Path) -> dict:
    stat = path.stat(); digest = sha256()
    with path.open("rb") as handle: digest.update(handle.read(1024 * 1024))
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "first_megabyte_sha256": digest.hexdigest()}


def run(context: RunContext, force: bool = False) -> tuple[list[Path], dict]:
    del force
    import mne
    stage, config = "01_inventory", context.config
    records = discover_recordings(config.external_data_root)
    if not records: raise FileNotFoundError(f"no ds007615 BrainVision recordings under {config.external_data_root}")
    context.start_stage(stage, len(records))
    rows, fingerprints = [], []
    for index, record in enumerate(records, 1):
        required = [record.vhdr_path, record.channel_path] + ([record.event_path] if record.event_path else [])
        missing = [str(path) for path in required if path is not None and not path.exists()]
        if missing: raise FileNotFoundError("incomplete BIDS recording: " + ", ".join(missing))
        raw = mne.io.read_raw_brainvision(record.vhdr_path, preload=False, verbose="ERROR")
        names, bad = read_channel_status(record.channel_path)
        rows.append({"participant": record.participant, "task": record.task,
                     "vhdr_path": str(record.vhdr_path), "channel_path": str(record.channel_path),
                     "event_path": str(record.event_path) if record.event_path else "",
                     "sampling_rate": float(raw.info["sfreq"]), "duration_seconds": float(raw.times[-1]),
                     "channels": len(names), "bad_channels": "|".join(bad)})
        fingerprints.extend(_quick_fingerprint(path) for path in required if path is not None)
        context.progress(stage, index, len(records), f"{record.participant} {record.task}")
    table_path = context.table_dir / "recording_inventory.csv"; pd.DataFrame(rows).to_csv(table_path, index=False)
    manifest_path = context.run_dir / "input_fingerprints.json"
    atomic_write_json(manifest_path, {"dataset_id": config.dataset_id, "dataset_version": config.dataset_version,
                                      "files": fingerprints})
    summary = {"recordings": len(rows), "participants": len(set(row["participant"] for row in rows)),
               "tasks": sorted(set(row["task"] for row in rows))}
    outputs = [table_path, manifest_path]; context.finish_stage(stage, outputs, summary); return outputs, summary
