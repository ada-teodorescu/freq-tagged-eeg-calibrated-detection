from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

from ..events import discover_headers
from ..reference import discover_references
from ..runtime import RunContext, atomic_write_json, sha256_file


def _all_inputs(context: RunContext) -> list[tuple[str, Path]]:
    config = context.config
    headers = discover_headers(config.paths.raw_root)
    references = discover_references(config.paths.preprocessed_root)
    inputs: list[tuple[str, Path]] = []
    for participant in context.participants:
        header = headers[participant]
        inputs.extend((kind, path) for kind, path in (
            ("raw_header", header.vhdr_path), ("raw_marker", header.vmrk_path), ("raw_eeg", header.eeg_path),
        ))
        reference = references[participant]
        inputs.extend((("eeglab_set", reference.set_path), ("eeglab_fdt", reference.fdt_path)))
    for name in ("Data_fft_all.mat", "Itc_per_stream.mat", "Itc_for_stats.mat"):
        inputs.append(("released_matlab", config.paths.matlab_root / name))
    for name in (
        "Itc_for_pca_oana.mat", "new_itc_pca1_for_stats.mat",
        "new_itc_pca2_for_stats.mat", "Sound_properties_may21.mat",
        "Zscores_sound_prop_pca_may21.mat",
    ):
        inputs.append(("repository_product", config.paths.original_repo_root / "Data files" / name))
    inputs.extend(("paper", path) for path in config.paths.paper_paths)
    return inputs


def run(context: RunContext, force: bool = False) -> tuple[list, dict]:
    stage = "00_inventory"
    config = context.config
    headers = discover_headers(config.paths.raw_root)
    references = discover_references(config.paths.preprocessed_root)
    missing_raw = sorted(set(context.participants) - set(headers))
    missing_reference = sorted(set(context.participants) - set(references))
    if missing_raw or missing_reference:
        raise FileNotFoundError(f"Missing raw={missing_raw}, reference={missing_reference}")
    errors = []
    cohort_rows = []
    for participant in context.participants:
        header = headers[participant]
        reference = references[participant]
        if not header.eeg_path.exists() or not header.vmrk_path.exists():
            errors.append(f"{participant}: incomplete BrainVision triplet")
        if len(header.channels) != config.channels or header.sampling_rate != config.sampling_rate:
            errors.append(f"{participant}: raw structure differs from configuration")
        if header.binary_format.upper() != "INT_16":
            errors.append(f"{participant}: unexpected binary format {header.binary_format}")
        if (
            reference.n_channels != config.channels
            or reference.n_times != config.samples_per_epoch
            or reference.n_trials != config.expected_trials
            or round(reference.sampling_rate) != config.sampling_rate
        ):
            errors.append(f"{participant}: released SET/FDT structure differs from configuration")
        if tuple(name.casefold() for name in header.channels) != tuple(name.casefold() for name in reference.channels):
            errors.append(f"{participant}: raw and SET/FDT channel orders differ")
        cohort_rows.append({
            "participant": participant, "raw_header": str(header.vhdr_path),
            "reference_set": str(reference.set_path), "channels": len(header.channels),
            "sampling_rate": header.sampling_rate, "archive_defined_inclusion": True,
        })
    inputs = _all_inputs(context)
    missing_files = [str(path) for _, path in inputs if not path.exists()]
    if missing_files:
        errors.append(f"Missing required files: {missing_files}")
    if errors:
        raise ValueError("Inventory failed:\n" + "\n".join(errors))

    context.start_stage(stage, len(inputs))
    checkpoint = context.checkpoint_dir / stage / "checksums.json"
    records = None
    if not force and context.checkpoint_valid(checkpoint):
        records = json.loads(checkpoint.read_text(encoding="utf-8"))
    if records is None:
        records = []
        for index, (kind, path) in enumerate(inputs, start=1):
            records.append({
                "kind": kind,
                "path": str(path.relative_to(config.paths.project_root)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
            context.progress(stage, index, len(inputs), path.name)
        atomic_write_json(checkpoint, records)
        context.mark_checkpoint(checkpoint, {"record_count": len(records)})
    else:
        context.progress(stage, len(inputs), len(inputs), "cached checksums")

    other_raw = sorted(set(headers) - set(config.participants), key=lambda value: int(value[1:]))
    manifest_path = context.table_dir / "input_manifest.csv"
    cohort_path = context.table_dir / "cohort.csv"
    report_path = context.run_dir / "inventory.json"
    pd.DataFrame(records).to_csv(manifest_path, index=False)
    pd.DataFrame(cohort_rows).to_csv(cohort_path, index=False)
    report = {
        "passed": True, "participants": list(context.participants),
        "other_raw_participants": other_raw,
        "behavioral_exclusion_status": "cannot be determined from released decision records",
        "input_files": len(records), "input_bytes": sum(row["bytes"] for row in records),
    }
    atomic_write_json(report_path, report)
    outputs = [manifest_path, cohort_path, report_path]
    context.finish_stage(stage, outputs, report)
    return outputs, report
