from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
from scipy.io import whosmat

from ..io import discover_epoch_files, validate_epoch_file
from ..runtime import RunContext, atomic_write_json, package_versions, sha256_file


def _input_files(context: RunContext) -> list[Path]:
    config = context.config
    folders = (
        config.data_root / "Matlab variables",
        config.data_root / "Pre-processed EEGLAB datasets",
        config.data_root / "Stimuli",
        config.original_repo_root / "Data files",
        config.original_repo_root / "Edge markers",
        config.original_repo_root / "EEG_sound analyses",
        config.original_repo_root / "R Stats",
    )
    return sorted(
        (path for folder in folders if folder.exists() for path in folder.iterdir() if path.is_file()),
        key=lambda path: str(path).lower(),
    )


def run(context: RunContext, force: bool = False) -> tuple[list[Path], dict]:
    stage = "00_inventory"
    config = context.config
    files = _input_files(context)
    epochs = discover_epoch_files(config)
    total = len(files) + len(epochs)
    context.start_stage(stage, total)
    errors = []
    expected_paths = [
        config.data_root / "Matlab variables" / name
        for name in ("Data_fft_all.mat", "Itc_per_stream.mat", "Itc_for_stats.mat")
    ]
    expected_paths.extend([
        config.original_repo_root / "Data files" / "Itc_for_pca_oana.mat",
        config.original_repo_root / "Data files" / "new_itc_pca1_for_stats.mat",
        config.original_repo_root / "Data files" / "new_itc_pca2_for_stats.mat",
        config.original_repo_root / "Data files" / "Sound_properties_may21.mat",
        config.original_repo_root / "Data files" / "Zscores_sound_prop_pca_may21.mat",
        config.original_repo_root / "EEG_sound analyses" / "corr_pd_itc4_june21.m",
        config.original_repo_root / "EEG_sound analyses" / "sort_data_from_eeglab.m",
    ])
    errors.extend(f"Missing required input: {path}" for path in expected_paths if not path.exists())
    set_count = len(list((config.data_root / "Pre-processed EEGLAB datasets").glob("*.set")))
    fdt_count = len(list((config.data_root / "Pre-processed EEGLAB datasets").glob("*.fdt")))
    wavs = list((config.data_root / "Stimuli").glob("*.wav"))
    if set_count != config.expected_participants:
        errors.append(f"SET count={set_count}, expected {config.expected_participants}")
    if fdt_count != config.expected_participants:
        errors.append(f"FDT count={fdt_count}, expected {config.expected_participants}")
    if len(wavs) != config.expected_all_wavs:
        errors.append(f"WAV count={len(wavs)}, expected {config.expected_all_wavs}")
    for stream_id in range(1, config.expected_target_wavs + 1):
        if not (config.data_root / "Stimuli" / f"{stream_id}.wav").exists():
            errors.append(f"Missing target WAV {stream_id}.wav")

    checkpoint = context.checkpoint_dir / stage / "checksums.json"
    checksum_records = {}
    if checkpoint.exists() and not force:
        try:
            checksum_records = json.loads(checkpoint.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            checksum_records = {}
    file_rows = []
    completed = 0
    for path in files:
        relative = str(path.relative_to(config.project_root)).replace("\\", "/")
        stat = path.stat()
        cached = checksum_records.get(relative, {})
        if force or cached.get("size") != stat.st_size or cached.get("mtime_ns") != stat.st_mtime_ns:
            digest = sha256_file(path)
            cached = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest}
            checksum_records[relative] = cached
            atomic_write_json(checkpoint, checksum_records)
        file_rows.append({"path": relative, **cached})
        completed += 1
        context.progress(stage, completed, total, relative)

    eeg_rows = []
    for epoch in epochs:
        errors.extend(validate_epoch_file(epoch, config))
        eeg_rows.append({
            "participant": epoch.participant,
            "set_path": str(epoch.set_path),
            "fdt_path": str(epoch.fdt_path),
            "channels": epoch.n_channels,
            "samples": epoch.n_times,
            "trials": epoch.n_trials,
            "sampling_rate": epoch.sampling_rate,
            "event_count": epoch.event_count,
            "channel_labels": "|".join(epoch.channel_labels),
            "has_ica_weights": epoch.has_ica_weights,
            "ica_components": epoch.ica_components,
        })
        completed += 1
        context.progress(stage, completed, total, epoch.participant)

    matlab = {}
    matlab_folder = config.data_root / "Matlab variables"
    for path in sorted(matlab_folder.glob("*.mat")):
        matlab[path.name] = [
            {"name": name, "shape": list(shape), "class": data_class}
            for name, shape, data_class in whosmat(path)
        ]
    original_data = config.original_repo_root / "Data files"
    for path in sorted(original_data.glob("*.mat")):
        matlab[f"original/{path.name}"] = [
            {"name": name, "shape": list(shape), "class": data_class}
            for name, shape, data_class in whosmat(path)
        ]
    manifest_path = context.table_dir / "input_manifest.csv"
    eeg_path = context.table_dir / "eeg_inventory.csv"
    report_path = context.run_dir / "inventory.json"
    pd.DataFrame(file_rows).to_csv(manifest_path, index=False)
    pd.DataFrame(eeg_rows).to_csv(eeg_path, index=False)
    report = {
        "passed": not errors,
        "errors": errors,
        "counts": {
            "participants": len(epochs), "set_files": set_count, "fdt_files": fdt_count,
            "wav_files": len(wavs), "target_wavs": config.expected_target_wavs,
        },
        "matlab_variables": matlab,
        "versions": package_versions(),
        "source_labels": ["stored_matlab", "original_repository", "public_set_fdt"],
        "ica_summary": {
            "participants_with_ica_weights": int(sum(row["has_ica_weights"] for row in eeg_rows)),
            "script_variant": "pow_itc_e2_no_ica.m",
            "interpretation": "ICA metadata is diagnostic only; it does not prove which preprocessing variant generated stored FFT products.",
        },
    }
    atomic_write_json(report_path, report)
    if errors:
        raise ValueError("Inventory validation failed:\n- " + "\n- ".join(errors))
    outputs = [manifest_path, eeg_path, report_path]
    context.finish_stage(stage, outputs, {"passed": True, **report["counts"]})
    return outputs, report
