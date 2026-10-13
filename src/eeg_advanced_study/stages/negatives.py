from __future__ import annotations

from pathlib import Path
import json
import re

import numpy as np
import pandas as pd

from eeg_core.runtime import atomic_write_json

from ..algorithms import first_crossings, fixed_stream_scores, phase_trajectories
from ..runtime import RunContext
from .features import feature_path
from .stream import METHODS


def _population_boundaries(context: RunContext) -> dict[tuple[str, str], np.ndarray]:
    config = context.config
    table = pd.read_csv(context.table_dir / "sequential_boundaries.csv")
    selected = table[(table["calibration"] == "population") &
                     np.isclose(table["duration_seconds"], config.primary_duration_seconds)]
    result = {}
    for (participant, method), group in selected.groupby(["participant", "method"]):
        result[(str(participant), str(method))] = group.sort_values("look_k")["boundary"].to_numpy(float)
    return result


def _rows_for_units(participant: str, family: str, kind: str, coefficients: np.ndarray,
                    boundaries: dict, k_values: tuple[int, ...]) -> list[dict]:
    global_r2, _ = phase_trajectories(coefficients, k_values)
    scores = fixed_stream_scores(global_r2, k_values)
    rows = []
    for method_index, method in enumerate(METHODS[:2]):
        limits = boundaries[(participant, method)]
        first = first_crossings(scores[..., method_index], limits, k_values)
        for unit in range(coefficients.shape[0]):
            rows.append({"participant": participant, "control_kind": kind, "family": family,
                         "unit": unit + 1, "method": method, "first_detection_k": int(first[unit]),
                         "false_positive": bool(first[unit] <= k_values[-1])})
    return rows


def _validate_external(path: Path, config, channel_labels: tuple[str, ...]) -> tuple[str, np.ndarray, str]:
    match = re.search(r"P(\d+)", path.stem, re.IGNORECASE)
    if match is None:
        raise ValueError(f"external negative filename must contain participant P#: {path.name}")
    participant = f"P{int(match.group(1))}"
    with np.load(path, allow_pickle=False) as archive:
        required = {"coefficients", "channel_labels", "harmonic_frequencies"}
        if not required.issubset(archive.files):
            raise ValueError(f"{path.name}: required arrays are {sorted(required)}")
        coefficients = archive["coefficients"]
        labels = tuple(archive["channel_labels"].astype(str).tolist())
        frequencies = np.asarray(archive["harmonic_frequencies"], dtype=float)
    if coefficients.ndim != 4 or coefficients.shape[1:] != (config.trials_per_stream, config.channels, 2):
        raise ValueError(f"{path.name}: coefficients must be unit x 10 x 32 x 2")
    if not np.iscomplexobj(coefficients) or labels != channel_labels or not np.allclose(frequencies, (4.0, 8.0)):
        raise ValueError(f"{path.name}: complex data, locked channel order, or 4/8-Hz labels invalid")
    return participant, coefficients.astype(np.complex64), path.name


def run(context: RunContext, force: bool = False) -> tuple[list[Path], dict]:
    del force
    stage, config = "05_negatives", context.config
    lock = json.loads((context.run_dir / "design_lock.json").read_text(encoding="utf-8"))
    participants = lock["participants"]
    external_files = sorted(config.external_negative_root.glob("*.npz")) if config.external_negative_root.exists() else []
    context.start_stage(stage, len(participants) + len(external_files))
    boundaries = _population_boundaries(context)
    rows = []
    for participant_index, participant in enumerate(participants, 1):
        with np.load(feature_path(context, participant), allow_pickle=False) as archive:
            off_target = archive["off_target"]
        for family_index, base in enumerate(config.off_target_bases_hz):
            rows.extend(_rows_for_units(participant, f"{base:g}+{2 * base:g} Hz", "off-target control",
                                        off_target[family_index], boundaries, config.k_values))
        context.progress(stage, participant_index, len(participants) + len(external_files), participant)
    external_errors = []
    external_valid_participants = []
    channel_labels = tuple(lock["channel_labels"])
    for file_index, path in enumerate(external_files, 1):
        try:
            participant, coefficients, family = _validate_external(path, config, channel_labels)
            if participant not in participants:
                raise ValueError(f"{path.name}: participant {participant} is not in the locked cohort")
            rows.extend(_rows_for_units(participant, family, "genuine external negative",
                                        coefficients, boundaries, config.k_values))
            external_valid_participants.append(participant)
        except (OSError, ValueError, KeyError) as error:
            external_errors.append(str(error))
        context.progress(stage, len(participants) + file_index, len(participants) + len(external_files), path.name)
    frame = pd.DataFrame(rows)
    result_path = context.table_dir / "negative_control_results.csv"
    frame.to_csv(result_path, index=False)
    summary_rows = []
    if not frame.empty:
        grouped = (frame.groupby(["control_kind", "family", "method"])["false_positive"]
                   .agg(false_positive_rate="mean", units="count").reset_index())
        summary_rows = grouped.to_dict("records")
    summary_path = context.table_dir / "negative_control_summary.csv"
    pd.DataFrame(summary_rows, columns=("control_kind", "family", "method", "false_positive_rate", "units")).to_csv(
        summary_path, index=False,
    )
    complete_external_cohort = set(external_valid_participants) == set(participants)
    status = {
        "off_target_label": "off-target controls, not genuine sham recordings",
        "off_target_families_hz": [[base, 2 * base] for base in config.off_target_bases_hz],
        "external_negative_status": "evaluated complete cohort" if complete_external_cohort and not external_errors else
                                    "evaluated incomplete cohort" if external_valid_participants and not external_errors else
                                    "invalid external negatives" if external_files else "external negatives unavailable",
        "external_files": [path.name for path in external_files], "external_errors": external_errors,
        "external_valid_participants": sorted(set(external_valid_participants)),
        "operational_fpr_claim_allowed": complete_external_cohort and not external_errors,
        "methods": list(METHODS[:2]),
    }
    status_path = context.run_dir / "negative_control_status.json"
    atomic_write_json(status_path, status)
    outputs = [result_path, summary_path, status_path]
    context.finish_stage(stage, outputs, status)
    return outputs, status
