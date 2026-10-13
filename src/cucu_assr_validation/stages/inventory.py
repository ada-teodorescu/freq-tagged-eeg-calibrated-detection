from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import json

import numpy as np
import pandas as pd

from ..runtime import RunContext, atomic_write_json


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _paths(root: Path, participant: str) -> dict[str, Path] | None:
    eeg = root / participant / "eeg"
    bdfs = sorted(eeg.glob(f"{participant}_task-ASSR*_eeg.bdf"))
    if len(bdfs) != 1:
        return None
    bdf = bdfs[0]
    stem = bdf.name.removesuffix("_eeg.bdf")
    return {"bdf": bdf, "events": eeg / f"{stem}_events.tsv", "channels": eeg / f"{stem}_channels.tsv",
            "eeg_json": eeg / f"{stem}_eeg.json"}


def select_standard_events(events: pd.DataFrame, frequency_hz: float, limit: int,
                           trials_per_block: int) -> pd.DataFrame:
    label = f"{int(frequency_hz)}_Hz_Standard"
    selected = events.loc[events["trial_type"] == label].sort_values(["onset", "sample"]).head(limit).copy()
    if len(selected) < limit or limit % trials_per_block:
        raise ValueError(f"need {limit} complete {frequency_hz:g}-Hz standard trials")
    selected["trial"] = np.arange(1, limit + 1)
    selected["block"] = (selected["trial"] - 1) // trials_per_block + 1
    selected["trial_in_block"] = (selected["trial"] - 1) % trials_per_block + 1
    return selected


def run(context: RunContext, force: bool = False) -> tuple[list, dict]:
    stage, config = "01_inventory", context.config
    participants_path = config.dataset_root / "participants.tsv"
    participants = pd.read_csv(participants_path, sep="\t")
    td = sorted(participants.loc[participants["group"] == config.group, "participant_id"].astype(str))
    context.start_stage(stage, len(td))
    inventory_rows, membership_rows, exclusions, fingerprints = [], [], [], []
    for index, participant in enumerate(td, 1):
        paths = _paths(config.dataset_root, participant)
        if paths is None:
            exclusions.append({"participant": participant, "reason": "missing_or_multiple_ASSR_recordings"})
            context.progress(stage, index, len(td), f"{participant} excluded")
            continue
        missing = [name for name, path in paths.items() if not path.exists()]
        if missing or paths["bdf"].stat().st_size <= 1024 * 1024:
            exclusions.append({"participant": participant, "reason": "missing_sidecar_or_annex_pointer:" + ",".join(missing)})
            context.progress(stage, index, len(td), f"{participant} excluded")
            continue
        eeg_meta = json.loads(paths["eeg_json"].read_text(encoding="utf-8"))
        channels = pd.read_csv(paths["channels"], sep="\t")
        events = pd.read_csv(paths["events"], sep="\t")
        eeg_channels = channels.loc[channels["type"].str.upper() == "EEG", "name"].astype(str).tolist()
        sfreq = float(eeg_meta["SamplingFrequency"])
        duration = float(eeg_meta["RecordingDuration"])
        reason = None
        counts = {}
        if sfreq != config.sampling_rate or len(eeg_channels) != 64 or len(set(eeg_channels)) != 64:
            reason = f"recording_contract:sfreq={sfreq},eeg_channels={len(eeg_channels)}"
        elif events["onset"].astype(float).min() < 0 or events["onset"].astype(float).max() >= duration:
            reason = "event_out_of_recording_bounds"
        else:
            for frequency in config.frequencies_hz:
                label = f"{int(frequency)}_Hz_Standard"
                counts[frequency] = int((events["trial_type"] == label).sum())
                if counts[frequency] < config.standard_limit(frequency):
                    reason = f"insufficient_{int(frequency)}Hz_standards:{counts[frequency]}"
                    break
        if reason:
            exclusions.append({"participant": participant, "reason": reason})
            context.progress(stage, index, len(td), f"{participant} excluded")
            continue
        checkpoint = context.checkpoint_dir / stage / f"{participant}.json"
        if context.checkpoint_valid(checkpoint) and not force:
            fingerprint = json.loads(checkpoint.read_text(encoding="utf-8"))
        else:
            fingerprint = {"participant": participant, "bdf_sha256": _sha256(paths["bdf"]),
                           "bdf_bytes": paths["bdf"].stat().st_size,
                           "events_sha256": _sha256(paths["events"]), "channels_sha256": _sha256(paths["channels"])}
            atomic_write_json(checkpoint, fingerprint)
            context.mark_checkpoint(checkpoint, {"participant": participant, "input_fingerprint": fingerprint["bdf_sha256"]})
        fingerprints.append(fingerprint)
        inventory_rows.append({"participant": participant, "bdf_path": str(paths["bdf"]),
                               "events_path": str(paths["events"]), "channels_path": str(paths["channels"]),
                               "sampling_rate": sfreq, "duration_seconds": duration,
                               "eeg_channels": len(eeg_channels), "standards_40": counts[40.0], "standards_27": counts[27.0],
                               "bdf_sha256": fingerprint["bdf_sha256"]})
        for frequency in config.frequencies_hz:
            selected = select_standard_events(events, frequency, config.standard_limit(frequency), config.trials_per_block)
            for row in selected.itertuples(index=False):
                membership_rows.append({"participant": participant, "frequency_hz": frequency,
                                        "trial": int(row.trial), "block": int(row.block),
                                        "trial_in_block": int(row.trial_in_block),
                                        "onset_seconds": float(row.onset), "sample": int(row.sample),
                                        "trial_type": str(row.trial_type)})
        context.progress(stage, index, len(td), participant)
    eligible = tuple(row["participant"] for row in inventory_rows)
    if len(eligible) < config.minimum_participants:
        raise RuntimeError(f"only {len(eligible)} eligible TD participants; require {config.minimum_participants}")
    inventory_path = context.table_dir / "recording_inventory.csv"
    membership_path = context.table_dir / "trial_membership.csv"
    exclusion_path = context.table_dir / "cohort_exclusions.csv"
    fingerprint_path = context.run_dir / "input_fingerprints.json"
    pd.DataFrame(inventory_rows).to_csv(inventory_path, index=False)
    pd.DataFrame(membership_rows).to_csv(membership_path, index=False)
    pd.DataFrame(exclusions, columns=("participant", "reason")).to_csv(exclusion_path, index=False)
    atomic_write_json(fingerprint_path, {"dataset_id": config.dataset_id, "dataset_version": config.dataset_version,
                                        "eligible_participants": list(eligible), "files": fingerprints})
    outputs = [inventory_path, membership_path, exclusion_path, fingerprint_path]
    summary = {"eligible_participants": len(eligible), "excluded_participants": len(exclusions),
               "blocks_40_per_participant": 17, "blocks_27_per_participant": 16}
    context.finish_stage(stage, outputs, summary)
    return outputs, summary
