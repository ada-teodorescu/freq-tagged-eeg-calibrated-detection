"""Reuse atomic unit checkpoints while recording support-specific provenance."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json
import tomllib

from cucu_article.runtime import atomic_write_json, atomic_write_text
from cucu_exposure_study.runtime import Config as BaseConfig, Run as BaseRun, read_config as read_base_config
from cucu_replication.io import inspect_set
from cucu_replication.runtime import utc_now


@dataclass(frozen=True)
class Config(BaseConfig):
    data_root: Path
    padding_samples: int
    sos_padlen: int


def read_config(path: str) -> Config:
    base = read_base_config(path)
    raw = tomllib.loads(base.path.read_text(encoding="utf-8"))
    config = Config(**vars(base), data_root=(base.root / raw["paths"]["data_root"]).resolve(),
                    padding_samples=int(raw["boundary"]["padding_samples"]),
                    sos_padlen=int(raw["boundary"]["sos_padlen"]))
    source = config.source
    if (source.sampling_rate != 1000 or source.matched_window_start_seconds != 0.5 or
            source.matched_window_end_seconds != 4.5 or source.hilbert_edge_crop_seconds != 0.5 or
            source.hilbert_order != 4 or source.hilbert_half_bandwidth_hz != 0.5):
        raise ValueError("Expected the source four-second, 1-kHz, order-4 +/-0.5-Hz pipeline")
    if config.padding_samples != 500 or config.sos_padlen != 27:
        raise ValueError("This design fixes synthetic margins at 500 samples and SOS padlen at 27")
    return config


class Run(BaseRun):
    def __init__(self, config: Config):
        super().__init__(config)
        folder = config.data_root / "Pre-processed EEGLAB datasets"
        self.epochs = {}
        for path in sorted(folder.glob("*.set")):
            epoch = inspect_set(path)
            if epoch.participant not in config.participants:
                continue
            if epoch.participant in self.epochs:
                raise ValueError(f"Duplicate EEG participant: {epoch.participant}")
            if (epoch.n_times != 5000 or epoch.n_channels != config.source.channels or
                    epoch.n_trials != config.source.streams * 10 or epoch.sampling_rate != 1000 or
                    epoch.channel_labels != config.channel_labels):
                raise ValueError(f"Unexpected EEG geometry: {epoch.participant}")
            self.epochs[epoch.participant] = epoch
        if set(self.epochs) != set(config.participants):
            raise ValueError(f"Missing EEG participants: {sorted(set(config.participants) - set(self.epochs))}")
        # Large source EEG uses file size/mtime identity, explicitly recorded below.
        self.eeg_manifest = {}
        for epoch in self.epochs.values():
            for path in (epoch.set_path, epoch.fdt_path):
                stat = path.stat()
                self.eeg_manifest[str(path)] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        identity = {"files_sha256": self.fingerprints, "eeg_metadata": self.eeg_manifest,
                    "versions": self.versions}
        self.identifier = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
        self.path = config.output_root / self.identifier

    def initialize(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        design_path = self.path / "design.json"
        if not design_path.exists():
            atomic_write_json(design_path, {
                "analysis_id": self.identifier, "created_at": utc_now(), "status": "exploratory",
                "participants": self.config.participants, "source_run": str(self.config.source_run),
                "endpoint": "static K=10 per stream; three streams per participant-condition",
                "scoring_interval_samples": [500, 4500], "sampling_rate": 1000,
                "primary_reflection": "odd", "sensitivity_reflection": "even",
                "synthetic_padding_samples_per_side": 500,
                "filter": "order-4 Butterworth SOS, harmonic +/-0.5 Hz, forward/backward",
                "filter_boundary": "odd internal padding, padlen=27; endpoint-scaled steady-state initialization",
                "analytic_boundary": "FFT Hilbert over 5000 padded samples; periodic extension; no extra zero padding",
                "restricted_input": "crop BEFORE padding/filtering/Hilbert; score all 4000 retained samples",
                "primary_contrasts": ["restricted_odd - hann", "restricted_odd - rectangular",
                                      "existing_hilbert - restricted_odd"],
                "sensitivity_contrasts": ["restricted_even - hann", "restricted_even - rectangular",
                                          "existing_hilbert - restricted_even", "restricted_even - restricted_odd"],
                "inference": "paired crossed-bootstrap pointwise 95% intervals; no refitting in bootstrap; no pass/fail gate",
                "model": "separate participant-held-out nonnegative logistic fit/threshold per method",
                "locked_c": self.config.source.locked_c, "target_fpr": self.config.source.target_fpr,
                "surrogates": "source seeds, ten trials; rotations shared across methods/channels, harmonic-coupled",
                "versions": self.versions,
            })
        atomic_write_json(self.path / "input_fingerprints.json", {
            "files_sha256": self.fingerprints, "eeg_size_mtime": self.eeg_manifest,
        })
        atomic_write_text(self.path / "config_snapshot.toml", self.config.path.read_text(encoding="utf-8"))
        atomic_write_json(self.config.output_root / "latest.json", {
            "analysis_id": self.identifier, "run_dir": str(self.path),
        })
