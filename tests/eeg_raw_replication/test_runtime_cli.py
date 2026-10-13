from pathlib import Path

import numpy as np

from eeg_raw_replication.cli import main
from eeg_raw_replication.config import load_config
from eeg_raw_replication.runtime import RunContext, atomic_save_npz
from eeg_raw_replication.stages import stage_registry


CONFIG = """
[paths]
data_root = "data"
output_root = "outputs/raw_replication"
export_root = "results/raw_replication"
papers = []
[cohort]
participants = ["P9"]
[expected]
channels = 32
streams = 45
trials_per_stream = 10
samples_per_epoch = 5000
sampling_rate = 1000
[preprocessing]
l_freq = 1.0
h_freq = 50.0
montage = "standard_1020"
filter_method = "fir"
filter_phase = "zero"
fir_window = "hamming"
fir_design = "firwin"
average_reference = true
[ica]
components = 31
decimation = 4
frontal_channels = ["Fp1", "Fp2"]
frontal_power_threshold = 0.12
iclabel_eye_threshold = 0.80
[analysis]
seed = 20260818
harmonics_hz = [4.0, 8.0, 12.0, 16.0]
crop_start_seconds = 0.5
[branches]
enabled = ["paper12", "no_ica"]
[agreement]
match_min_r = 0.999
match_max_nrmse = 0.05
approximate_min_r = 0.90
approximate_max_nrmse = 0.50
[runtime]
workers = 1
maximum_checkpoint_gb = 20.0
minimum_progress_interval_seconds = 0.0
"""


def _config(tmp_path: Path):
    path = tmp_path / "raw_replication_config.toml"
    path.write_text(CONFIG, encoding="utf-8")
    return load_config(path)


def test_checkpoint_checksum_detects_corruption(tmp_path):
    context = RunContext.create(_config(tmp_path))
    checkpoint = context.checkpoint_dir / "synthetic.npz"
    atomic_save_npz(checkpoint, values=np.arange(5))
    context.mark_checkpoint(checkpoint, {"unit": "test"})
    assert context.checkpoint_valid(checkpoint)
    checkpoint.write_bytes(checkpoint.read_bytes() + b"corrupt")
    assert not context.checkpoint_valid(checkpoint)


def test_selection_changes_analysis_id_and_stale_is_selective(tmp_path):
    config = _config(tmp_path)
    first = RunContext.create(config, branches=("paper12",))
    second = RunContext.create(config, branches=("no_ica",))
    assert first.analysis_id != second.analysis_id
    first.start_stage("one", 1)
    first.finish_stage("one", [], {})
    first.start_stage("two", 1)
    first.finish_stage("two", [], {})
    first.mark_stale({"two"}, "test")
    assert first.read_state()["stages"]["one"]["status"] == "completed"
    assert first.read_state()["stages"]["two"]["status"] == "stale"


def test_cli_lists_isolated_stage_registry(capsys):
    root = Path(__file__).resolve().parents[2]
    assert main(["--config", str(root / "raw_replication_config.toml"), "list"]) == 0
    output = capsys.readouterr().out
    assert "00_inventory" in output
    assert "07_report" in output
    assert tuple(stage_registry()) == (
        "00_inventory", "01_events", "02_preprocess", "03_epoch_comparison",
        "04_spectral", "05_product_comparison", "06_paper_results", "07_report",
    )


def test_existing_replication_registry_is_unchanged():
    from eeg_replication.stages import stage_registry as existing_registry
    assert tuple(existing_registry()) == (
        "00_inventory", "01_stored_fft", "02_source_audit", "03_published_results",
        "04_hilbert", "05_filter_null", "06_frequency_correction", "07_report",
    )
