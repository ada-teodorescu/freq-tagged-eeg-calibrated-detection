import json
from pathlib import Path

from eeg_replication.cli import main
from eeg_replication.config import load_config
from eeg_replication.runtime import (
    RunContext,
    atomic_save_npz,
    atomic_write_json,
    ordered_parallel_map,
)


CONFIG = """
[paths]
data_root = "data"
output_root = "output"
[expected]
participants = 25
channels = 32
streams = 45
target_wavs = 45
all_wavs = 95
trials_per_stream = 10
samples_per_epoch = 5000
sampling_rate = 1000
[analysis]
seed = 20260818
workers = 1
harmonics_hz = [4.0, 8.0, 12.0, 16.0]
crop_start_seconds = 0.5
crop_end_seconds = 0.5
exact_tolerance = 1e-12
bootstrap_iterations = 100
permutation_iterations = 100
[hilbert]
filter_order = 4
half_bandwidth_hz = 0.5
[filter_null]
orders = [2, 4, 8]
half_bandwidths_hz = [0.25, 0.5, 1.0]
edge_crops_seconds = [0.0, 0.5]
randomization_batches = 1
sampling_rate = 100
[acoustics]
expected_landmarks = 20
candidate_min_distance_ms = 80
ioi_target_ms = 250
ioi_soft_scale_ms = 50
envelope_lowpass_hz = 20.0
[runtime]
minimum_progress_interval_seconds = 0.0
"""


def _configuration(tmp_path: Path):
    path = tmp_path / "replication_config.toml"
    path.write_text(CONFIG, encoding="utf-8")
    for folder in ("Matlab variables", "Pre-processed EEGLAB datasets", "Stimuli"):
        (tmp_path / "data" / folder).mkdir(parents=True, exist_ok=True)
    return load_config(path)


def test_ledger_and_checkpoint_resume_contract(tmp_path):
    context = RunContext.create(_configuration(tmp_path))
    context.start_stage("synthetic", 2)
    context.progress("synthetic", 1, 2, "unit-1")
    checkpoint = context.checkpoint_dir / "synthetic" / "unit.json"
    atomic_write_json(checkpoint, {"value": 4})
    context.mark_checkpoint(checkpoint, {"unit": 1})
    assert context.checkpoint_valid(checkpoint)
    context.finish_stage("synthetic", [checkpoint], {"passed": True})
    resumed = RunContext.create(_configuration(tmp_path))
    assert resumed.stage_completed("synthetic")
    assert resumed.checkpoint_valid(checkpoint)
    assert json.loads(resumed.state_path.read_text())["stages"]["synthetic"]["status"] == "completed"


def test_cli_lists_numbered_stages(capsys):
    main(["--config", "replication_config.toml", "list"])
    output = capsys.readouterr().out
    assert "00_inventory" in output
    assert "07_report" in output


def test_changed_configuration_marks_previous_completed_stage_stale(tmp_path):
    configuration = _configuration(tmp_path)
    first = RunContext.create(configuration)
    first.start_stage("synthetic", 1)
    first.finish_stage("synthetic", [], {"passed": True})

    config_path = tmp_path / "replication_config.toml"
    config_path.write_text(CONFIG.replace("seed = 20260818", "seed = 20260819"), encoding="utf-8")
    second = RunContext.create(load_config(config_path))

    assert second.analysis_id != first.analysis_id
    old_record = first.read_state()["stages"]["synthetic"]
    assert old_record["status"] == "stale"
    assert second.read_state()["stages"] == {}


def test_failed_unit_is_recorded_and_can_be_restarted(tmp_path):
    context = RunContext.create(_configuration(tmp_path))
    context.start_stage("synthetic", 3)
    context.progress("synthetic", 1, 3, "unit-2")
    context.fail_stage("synthetic", RuntimeError("deliberate failure"))
    record = context.read_state()["stages"]["synthetic"]
    assert record["status"] == "failed"
    assert record["completed_units"] == 1
    assert "deliberate failure" in record["error"]

    context.start_stage("synthetic", 3)
    assert context.read_state()["stages"]["synthetic"]["status"] == "running"


def test_atomic_npz_and_worker_count_are_deterministic(tmp_path):
    checkpoint = tmp_path / "checkpoint.npz"
    atomic_save_npz(checkpoint, values=[1.0, 2.0, 3.0])
    assert checkpoint.exists()
    assert not checkpoint.with_name("checkpoint.npz.tmp.npz").exists()

    def square(value):
        return value * value

    items = list(range(20))
    assert ordered_parallel_map(square, items, 1) == ordered_parallel_map(square, items, 4)
