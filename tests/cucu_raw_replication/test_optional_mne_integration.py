import os
from pathlib import Path

import pytest


def test_locked_mne_filter_design_is_explicit():
    pytest.importorskip("mne")
    from cucu_raw_replication.config import load_config
    from cucu_raw_replication.mne_pipeline import filter_diagnostics
    root = Path(__file__).resolve().parents[2]
    diagnostics = filter_diagnostics(load_config(root / "raw_replication_config.toml"))
    assert diagnostics["l_freq_hz"] == 1.0
    assert diagnostics["h_freq_hz"] == 50.0
    assert diagnostics["low_transition_bandwidth_hz"] == 1.0
    assert diagnostics["high_transition_bandwidth_hz"] == 12.5
    assert diagnostics["filter_length_samples"] > 0


@pytest.mark.skipif(os.environ.get("CUCU_RUN_RAW_INTEGRATION") != "1", reason="set CUCU_RUN_RAW_INTEGRATION=1")
def test_one_participant_raw_preprocessing():
    pytest.importorskip("mne")
    from cucu_raw_replication.cli import main
    root = Path(__file__).resolve().parents[2]
    result = main([
        "--config", str(root / "raw_replication_config.toml"), "run", "02_preprocess",
        "--participants", "P13", "--branches", "primary,no_ica",
    ])
    assert result == 0


@pytest.mark.skipif(os.environ.get("CUCU_RUN_RAW_E2E") != "1", reason="set CUCU_RUN_RAW_E2E=1")
def test_full_25_participant_pipeline():
    pytest.importorskip("mne")
    pytest.importorskip("mne_icalabel")
    from cucu_raw_replication.cli import main
    root = Path(__file__).resolve().parents[2]
    assert main(["--config", str(root / "raw_replication_config.toml"), "run-all"]) == 0
