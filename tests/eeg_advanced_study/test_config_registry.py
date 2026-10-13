from __future__ import annotations

from eeg_advanced_study.config import load_config
from eeg_advanced_study.stages import stage_registry


def test_locked_config_and_stage_graph():
    config = load_config("advanced_study_config.toml")
    assert config.k_values == tuple(range(2, 11))
    assert config.ridge_grid == (0.05, 0.1, 0.25, 0.5, 1.0)
    known = {4.0, 8.0, 12.0, 16.0, 20.0, 24.0}
    assert all(base not in known and 2 * base not in known for base in config.off_target_bases_hz)
    stages = stage_registry()
    assert tuple(stages) == (
        "00_freeze", "01_features", "02_stream", "03_calibration",
        "04_condition", "05_negatives", "06_statistics", "07_report",
    )
    assert stages["07_report"].dependencies == ("06_statistics",)
