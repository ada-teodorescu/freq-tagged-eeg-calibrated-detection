from __future__ import annotations

from cucu_accuracy_study.config import load_config
from cucu_accuracy_study.stages import stage_registry
from cucu_algorithm_study.stages import stage_registry as repaired_registry


def test_accuracy_config_and_stage_registry() -> None:
    config = load_config("accuracy_study_config.toml")
    assert config.primary_surrogates == 300
    assert config.target_sensitivity == 0.90
    assert config.target_fpr == 0.05
    assert len(config.regions) == 5
    assert list(stage_registry()) == [
        "00_freeze", "01_condition_features", "02_nested_benchmark", "03_robustness", "04_report",
    ]


def test_repaired_study_registry_is_unchanged() -> None:
    assert list(repaired_registry()) == [
        "00_diagnosis", "01_surrogate_features", "02_repaired_benchmark", "03_report",
    ]
