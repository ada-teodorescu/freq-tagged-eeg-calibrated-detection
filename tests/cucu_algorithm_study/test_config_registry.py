from __future__ import annotations

from cucu_algorithm_study.config import load_config
from cucu_algorithm_study.stages import stage_registry


def test_config_and_stage_registry()->None:
    config=load_config("algorithm_study_config.toml")
    assert config.harmonic_subsets==((0,),(0,1),(0,1,2),(0,1,2,3))
    assert list(stage_registry())==["00_diagnosis","01_surrogate_features","02_repaired_benchmark","03_report"]
