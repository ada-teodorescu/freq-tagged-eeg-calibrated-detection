from __future__ import annotations

import numpy as np

from cucu_algorithm_study.surrogates import same_frequency_scores
from cucu_detection.detectors import deterministic_subsets


def test_same_frequency_surrogates_destroy_locking_and_are_deterministic()->None:
    target=np.ones((10,3,4),dtype=np.complex128);subsets=deterministic_subsets(10,10,200,1)
    first=same_frequency_scores(target,subsets,200,np.random.SeedSequence(7));second=same_frequency_scores(target,subsets,200,np.random.SeedSequence(7))
    np.testing.assert_array_equal(first["null_phase"],second["null_phase"]);np.testing.assert_allclose(first["positive_phase"],1)
    assert first["null_phase"].mean()<0.2;assert first["null_phase"].shape==(1,200,4)
