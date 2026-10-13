from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from cucu_replication.statistics import covariance_pca, project_pca, rm_anova_oneway, wilson_interval


def test_covariance_pca_orientation_and_projection():
    values = np.asarray([[1, 2], [2, 4], [3, 6], [4, 8]], dtype=float)
    pca = covariance_pca(values)
    assert pca.loadings[0, 0] > 0
    np.testing.assert_allclose(project_pca(values, pca), pca.scores, atol=1e-15)
    assert pca.explained_variance_ratio[0] > 0.999


def test_repeated_measures_anova_detects_level_effect():
    rng = np.random.default_rng(2)
    subject_offsets = rng.normal(size=(20, 1))
    values = subject_offsets + np.asarray([0.0, 1.0, 2.0])[None, :] + rng.normal(scale=0.1, size=(20, 3))
    result = rm_anova_oneway(values)
    assert result["p_gg"] < 0.001
    assert 0.5 <= result["gg_epsilon"] <= 1.0


def test_wilson_interval_contains_observed_proportion():
    low, high = wilson_interval(5, 100)
    assert low < 0.05 < high


def test_local_pca_reproduces_published_variance_split():
    path = Path("data/Matlab variables/Itc_for_stats.mat")
    if not path.exists():
        pytest.skip("public study data are not installed")
    values = loadmat(path, squeeze_me=True)
    condition = np.stack([values[f"meanitc{h}"] for h in (4, 8, 12, 16)], axis=-1)
    participant = np.transpose(condition, (1, 0, 2)).mean(axis=1)
    explained = covariance_pca(participant).explained_variance_ratio
    np.testing.assert_allclose(explained[:2], [0.762027, 0.200788], atol=5e-4)
