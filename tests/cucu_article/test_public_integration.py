from __future__ import annotations

import os
import numpy as np
import pytest

from cucu_article.spectral import hann_coefficients
from cucu_replication.io import inspect_set, load_stream_trials


@pytest.mark.skipif(os.environ.get("CUCU_ARTICLE_INTEGRATION") != "1", reason="opt-in public EEG integration test")
def test_public_stream_hann_coefficients_match_completed_cache() -> None:
    epoch = inspect_set(__import__("pathlib").Path("data/Pre-processed EEGLAB datasets/P13 E2 epochs.set"))
    trials = load_stream_trials(epoch, 1)[..., 500:]
    target, neighbors = hann_coefficients(trials, 1000, 4.5, 4500, (4, 8, 12, 16), 18)
    cache = np.load("outputs/improvement/runs/aa16d5e06b5f/cache/public_set_fdt/participant_03/stream_01.npz")
    np.testing.assert_allclose(target, cache["F1_target"], rtol=0, atol=1e-10)
    np.testing.assert_allclose(neighbors, cache["F1_neighbors"], rtol=0, atol=1e-10)
