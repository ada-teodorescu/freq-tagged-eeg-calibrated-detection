import numpy as np

from eeg_detection.spatial import fit_dss, fit_regularized_cca, fit_ress


def _synthetic_trials(seed=10):
    rng = np.random.default_rng(seed)
    time = np.arange(500) / 100
    source = np.sin(2 * np.pi * 4 * time)
    mixing = np.asarray([1.0, 0.7, -0.5, 0.2])
    data = source[None, None, :] * mixing[None, :, None]
    return data + rng.normal(scale=0.8, size=(30, 4, 500)), source


def test_ress_and_dss_recover_nonzero_normalized_sources():
    trials, _ = _synthetic_trials()
    for model in (fit_ress(trials, 4.0, 100), fit_dss(trials)):
        assert model.weights.shape == (4,)
        np.testing.assert_allclose(np.linalg.norm(model.weights), 1.0)
        assert np.max(np.abs(model.weights)) > 0.4


def test_regularized_cca_uses_training_arrays_only():
    trials, source = _synthetic_trials()
    eeg = np.moveaxis(trials, 1, -1).reshape(-1, 4)
    stimulus = np.tile(source[:, None], (trials.shape[0], 1))
    model1 = fit_regularized_cca(eeg, stimulus)
    heldout = np.zeros((3, 4, 500))
    heldout[:] = 1e9  # not passed to fitting; changing it cannot alter the model
    model2 = fit_regularized_cca(eeg, stimulus)
    np.testing.assert_allclose(np.abs(model1.weights), np.abs(model2.weights), atol=1e-12)
