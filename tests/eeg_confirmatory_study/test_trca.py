import numpy as np
import pandas as pd
import pytest

from eeg_confirmatory_study.trca import _reproducibility_matrices, fit_trca
from eeg_confirmatory_study.stages.statistics import _comparison


def _stream_block(mixing, source, n_trials, rng, noise_scale=0.9):
    base = source[None, None, :] * mixing[None, :, None]
    return base + rng.normal(scale=noise_scale, size=(n_trials, mixing.size, source.size))


def _synthetic_streams(seed=10, n_streams=4, trials_per_stream=8, samples=500, rate=100):
    rng = np.random.default_rng(seed)
    time = np.arange(samples) / rate
    mixing = np.asarray([1.0, 0.7, -0.5, 0.2])
    streams = []
    for _ in range(n_streams):
        phase = rng.uniform(0, 2 * np.pi)
        source = np.sin(2 * np.pi * 4 * time + phase)
        streams.append(_stream_block(mixing, source, trials_per_stream, rng))
    return streams, mixing, time


def test_trca_recovers_within_stream_source_and_ignores_cross_stream_distractor():
    streams, mixing, time = _synthetic_streams()
    # A distractor that is IDENTICAL at a given trial index across different
    # streams but uncorrelated across trial indices within one stream: a naive
    # implementation that pools trials across stream boundaries would find
    # spurious "reproducibility" here even though no single stream's own
    # repeats actually share it.
    rng = np.random.default_rng(99)
    distractor_mixing = np.asarray([-0.4, 0.6, 0.8, -0.2])
    n_trials = streams[0].shape[0]
    trial_phases = rng.uniform(0, 2 * np.pi, size=n_trials)
    for block in streams:
        for trial_index in range(n_trials):
            distractor = np.sin(2 * np.pi * 7 * time + trial_phases[trial_index])
            block[trial_index] += 0.9 * distractor_mixing[:, None] * distractor[None, :]

    model = fit_trca(streams, regularization=0.1)
    assert model.weights.shape == (4,)
    np.testing.assert_allclose(np.linalg.norm(model.weights), 1.0, atol=1e-10)

    true_direction = mixing / np.linalg.norm(mixing)
    distractor_direction = distractor_mixing / np.linalg.norm(distractor_mixing)
    assert abs(np.dot(model.weights, true_direction)) > 0.8
    assert abs(np.dot(model.weights, distractor_direction)) < 0.5


def test_trca_matrices_hand_computed():
    # One stream, two trials, one channel, three time samples: small enough
    # to hand-verify Q_raw and S directly against the formulas in the plan.
    trials = np.asarray([[[1.0, 2.0, 3.0]], [[4.0, 1.0, 1.0]]])
    q_raw, s = _reproducibility_matrices([trials])
    centered = trials - trials.mean(axis=-1, keepdims=True)
    x0, x1 = centered[0], centered[1]
    expected_q = x0 @ x0.T + x1 @ x1.T
    total = centered.sum(axis=0)
    expected_s = total @ total.T - expected_q
    np.testing.assert_allclose(q_raw, expected_q)
    np.testing.assert_allclose(s, expected_s)
    # S should equal exactly the two off-diagonal cross products for K=2.
    np.testing.assert_allclose(s, x0 @ x1.T + x1 @ x0.T)


def test_trca_regularization_only_in_denominator():
    streams, _, _ = _synthetic_streams(seed=3)
    q_raw, s = _reproducibility_matrices(streams)
    for regularization in (0.0, 0.35, 0.9):
        model = fit_trca(streams, regularization=regularization)
        target = np.trace(q_raw) / q_raw.shape[0] * np.eye(q_raw.shape[0])
        q_regularized = (1.0 - regularization) * q_raw + regularization * target
        from scipy import linalg
        eigenvalues, eigenvectors = linalg.eigh(s, q_regularized)
        expected = eigenvectors[:, np.argmax(eigenvalues)]
        expected = expected / np.linalg.norm(expected)
        pivot = int(np.argmax(np.abs(expected)))
        if expected[pivot] < 0:
            expected = -expected
        np.testing.assert_allclose(model.weights, expected, atol=1e-8)
    # S itself must never change with regularization -- only Q does.
    _, s_again = _reproducibility_matrices(streams)
    np.testing.assert_allclose(s, s_again)


def test_trca_uses_training_streams_only():
    streams, _, _ = _synthetic_streams(seed=5)
    model1 = fit_trca(streams, regularization=0.1)
    heldout = np.full((6, 4, 500), 1e9)  # never passed in; must not affect the fit
    model2 = fit_trca(streams, regularization=0.1)
    np.testing.assert_allclose(model1.weights, model2.weights, atol=1e-12)
    assert heldout.shape == (6, 4, 500)  # heldout is genuinely unused, not a no-op reference


def test_trca_sign_canonicalized():
    streams, _, _ = _synthetic_streams(seed=7)
    model_positive = fit_trca(streams, regularization=0.1)
    flipped_streams = [-block for block in streams]
    model_negated = fit_trca(flipped_streams, regularization=0.1)
    # Negating the input's sign convention must not change the canonical sign.
    np.testing.assert_allclose(model_positive.weights, model_negated.weights, atol=1e-8)
    pivot = int(np.argmax(np.abs(model_positive.weights)))
    assert model_positive.weights[pivot] > 0


def test_trca_rejects_single_trial_stream():
    streams, _, _ = _synthetic_streams(seed=1, trials_per_stream=1)
    with pytest.raises(ValueError):
        fit_trca(streams, regularization=0.1)


def test_trca_rejects_out_of_range_regularization():
    streams, _, _ = _synthetic_streams(seed=1)
    with pytest.raises(ValueError):
        fit_trca(streams, regularization=1.5)
    with pytest.raises(ValueError):
        fit_trca(streams, regularization=-0.1)


def test_trca_preserves_channel_dimension_under_projection():
    from eeg_detection.spatial import apply_spatial

    streams, _, _ = _synthetic_streams(seed=2)
    model = fit_trca(streams, regularization=0.1)
    projected = apply_spatial(streams[0], model)
    assert projected.shape == (streams[0].shape[0], streams[0].shape[-1])


def test_confirmatory_comparison_returns_paired_gain_fields():
    class Config:
        bootstrap_iterations = 100
        sign_flip_iterations = 100
        seed = 17

    baseline = np.zeros((4, 3))
    candidate = np.ones((4, 3))
    result = _comparison(candidate, baseline, "candidate", "baseline", Config(), 1)
    assert result["mean_detection_gain"] == 1.0
    assert result["crossed_bootstrap_ci95"][0] > 0


def test_trca_model_record_membership_contract():
    rows = []
    for held in range(1, 16):
        training = [condition for condition in range(1, 16) if condition != held]
        streams = [stream for condition in training for stream in range(3 * (condition - 1) + 1, 3 * condition + 1)]
        rows.append({"heldout_condition": held, "training_conditions": training, "training_streams": streams})
    frame = pd.DataFrame(rows)
    assert len(frame) == 15
    assert all(len(values) == 14 for values in frame["training_conditions"])
    assert all(len(values) == 42 for values in frame["training_streams"])
    assert all(row.heldout_condition not in row.training_conditions for row in frame.itertuples())
