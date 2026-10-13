import sys
import types

import numpy as np

from cucu_raw_replication.mne_pipeline import (
    frontal_power_scores, iclabel_exclusions, paper_component_exclusions,
)


def test_paper_rule_uses_sign_invariant_fp1_fp2_power():
    components = np.zeros((4, 2))
    components[:, 0] = [1, 1, 1, 1]       # frontal fraction 0.5
    components[:, 1] = [-0.1, 0.1, 1, 1]  # frontal fraction below 0.12
    names = ["Fp1", "Fp2", "Cz", "Oz"]
    scores, excluded = paper_component_exclusions(components, names, ("Fp1", "Fp2"), 0.12)
    assert np.allclose(scores, [0.5, 0.01 / 1.01])
    assert excluded == [0]
    assert np.allclose(frontal_power_scores(-components, names, ("Fp1", "Fp2")), scores)


def test_iclabel_branch_rejects_only_confident_eye_components(monkeypatch):
    module = types.ModuleType("mne_icalabel")
    module.label_components = lambda epochs, ica, method: {
        "labels": ["eye blink", "brain", "eye blink"],
        "y_pred_proba": np.asarray([0.91, 0.99, 0.60]),
    }
    monkeypatch.setitem(sys.modules, "mne_icalabel", module)
    labels, probabilities, excluded = iclabel_exclusions(object(), object(), 0.8)
    assert labels == ["eye blink", "brain", "eye blink"]
    assert np.allclose(probabilities, [0.91, 0.99, 0.60])
    assert excluded == [0]
