from __future__ import annotations

import numpy as np

from eeg_algorithm_study.algorithms import fit_monotone_logistic,select_harmonic_subset,training_observations


def test_monotone_logistic_never_uses_negative_evidence()->None:
    rng=np.random.default_rng(3);negative=rng.normal(1,0.2,size=(200,4));positive=rng.normal(-1,0.2,size=(200,4))
    values=np.vstack((negative,positive));labels=np.r_[np.zeros(200),np.ones(200)]
    model=fit_monotone_logistic(values,labels,10)
    assert model.converged;assert np.all(model.coefficients>=0);np.testing.assert_allclose(model.coefficients,0,atol=1e-6)


def test_training_observations_keep_groups_and_classes()->None:
    positive=np.ones((4,6,3,4));null=np.zeros((4,6,3,5,4));values,labels,groups=training_observations(positive,null,np.arange(4))
    assert values.shape==(4*2+4*2*5,4);assert labels.sum()==8;assert set(groups)==set(range(4))


def test_subset_selection_favors_informative_harmonics()->None:
    rng=np.random.default_rng(9);positive=rng.normal(size=(12,6,2,4));null=rng.normal(size=(12,6,2,5,4))
    positive[...,0]+=3;positive[...,1:]-=1
    selected,scores=select_harmonic_subset(positive,null,((0,),(0,1),(0,1,2),(0,1,2,3)),np.arange(12),6,10)
    assert selected==(0,);assert len(scores)==4
