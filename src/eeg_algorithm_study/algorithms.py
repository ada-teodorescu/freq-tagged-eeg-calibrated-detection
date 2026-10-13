from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import optimize, special

from eeg_core.algorithms import Standardizer, fit_standardizer, grouped_folds
from eeg_core.stages.benchmark import auc_fast


@dataclass(frozen=True)
class MonotoneModel:
    coefficients: np.ndarray
    intercept: float
    standardizer: Standardizer
    c_value: float
    converged: bool

    def decision_function(self, values: np.ndarray) -> np.ndarray:
        return self.standardizer.transform(values) @ self.coefficients + self.intercept


def fit_monotone_logistic(
    values: np.ndarray, labels: np.ndarray, c_value: float, maximum_iterations: int=100, tolerance: float=1e-8,
) -> MonotoneModel:
    """Balanced L2 logistic model constrained to non-negative harmonic evidence."""
    x=np.asarray(values,dtype=float); y=np.asarray(labels,dtype=float).reshape(-1)
    if x.ndim!=2 or x.shape[0]!=y.size or set(np.unique(y))!={0.0,1.0}: raise ValueError("invalid training data")
    standardizer=fit_standardizer(x); z=standardizer.transform(x)
    counts=np.bincount(y.astype(int),minlength=2); weights=np.where(y==1,y.size/(2*counts[1]),y.size/(2*counts[0])); weight_sum=weights.sum()
    def objective(parameters:np.ndarray)->tuple[float,np.ndarray]:
        coefficients,intercept=parameters[:-1],parameters[-1]; logits=z@coefficients+intercept
        loss=np.sum(weights*(np.logaddexp(0.0,logits)-y*logits))/weight_sum+0.5*np.dot(coefficients,coefficients)/c_value
        residual=weights*(special.expit(logits)-y)/weight_sum
        gradient=np.concatenate((z.T@residual+coefficients/c_value,[residual.sum()]))
        return float(loss),gradient
    result=optimize.minimize(objective,np.zeros(x.shape[1]+1),jac=True,method="L-BFGS-B",
        bounds=[(0.0,None)]*x.shape[1]+[(None,None)],options={"maxiter":maximum_iterations,"ftol":tolerance,"gtol":tolerance})
    return MonotoneModel(np.asarray(result.x[:-1]),float(result.x[-1]),standardizer,float(c_value),bool(result.success))


def select_monotone_regularization(
    values:np.ndarray,labels:np.ndarray,groups:np.ndarray,c_grid:tuple[float,...],folds:int,seed:int,
    maximum_iterations:int,tolerance:float,
)->tuple[float,dict[float,float]]:
    x,y,g=np.asarray(values),np.asarray(labels),np.asarray(groups); fold_groups=grouped_folds(g,folds,seed); results={}
    for c_value in sorted(c_grid):
        aucs=[]
        for validation_groups in fold_groups:
            validation=np.isin(g,validation_groups); training=~validation
            model=fit_monotone_logistic(x[training],y[training],c_value,maximum_iterations,tolerance)
            scores=model.decision_function(x[validation]); aucs.append(auc_fast(scores[y[validation]==1],scores[y[validation]==0]))
        results[float(c_value)]=float(np.mean(aucs))
    best=max(results.values()); selected=min(c for c,score in results.items() if np.isclose(score,best,rtol=0,atol=1e-12))
    return float(selected),results


def training_observations(positive:np.ndarray,null:np.ndarray,participant_ids:np.ndarray)->tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Participant-condition observations from stream/subset means."""
    p=np.asarray(positive,dtype=float); n=np.asarray(null,dtype=float)
    participants,streams=p.shape[:2]; conditions=streams//3
    pos=p.reshape(participants,conditions,3,*p.shape[2:]).mean(axis=(2,3))
    neg=n.reshape(participants,conditions,3,*n.shape[2:]).mean(axis=(2,3))
    pos_values=pos.reshape(-1,pos.shape[-1]); neg_values=neg.reshape(-1,neg.shape[-1])
    values=np.concatenate((pos_values,neg_values)); labels=np.r_[np.ones(len(pos_values),dtype=np.int8),np.zeros(len(neg_values),dtype=np.int8)]
    group_base=np.repeat(np.asarray(participant_ids),conditions); groups=np.r_[group_base,np.repeat(group_base,n.shape[-2])]
    return values,labels,groups


def select_harmonic_subset(
    positive:np.ndarray,null:np.ndarray,candidates:tuple[tuple[int,...],...],participant_ids:np.ndarray,folds:int,seed:int,
)->tuple[tuple[int,...],dict[str,float]]:
    """Training-only grouped selection; ties prefer fewer harmonics."""
    fold_groups=grouped_folds(participant_ids,folds,seed); scores={}
    for candidate in candidates:
        candidate_scores=[]
        pos=np.sum(positive[...,list(candidate)],axis=-1); neg=np.sum(null[...,list(candidate)],axis=-1)
        for validation_groups in fold_groups:
            mask=np.isin(participant_ids,validation_groups)
            candidate_scores.append(auc_fast(pos[mask],neg[mask]))
        scores["_".join(str(i) for i in candidate)]=float(np.mean(candidate_scores))
    best=max(scores.values()); eligible=[candidate for candidate in candidates if np.isclose(scores["_".join(str(i) for i in candidate)],best,rtol=0,atol=1e-12)]
    return min(eligible,key=lambda candidate:(len(candidate),candidate)),scores
