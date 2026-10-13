from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

from cucu_algorithm_study.algorithms import fit_monotone_logistic, select_monotone_regularization
from cucu_article.stages.benchmark import auc_fast, empirical_threshold_fast
from ..features import matched_condition_observations
from ..runtime import RunContext, atomic_save_npz, atomic_write_json
from .features import feature_path


ALGORITHMS = (
    "C1_ConditionITPC4",
    "C2_Condition4plus8",
    "C3_Reliability4plus8",
    "C4_MonotoneHarmonic",
    "C5_RegionalMonotone",
)


def _load(context: RunContext, participants: list[str]) -> dict[str, np.ndarray]:
    names = (
        "positive_global", "null_global", "positive_weighted_global", "null_weighted_global",
        "positive_regional", "null_regional",
    )
    values: dict[str, list[np.ndarray]] = {name: [] for name in names}
    for participant in participants:
        with np.load(feature_path(context, participant), allow_pickle=False) as archive:
            for name in names:
                values[name].append(archive[name])
    return {name: np.stack(items) for name, items in values.items()}


def _model_arrays(features: dict[str, np.ndarray], key: str) -> tuple[np.ndarray, np.ndarray]:
    if key == "harmonic":
        return features["positive_global"], features["null_global"]
    positive = features["positive_regional"]
    null = features["null_regional"]
    return positive.reshape(*positive.shape[:2], -1), null.reshape(*null.shape[:3], -1)


def _fit_model(context: RunContext, positive: np.ndarray, null: np.ndarray,
               participant_numbers: np.ndarray, heldout: int, salt: int):
    training = participant_numbers != heldout
    seed = int(np.random.SeedSequence([context.config.seed, heldout, salt]).generate_state(1)[0])
    values, labels, groups = matched_condition_observations(
        positive[training], null[training], participant_numbers[training],
        context.config.training_nulls_per_condition, seed,
    )
    if heldout in set(groups.tolist()):
        raise AssertionError("held-out participant entered ML training")
    best_c, cv_scores = select_monotone_regularization(
        values, labels, groups, context.config.c_grid, context.config.inner_group_folds,
        seed, context.config.maximum_iterations, context.config.tolerance,
    )
    model = fit_monotone_logistic(
        values, labels, best_c, context.config.maximum_iterations, context.config.tolerance,
    )
    return model, cv_scores, training


def _scores(model, values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values).reshape(-1, values.shape[-1])
    return model.decision_function(flat).reshape(values.shape[:-1])


def run(context: RunContext, force: bool = False) -> tuple[list[Path], dict]:
    del force
    stage = "02_nested_benchmark"
    config = context.config
    lock = json.loads((context.run_dir / "design_lock.json").read_text(encoding="utf-8"))
    participants = list(lock["participants"])
    participant_numbers = np.arange(len(participants))
    features = _load(context, participants)
    scale = 2 * config.trials_per_stream
    fixed = {
        "C1_ConditionITPC4": (
            scale * features["positive_global"][..., 0], scale * features["null_global"][..., 0],
        ),
        "C2_Condition4plus8": (
            scale * features["positive_global"][..., :2].sum(axis=-1),
            scale * features["null_global"][..., :2].sum(axis=-1),
        ),
        "C3_Reliability4plus8": (
            scale * features["positive_weighted_global"][..., :2].sum(axis=-1),
            scale * features["null_weighted_global"][..., :2].sum(axis=-1),
        ),
    }
    harmonic_positive, harmonic_null = _model_arrays(features, "harmonic")
    regional_positive, regional_null = _model_arrays(features, "regional")
    rows: list[dict] = []
    model_rows: list[dict] = []
    model_paths: list[Path] = []
    context.start_stage(stage, len(participants))
    for heldout, participant in enumerate(participants):
        evaluated = dict(fixed)
        fold_models = {}
        for algorithm, positive, null, salt in (
            ("C4_MonotoneHarmonic", harmonic_positive, harmonic_null, 401),
            ("C5_RegionalMonotone", regional_positive, regional_null, 501),
        ):
            model, cv_scores, training = _fit_model(
                context, positive, null, participant_numbers, heldout, salt,
            )
            if not model.converged or np.any(model.coefficients < -1e-12):
                raise AssertionError(f"{algorithm} produced an invalid monotone fit")
            positive_scores = _scores(model, positive)
            null_scores = _scores(model, null)
            evaluated[algorithm] = (positive_scores, null_scores)
            fold_models[algorithm] = model
            model_rows.append({
                "participant": participant, "heldout_participant_index": heldout, "algorithm": algorithm,
                "selected_c": model.c_value, "intercept": model.intercept, "converged": model.converged,
                "feature_count": len(model.coefficients), "coefficients": json.dumps(model.coefficients.tolist()),
                "training_mean": json.dumps(model.standardizer.mean.tolist()),
                "training_scale": json.dumps(model.standardizer.scale.tolist()),
                **{f"inner_auc_c_{value:g}": cv_scores[value] for value in sorted(cv_scores)},
            })
            model_path = context.checkpoint_dir / stage / f"{participant}_{algorithm}.npz"
            atomic_save_npz(model_path, coefficients=model.coefficients, intercept=np.asarray(model.intercept),
                            mean=model.standardizer.mean, scale=model.standardizer.scale,
                            c_value=np.asarray(model.c_value))
            context.mark_checkpoint(model_path, {"participant": participant, "algorithm": algorithm})
            model_paths.append(model_path)

        training = participant_numbers != heldout
        for algorithm in ALGORITHMS:
            positive_scores, null_scores = evaluated[algorithm]
            for target_fpr in config.operating_fprs:
                threshold = empirical_threshold_fast(null_scores[training], target_fpr)
                held_positive = positive_scores[heldout]
                held_null = null_scores[heldout]
                for condition in range(config.conditions):
                    rows.append({
                        "participant": participant, "participant_index": heldout, "condition": condition,
                        "algorithm": algorithm, "target_fpr": target_fpr, "surrogate_block": "pooled",
                        "detected": int(held_positive[condition] > threshold),
                        "heldout_fpr": float(np.mean(held_null[condition] > threshold)),
                        "auc": auc_fast(held_positive[condition:condition + 1], held_null[condition]),
                        "positive_score": float(held_positive[condition]), "threshold": threshold,
                    })
            for block_index in range(len(config.surrogate_seeds)):
                start = block_index * config.surrogates_per_seed
                stop = start + config.surrogates_per_seed
                threshold = empirical_threshold_fast(null_scores[training, :, start:stop], config.target_fpr)
                held_positive = positive_scores[heldout]
                held_null = null_scores[heldout, :, start:stop]
                for condition in range(config.conditions):
                    rows.append({
                        "participant": participant, "participant_index": heldout, "condition": condition,
                        "algorithm": algorithm, "target_fpr": config.target_fpr,
                        "surrogate_block": f"seed_{block_index + 1}",
                        "detected": int(held_positive[condition] > threshold),
                        "heldout_fpr": float(np.mean(held_null[condition] > threshold)),
                        "auc": auc_fast(held_positive[condition:condition + 1], held_null[condition]),
                        "positive_score": float(held_positive[condition]), "threshold": threshold,
                    })
        context.progress(stage, heldout + 1, len(participants), participant)

    score_path = context.table_dir / "condition_scores.csv"
    model_table = context.table_dir / "models.csv"
    pd.DataFrame(rows).to_csv(score_path, index=False)
    pd.DataFrame(model_rows).to_csv(model_table, index=False)
    summary = {
        "participants": len(participants), "conditions": config.conditions, "methods": list(ALGORITHMS),
        "outer_validation": "leave_one_participant_out", "ml_training_unit": "participant_condition",
        "ml_evaluation_unit": "participant_condition", "heldout_participant_excluded": True,
    }
    report = context.run_dir / "nested_benchmark.json"
    atomic_write_json(report, summary)
    outputs = [score_path, model_table, report, *model_paths]
    context.finish_stage(stage, outputs, summary)
    return outputs, summary
