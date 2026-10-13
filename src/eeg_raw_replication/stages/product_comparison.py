from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from ..reference import load_released_harmonics, load_released_stream_itpc
from ..runtime import RunContext, atomic_write_json
from ..statistics import agreement_metrics, classify_agreement


def _stored_participant_order(context: RunContext) -> tuple[str, ...]:
    paths = sorted(context.config.paths.preprocessed_root.glob("*.set"), key=lambda path: path.name.casefold())
    return tuple(path.name.split(" ", 1)[0].upper() for path in paths)


def _classification(context: RunContext, metrics: dict[str, float]) -> str:
    config = context.config
    return classify_agreement(
        metrics, config.match_min_r, config.match_max_nrmse,
        config.approximate_min_r, config.approximate_max_nrmse,
    )


def run(context: RunContext, force: bool = False) -> tuple[list, dict]:
    stage = "05_product_comparison"
    context.start_stage(stage, len(context.branches))
    with np.load(context.run_dir / "harmonic_itpc.npz", allow_pickle=False) as archive:
        raw_stream = np.asarray(archive["mean_itpc"], dtype=float)
    released_stream = load_released_stream_itpc(context.config.paths.original_repo_root)
    released_condition = load_released_harmonics(context.config.paths.matlab_root, context.config.harmonics_hz)
    stored_order = _stored_participant_order(context)
    stored_lookup = {participant: index for index, participant in enumerate(stored_order)}
    rows = []
    harmonic_rows = []
    assignment_rows = []
    summaries = {}
    for branch_index, branch in enumerate(context.branches):
        correlations = np.empty((len(context.participants), released_stream.shape[0]), dtype=float)
        for public_index, participant in enumerate(context.participants):
            candidate = raw_stream[branch_index, public_index]
            stored_index = stored_lookup[participant]
            metrics = agreement_metrics(candidate, released_stream[stored_index])
            classification = _classification(context, metrics)
            rows.append({
                "branch": branch, "participant": participant,
                "released_index": stored_index, "mapping": "lexicographic_same_id",
                "classification": classification, **metrics,
            })
            candidate_condition = candidate.reshape(15, 3, -1).mean(axis=1)
            condition_metrics = agreement_metrics(candidate_condition, released_condition[stored_index])
            rows.append({
                "branch": branch, "participant": participant,
                "released_index": stored_index, "mapping": "condition_same_id",
                "classification": _classification(context, condition_metrics), **condition_metrics,
            })
            for harmonic_index, frequency in enumerate(context.config.harmonics_hz):
                harmonic_metrics = agreement_metrics(candidate[:, harmonic_index], released_stream[stored_index, :, harmonic_index])
                harmonic_rows.append({
                    "branch": branch, "participant": participant, "harmonic_hz": frequency,
                    "classification": _classification(context, harmonic_metrics), **harmonic_metrics,
                })
            for released_index in range(released_stream.shape[0]):
                correlations[public_index, released_index] = np.corrcoef(
                    candidate.ravel(), released_stream[released_index].ravel(),
                )[0, 1]
        public_indices, released_indices = linear_sum_assignment(-correlations)
        for public_index, released_index in zip(public_indices, released_indices, strict=True):
            assignment_rows.append({
                "branch": branch, "public_participant": context.participants[public_index],
                "diagnostic_released_participant": stored_order[released_index],
                "pearson_r": float(correlations[public_index, released_index]),
                "scientific_mapping": False,
            })
        same_id = [row for row in rows if row["branch"] == branch and row["mapping"] == "lexicographic_same_id"]
        summaries[branch] = {
            "same_id_mean_r": float(np.mean([row["pearson_r"] for row in same_id])),
            "same_id_mean_nrmse": float(np.mean([row["normalized_rmse"] for row in same_id])),
            "optimal_assignment_mean_r": float(correlations[public_indices, released_indices].mean()),
            "classification_counts": {
                name: sum(row["classification"] == name for row in same_id)
                for name in ("matches", "approximately_agrees", "does_not_agree", "cannot_be_determined")
            },
        }
        context.progress(stage, branch_index + 1, len(context.branches), branch)
    agreement_path = context.table_dir / "released_product_agreement.csv"
    harmonic_path = context.table_dir / "released_harmonic_agreement.csv"
    assignment_path = context.table_dir / "diagnostic_participant_assignment.csv"
    report_path = context.run_dir / "product_comparison.json"
    pd.DataFrame(rows).to_csv(agreement_path, index=False)
    pd.DataFrame(harmonic_rows).to_csv(harmonic_path, index=False)
    pd.DataFrame(assignment_rows).to_csv(assignment_path, index=False)
    report = {
        "same_id_mapping_is_scientifically_primary": True,
        "optimal_assignment_is_diagnostic_only": True,
        "stored_order_hypothesis": list(stored_order),
        "thresholds": {
            "matches": {"minimum_r": context.config.match_min_r, "maximum_nrmse": context.config.match_max_nrmse},
            "approximately_agrees": {
                "minimum_r": context.config.approximate_min_r,
                "maximum_nrmse": context.config.approximate_max_nrmse,
            },
        },
        "branches": summaries,
    }
    atomic_write_json(report_path, report)
    outputs = [agreement_path, harmonic_path, assignment_path, report_path]
    context.finish_stage(stage, outputs, {"branches": summaries})
    return outputs, report
