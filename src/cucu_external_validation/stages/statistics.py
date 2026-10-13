from __future__ import annotations

from pathlib import Path

import pandas as pd

from cucu_article.runtime import atomic_write_json
from ..runtime import RunContext
from ..statistics import participant_cluster_interval


def _summarize(frame: pd.DataFrame, task: str, endpoint: str, config, seed_salt: int,
               analysis: str = "primary_all_epochs") -> dict:
    column = "static_false_positive" if endpoint == "static" else "sequential_false_positive"
    selected = frame[frame["task"] == task]
    participant_rates = selected.groupby("participant")[column].mean().astype(float)
    interval = participant_cluster_interval(participant_rates.to_numpy(), config.bootstrap_iterations,
                                            config.seed + seed_salt, config.one_sided_confidence)
    return {"analysis": analysis, "task": task, "endpoint": endpoint,
            "participants": int(len(participant_rates)), "units": int(len(selected)), **interval,
            "upper_limit": config.external_fpr_upper_limit,
            "upper_bound_pass": bool(interval["one_sided_upper"] < config.external_fpr_upper_limit)}


def run(context: RunContext, force: bool = False) -> tuple[list[Path], dict]:
    del force
    stage, config = "07_statistics", context.config
    context.start_stage(stage, 4)
    scores = pd.read_csv(context.table_dir / "external_negative_scores.csv")
    tasks = (config.primary_task, *config.secondary_tasks)
    primary_rows = []
    for task_index, task in enumerate(tasks):
        for endpoint_index, endpoint in enumerate(("static", "sequential")):
            primary_rows.append(_summarize(scores, task, endpoint, config, 10 * task_index + endpoint_index))
    context.progress(stage, 1, 4, "primary external FPR summaries")

    artifacts = pd.read_csv(context.table_dir / "epoch_artifact_sensitivity.csv")
    acceptable = (artifacts.groupby(["participant", "task", "unit"])["sensitivity_rejected"]
                  .any().rename("unit_rejected").reset_index())
    filtered = scores.merge(acceptable, on=["participant", "task", "unit"], how="left")
    filtered = filtered[~filtered["unit_rejected"].fillna(True)]
    sensitivity_rows = []
    for task_index, task in enumerate(tasks):
        if filtered[filtered["task"] == task]["participant"].nunique() < 2: continue
        for endpoint_index, endpoint in enumerate(("static", "sequential")):
            row = _summarize(filtered, task, endpoint, config, 100 + 10 * task_index + endpoint_index,
                             "artifact_sensitivity")
            row["upper_bound_pass"] = None
            sensitivity_rows.append(row)
    context.progress(stage, 2, 4, "artifact-rejection sensitivity")

    summary_path = context.table_dir / "external_fpr_summary.csv"
    pd.DataFrame(primary_rows + sensitivity_rows).to_csv(summary_path, index=False)
    primary = [row for row in primary_rows if row["task"] == config.primary_task]
    primary_cohort_ok = all(row["participants"] >= config.minimum_primary_participants for row in primary)
    secondary_counts = {task: scores[scores["task"] == task]["participant"].nunique() for task in config.secondary_tasks}
    secondary_cohorts_ok = all(count >= config.minimum_secondary_participants for count in secondary_counts.values())
    endpoint_passes = {row["endpoint"]: bool(row["upper_bound_pass"]) for row in primary}
    submission_gate_passed = bool(primary_cohort_ok and secondary_cohorts_ok and all(endpoint_passes.values()))
    context.progress(stage, 3, 4, "submission gate")

    conclusions = {
        "analysis_id": context.analysis_id, "dataset_id": config.dataset_id,
        "primary_task": config.primary_task, "endpoint_passes": endpoint_passes,
        "primary_cohort_ok": primary_cohort_ok, "secondary_cohorts_ok": secondary_cohorts_ok,
        "secondary_participants": {key: int(value) for key, value in secondary_counts.items()},
        "submission_gate_passed": submission_gate_passed,
        "manuscript_replacement_allowed": submission_gate_passed,
        "failure_rule": "Preserve and report failure; do not retune preprocessing, models, thresholds, boundaries, or margin.",
        "interpretation_guardrail": "Passing supports transfer to this independent jittered-auditory cohort, not universal or clinical operational FPR.",
    }
    conclusion_path = context.run_dir / "external_validation_conclusions.json"
    atomic_write_json(conclusion_path, conclusions)
    context.progress(stage, 4, 4, "conclusions locked")
    outputs = [summary_path, conclusion_path]
    context.finish_stage(stage, outputs, {"submission_gate_passed": submission_gate_passed})
    return outputs, conclusions
