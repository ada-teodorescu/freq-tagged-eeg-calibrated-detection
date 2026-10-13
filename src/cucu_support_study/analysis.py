"""Static, paired support comparison using the source feature/model specification."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from cucu_accuracy_study.features import exact_harmonic_coefficients, region_indices
from cucu_article.runtime import atomic_write_text
from cucu_confirmatory_study.stages.hilbert_condition import _condition_features
from cucu_detection.estimators import hilbert_coefficients
from cucu_detection.evaluation import crossed_bootstrap_gain
from cucu_exposure_study.analysis import fit_fold, load_bundle
from cucu_replication.io import load_stream_trials
from cucu_validation_extension.algorithms import rectangular_ls_coefficients

from .estimators import restricted_hilbert
from .runtime import Run


METHODS = ("hann", "rectangular", "existing_hilbert", "restricted_odd", "restricted_even")
PRIMARY_CONTRASTS = (("restricted_odd", "hann"), ("restricted_odd", "rectangular"),
                     ("existing_hilbert", "restricted_odd"))
SENSITIVITY_CONTRASTS = (("restricted_even", "hann"), ("restricted_even", "rectangular"),
                         ("existing_hilbert", "restricted_even"), ("restricted_even", "restricted_odd"))


def _verify_coefficients(actual: np.ndarray, expected: np.ndarray, label: str) -> float:
    """Reject source mismatch before mixing saved and newly estimated features."""
    difference = float(np.max(np.abs(actual - expected)))
    scale = max(float(np.max(np.abs(expected))), 1e-12)
    if difference > 1e-8 + 1e-5 * scale:
        raise ValueError(f"Source coefficient mismatch for {label}: relative maximum error {difference / scale:g}")
    return difference / scale


def coefficients(run: Run) -> None:
    config, source = run.config, run.config.source
    for index, participant in enumerate(config.participants):
        path = run.checkpoint("coefficients", participant)
        if not run.valid(path):
            run.progress("coefficients", index, len(config.participants), participant)
            epoch = run.epochs[participant]
            with np.load(config.coefficient_path(participant), allow_pickle=False) as saved:
                if (str(saved["participant"].item()) != participant or
                        tuple(saved["channel_labels"].tolist()) != config.channel_labels or
                        tuple(saved["harmonics_hz"].tolist()) != source.harmonics_hz):
                    raise ValueError(f"Saved coefficient metadata mismatch: {participant}")
                arrays = {"hann": saved["hann_matched"], "existing_hilbert": saved["hilbert"]}
            shape = (source.streams, 10, source.channels, 4)
            for key, value in arrays.items():
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError(f"Invalid saved coefficient array: {participant}/{key}")
            for key in ("rectangular", "restricted_odd", "restricted_even"):
                arrays[key] = np.empty(shape, dtype=np.complex64)
            errors = np.empty((source.streams, 2))
            for stream in range(source.streams):
                raw = load_stream_trials(epoch, stream + 1, dtype=np.float32)
                if not np.isfinite(raw).all():
                    raise ValueError(f"Nonfinite EEG: {participant}/{stream + 1}")
                # Reconstruct both reused methods to detect source data/code drift.
                hann = exact_harmonic_coefficients(raw[..., 500:], 1000, source.harmonics_hz, 0.0, 4.0)
                existing = hilbert_coefficients(raw, 1000, np.asarray(source.harmonics_hz), 4, 0.5, 0.5)
                errors[stream, 0] = _verify_coefficients(hann, arrays["hann"][stream], f"{participant}/Hann")
                errors[stream, 1] = _verify_coefficients(existing, arrays["existing_hilbert"][stream], f"{participant}/Hilbert")
                arrays["rectangular"][stream] = rectangular_ls_coefficients(
                    raw, 1000, np.asarray(source.harmonics_hz), 500, 4500)
                for reflection in ("odd", "even"):
                    arrays[f"restricted_{reflection}"][stream] = restricted_hilbert(
                        raw, 1000, source.harmonics_hz, padding=config.padding_samples,
                        sos_padlen=config.sos_padlen, reflection=reflection)
                run.progress("coefficients", index, len(config.participants),
                             f"{participant}: stream {stream + 1}/{source.streams}")
            if any(not np.isfinite(arrays[key]).all() for key in METHODS):
                raise ValueError(f"Nonfinite estimated coefficient: {participant}")
            run.save(path, **arrays, source_relative_errors=errors,
                     source_epoch_indices=np.stack([
                         np.flatnonzero(epoch.stream_by_trial == stream + 1) for stream in range(source.streams)]))
        run.progress("coefficients", index + 1, len(config.participants), participant)
    run.progress("coefficients", len(config.participants), len(config.participants), "all participants", "completed")


def features(run: Run) -> None:
    config = run.config
    regions = region_indices(config.channel_labels, config.source.regions)
    # The existing helper implements static K=10 and exactly the source surrogate seeds.
    context = SimpleNamespace(config=config.source)
    for index, participant in enumerate(config.participants):
        run.progress("features", index, len(config.participants), participant)
        destination = run.checkpoint("features", participant)
        if not run.valid(destination):
            arrays = {}
            with np.load(run.checkpoint("coefficients", participant), allow_pickle=False) as archive:
                for method in METHODS:
                    positive, null = _condition_features(context, index, archive[method], regions)
                    arrays[f"{method}_positive"] = positive.reshape(config.source.conditions, -1)
                    arrays[f"{method}_null"] = null.reshape(config.source.conditions, config.source.surrogates, -1)
            run.save(destination, **arrays)
    run.progress("features", len(config.participants), len(config.participants), "all participants", "completed")


def folds(run: Run) -> None:
    total = len(METHODS) * len(run.config.participants)
    done = 0
    for method in METHODS:
        positive = load_bundle(run, method, "positive")
        null = load_bundle(run, method, "null")
        for held, participant in enumerate(run.config.participants):
            run.progress("folds", done, total, f"{method}/{participant}")
            destination = run.checkpoint("folds", f"{method}_{participant}")
            if not run.valid(destination):
                # The shared fitter accepts arbitrary leading observation axes;
                # here there is one condition axis and no early/late dimension.
                run.save(destination, **fit_fold(positive, null, held, run.config))
            done += 1
    run.progress("folds", done, total, "all held-out fits", "completed")


def report(run: Run) -> None:
    run.progress("report", 0, 1, "paired statistics")
    scores, metrics, membership, audit = [], [], [], []
    decisions = {}
    for method in METHODS:
        method_decisions = []
        for participant in run.config.participants:
            with np.load(run.checkpoint("folds", f"{method}_{participant}"), allow_pickle=False) as fold:
                threshold = float(fold["threshold"])
                detected = fold["positive_scores"] > threshold
                null_detected = fold["null_scores"] > threshold
                method_decisions.append(detected.astype(float))
                metrics.append({"participant": participant, "method": method,
                                "detection_rate": float(detected.mean()),
                                "surrogate_fpr": float(null_detected.mean()), "threshold": threshold})
                for condition in range(run.config.source.conditions):
                    scores.append({"participant": participant, "method": method, "condition": condition + 1,
                                   "score": float(fold["positive_scores"][condition]), "threshold": threshold,
                                   "detected": bool(detected[condition]),
                                   "surrogate_fpr": float(null_detected[condition].mean())})
        decisions[method] = np.stack(method_decisions)
    comparisons = []
    for index, (candidate, comparator) in enumerate(PRIMARY_CONTRASTS + SENSITIVITY_CONTRASTS):
        gain = decisions[candidate] - decisions[comparator]
        low, high = crossed_bootstrap_gain(gain, run.config.bootstrap_iterations, run.config.seed + index)
        comparisons.append({"family": "primary_exploratory" if index < 3 else "padding_sensitivity",
                            "candidate": candidate, "comparator": comparator,
                            "gain": float(gain.mean()), "ci95_low": low, "ci95_high": high})
    for participant in run.config.participants:
        with np.load(run.checkpoint("coefficients", participant), allow_pickle=False) as archive:
            errors = archive["source_relative_errors"]
            audit.append({"participant": participant, "hann_max_relative_error": float(errors[:, 0].max()),
                          "hilbert_max_relative_error": float(errors[:, 1].max())})
            for stream, epochs in enumerate(archive["source_epoch_indices"]):
                for repetition, epoch_index in enumerate(epochs):
                    membership.append({"participant": participant, "condition": stream // 3 + 1,
                                       "stream": stream + 1, "repetition": repetition + 1,
                                       "source_epoch_1based": int(epoch_index) + 1})
    metric_frame = pd.DataFrame(metrics)
    summary = metric_frame.groupby("method", sort=False)[["detection_rate", "surrogate_fpr"]].mean().reset_index()
    frames = {"summary": summary, "participant_metrics": metric_frame, "condition_scores": pd.DataFrame(scores),
              "comparisons": pd.DataFrame(comparisons), "trial_membership": pd.DataFrame(membership),
              "source_coefficient_audit": pd.DataFrame(audit)}
    for name, frame in frames.items():
        atomic_write_text(run.path / "tables" / f"{name}.csv", frame.to_csv(index=False))
    lines = ["# Restricted-input Hilbert comparison", "", f"Analysis: `{run.identifier}` (exploratory)", "",
             "Static K=10 per stream; three streams per participant-condition. Each estimator has its own "
             "participant-held-out model, standardizer, and training-null threshold.", "",
             "## Detection and calibration", ""]
    for row in summary.itertuples(index=False):
        lines.append(f"- {row.method}: detection {100 * row.detection_rate:.1f}%; held-out surrogate FPR {100 * row.surrogate_fpr:.2f}%.")
    lines.extend(["", "## Paired contrasts", ""])
    for row in comparisons:
        lines.append(f"- {row['family']}: {row['candidate']} minus {row['comparator']}: "
                     f"{100 * row['gain']:.2f} pp (pointwise 95% CI {100 * row['ci95_low']:.2f} to {100 * row['ci95_high']:.2f}).")
    lines.extend(["", "## Interpretation and limits", "",
                  "Odd-reflection restricted Hilbert is the primary control. Even reflection is a fixed "
                  "sensitivity analysis, not a candidate to select by performance. Both crop to samples "
                  "[500,4500) before any new filtering or analytic transformation. Synthetic margins "
                  "contain no additional recorded EEG. See design.json for both filter and FFT boundaries.", "",
                  "An advantage over Hann with restricted input shows that additional estimator-input samples "
                  "are unnecessary for that advantage under the specified boundary handling. The rectangular "
                  "contrast addresses an untapered comparator. Existing minus restricted Hilbert combines "
                  "the effects of recorded context and replacing that context with synthetic boundary assumptions; "
                  "it does not isolate the Hilbert transform itself.", "",
                  "The public EEG was already preprocessed. Restriction applies to the inputs of the additional "
                  "target-frequency estimator, not to the temporal support of all upstream preprocessing.", "",
                  "Review surrogate FPRs before interpreting detection gains. Pointwise crossed-bootstrap "
                  "intervals resample participants and conditions with pairing preserved, hold fitted models "
                  "fixed, and have no multiplicity adjustment. No new confirmatory gate or equivalence claim "
                  "is made. No behavioral learning or perception endpoint is measured.", "",
                  "Source coefficients are checked against recomputation from every stream. Large EEG input "
                  "identity uses size/mtime metadata; configuration, source checkpoints, and code use SHA-256. "
                  "Outputs are separate from the original study and manuscript."])
    atomic_write_text(run.path / "report.md", "\n".join(lines) + "\n")
    run.progress("report", 1, 1, "report.md", "completed")
