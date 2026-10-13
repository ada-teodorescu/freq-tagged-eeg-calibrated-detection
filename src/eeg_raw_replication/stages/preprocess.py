from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..events import discover_headers
from ..mne_pipeline import (
    apply_ica, filter_diagnostics, fit_ica, iclabel_exclusions, make_epochs, paper_component_exclusions,
    require_mne, save_epochs_atomic, save_ica_atomic,
)
from ..runtime import RunContext, atomic_write_json
from .common import preprocess_paths


def _save_qc_figure(participant: str, scores: np.ndarray, paper_excluded: list[int], destination: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 3.8), constrained_layout=True)
    indices = np.arange(scores.size)
    colors = np.where(np.isin(indices, paper_excluded), "#c23b22", "#4978a8")
    axis.bar(indices, scores * 100.0, color=colors)
    axis.axhline(12.0, color="black", linestyle="--", linewidth=1)
    axis.set(title=f"{participant}: reconstructed frontal ICA power", xlabel="Component", ylabel="Fp1+Fp2 power (%)")
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def _save_topographies(ica, excluded: list[int], destination: Path) -> list[str]:
    picks = excluded or list(range(min(5, int(ica.n_components_))))
    figures = ica.plot_components(picks=picks, show=False)
    if not isinstance(figures, list):
        figures = [figures]
    saved = []
    for index, figure in enumerate(figures, start=1):
        path = destination.with_name(f"{destination.stem}_{index:02d}{destination.suffix}")
        figure.savefig(path, dpi=150)
        saved.append(str(path))
        import matplotlib.pyplot as plt
        plt.close(figure)
    return saved


def run(context: RunContext, force: bool = False) -> tuple[list, dict]:
    stage = "02_preprocess"
    require_mne()
    config = context.config
    headers = discover_headers(config.paths.raw_root)
    context.start_stage(stage, len(context.participants))
    summaries = []
    outputs: list[Path] = []
    need_ica = bool({"paper12", "iclabel"} & set(context.branches))
    for completed, participant in enumerate(context.participants, start=1):
        paths = preprocess_paths(context, participant)
        if not force and context.checkpoint_valid(paths["summary"]):
            import json
            summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
            summaries.append(summary)
            outputs.extend(path for path in paths.values() if path.exists())
            context.progress(stage, completed, len(context.participants), f"{participant} cached")
            continue

        base = make_epochs(headers[participant], config, lowpass_only=False)
        if not np.array_equal(np.bincount(base.events[:, 2], minlength=46)[1:], np.full(45, 10)):
            raise ValueError(f"{participant}: MNE epochs changed target event counts")
        save_epochs_atomic(base, paths["base"])
        context.mark_checkpoint(paths["base"], {"participant": participant, "kind": "bandpass_no_ica_epochs"})

        rank = None
        paper_scores = np.asarray([], dtype=float)
        paper_excluded: list[int] = []
        label_names: list[str] = []
        label_probabilities = np.asarray([], dtype=float)
        label_excluded: list[int] = []
        topographies: list[str] = []
        if need_ica:
            ica, rank = fit_ica(base, config)
            paper_scores, paper_excluded = paper_component_exclusions(
                ica.get_components(), base.ch_names,
                config.frontal_channels, config.frontal_power_threshold,
            )
            save_ica_atomic(ica, paths["ica"])
            context.mark_checkpoint(paths["ica"], {"participant": participant, "rank": rank})
            if "paper12" in context.branches:
                primary = apply_ica(base, ica, paper_excluded)
                save_epochs_atomic(primary, paths["paper12"])
                context.mark_checkpoint(paths["paper12"], {
                    "participant": participant, "branch": "paper12",
                    "excluded_components": paper_excluded,
                })
            if "iclabel" in context.branches:
                label_names, label_probabilities, label_excluded = iclabel_exclusions(
                    base, ica, config.iclabel_eye_threshold,
                )
            qc_figure = context.figure_dir / "participants" / f"{participant}_ica_scores.png"
            _save_qc_figure(participant, paper_scores, paper_excluded, qc_figure)
            topography_root = context.figure_dir / "participants" / f"{participant}_ica_topographies.png"
            try:
                topographies = _save_topographies(ica, paper_excluded, topography_root)
            except Exception as error:  # QC plotting must not invalidate numerical products.
                topographies = [f"plotting unavailable: {type(error).__name__}: {error}"]

        decisions = {
            "participant": participant,
            "ica": {
                "method": "extended_infomax", "seed": config.seed,
                "components": config.ica_components, "fit_decimation": config.ica_decimation,
                "rank": rank,
            },
            "paper12": {
                "formula": "sum squared scalp-map weights at Fp1,Fp2 / sum across all EEG channels",
                "threshold": config.frontal_power_threshold,
                "scores": paper_scores.tolist(), "excluded_components": paper_excluded,
                "verified_author_implementation": False,
            },
            "iclabel": {
                "eye_threshold": config.iclabel_eye_threshold,
                "labels": label_names, "maximum_probabilities": label_probabilities.tolist(),
                "excluded_components": label_excluded,
                "limitation": "ICLabel was trained for 1-100 Hz data; this fixed sensitivity branch uses the paper's 1-50 Hz data.",
            },
        }
        atomic_write_json(paths["decisions"], decisions)
        context.mark_checkpoint(paths["decisions"], {"participant": participant})
        summary = {
            "participant": participant, "epochs": len(base), "channels": len(base.ch_names),
            "samples_per_epoch": base.get_data(copy=False).shape[-1], "sampling_rate": float(base.info["sfreq"]),
            "paper12_excluded": paper_excluded, "iclabel_excluded": label_excluded,
            "topography_outputs": topographies,
            "documented_filter": filter_diagnostics(config),
        }
        atomic_write_json(paths["summary"], summary)
        context.mark_checkpoint(paths["summary"], {"participant": participant})
        summaries.append(summary)
        outputs.extend(path for path in paths.values() if path.exists())
        context.progress(stage, completed, len(context.participants), participant)

    component_rows = []
    for summary in summaries:
        participant = summary["participant"]
        import json
        decisions = json.loads(preprocess_paths(context, participant)["decisions"].read_text(encoding="utf-8"))
        for index, score in enumerate(decisions["paper12"]["scores"]):
            component_rows.append({
                "participant": participant, "component": index,
                "frontal_power_fraction": score,
                "paper12_rejected": index in decisions["paper12"]["excluded_components"],
                "iclabel": (decisions["iclabel"]["labels"] or [None] * config.ica_components)[index],
                "iclabel_probability": (decisions["iclabel"]["maximum_probabilities"] or [None] * config.ica_components)[index],
                "iclabel_rejected": index in decisions["iclabel"]["excluded_components"],
            })
    component_path = context.table_dir / "ica_components.csv"
    pd.DataFrame(component_rows).to_csv(component_path, index=False)
    outputs.append(component_path)
    used_gb = context.checkpoint_bytes() / 1024 ** 3
    if used_gb > config.maximum_checkpoint_gb:
        raise RuntimeError(f"Checkpoint footprint {used_gb:.2f} GB exceeds configured {config.maximum_checkpoint_gb:.2f} GB")
    report = {"passed": True, "participants": summaries, "checkpoint_gb": used_gb}
    report_path = context.run_dir / "preprocessing.json"
    atomic_write_json(report_path, report)
    outputs.append(report_path)
    context.finish_stage(stage, outputs, {"passed": True, "participants": len(summaries), "checkpoint_gb": used_gb})
    return outputs, report
