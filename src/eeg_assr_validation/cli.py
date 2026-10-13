from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
import shutil
import time

import pandas as pd

from .config import load_config
from .runtime import RunContext
from .stages import Stage, stage_registry


def _resolve(value: str, stages: dict[str, Stage]) -> str:
    if value in stages:
        return value
    matches = [name for name in stages if name.startswith(value)]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"Unknown or ambiguous stage {value!r}; choices: {', '.join(stages)}")


def _descendants(name: str, stages: dict[str, Stage]) -> set[str]:
    found = {name}
    changed = True
    while changed:
        before = len(found)
        found |= {candidate for candidate, stage in stages.items() if set(stage.dependencies) & found}
        changed = len(found) != before
    return found


def _execute(context: RunContext, name: str, stages: dict[str, Stage], forced: set[str], visiting=None) -> None:
    visiting = set() if visiting is None else visiting
    if name in visiting:
        raise RuntimeError(f"Circular dependency at {name}")
    visiting.add(name)
    for dependency in stages[name].dependencies:
        _execute(context, dependency, stages, forced, visiting)
    visiting.remove(name)
    if context.stage_completed(name) and name not in forced:
        print(f"[{name}] already completed; using checkpoints", flush=True)
        return
    try:
        stages[name].function(context, name in forced)
        forced.discard(name)
    except BaseException as error:
        context.fail_stage(name, error)
        raise


def doctor(config) -> int:
    checks: list[tuple[str, bool, str]] = []
    for package in ("numpy", "scipy", "pandas", "matplotlib", "mne"):
        try:
            checks.append((f"dependency:{package}", True, metadata.version(package)))
        except metadata.PackageNotFoundError:
            checks.append((f"dependency:{package}", False, "missing"))
    checks.append(("dataset_description", (config.dataset_root / "dataset_description.json").exists(), str(config.dataset_root)))
    participants = config.dataset_root / "participants.tsv"
    checks.append(("participants", participants.exists(), str(participants)))
    bdfs = sorted(config.dataset_root.rglob("*task-ASSR*_eeg.bdf")) if config.dataset_root.exists() else []
    real = [path for path in bdfs if path.stat().st_size > 1024 * 1024]
    checks.append(("ASSR_payloads", len(real) >= config.minimum_participants,
                   f"{len(real)} real BDF payloads of {len(bdfs)} metadata records"))
    sidecars_ok = bool(real) and all(path.with_name(path.name.replace("_eeg.bdf", "_events.tsv")).exists() and
                                     path.with_name(path.name.replace("_eeg.bdf", "_channels.tsv")).exists() and
                                     path.with_name(path.name.replace("_eeg.bdf", "_eeg.json")).exists() for path in real)
    checks.append(("BIDS_sidecars", sidecars_ok, "events/channels/eeg JSON for every downloaded ASSR payload"))
    eligible = 0
    contract_errors = []
    if participants.exists():
        table = pd.read_csv(participants, sep="\t")
        for participant in table.loc[table["group"] == config.group, "participant_id"].astype(str):
            eeg = config.dataset_root / participant / "eeg"
            matches = sorted(eeg.glob(f"{participant}_task-ASSR*_eeg.bdf"))
            if len(matches) != 1 or matches[0].stat().st_size <= 1024 * 1024:
                continue
            bdf = matches[0]; stem = bdf.name.removesuffix("_eeg.bdf")
            try:
                events = pd.read_csv(eeg / f"{stem}_events.tsv", sep="\t")
                channels = pd.read_csv(eeg / f"{stem}_channels.tsv", sep="\t")
                recording = json.loads((eeg / f"{stem}_eeg.json").read_text(encoding="utf-8"))
                counts = {frequency: int((events["trial_type"] == f"{int(frequency)}_Hz_Standard").sum())
                          for frequency in config.frequencies_hz}
                channel_count = int((channels["type"].str.upper() == "EEG").sum())
                if (float(recording["SamplingFrequency"]) == config.sampling_rate and channel_count == 64 and
                        all(counts[frequency] >= config.standard_limit(frequency) for frequency in config.frequencies_hz)):
                    eligible += 1
                else:
                    contract_errors.append(participant)
            except (OSError, KeyError, ValueError):
                contract_errors.append(participant)
    checks.append(("eligible_TD_contract", eligible >= config.minimum_participants,
                   f"{eligible} eligible; metadata-contract failures among downloaded TD: {', '.join(contract_errors) or 'none'}"))
    free = shutil.disk_usage(config.project_root).free / 1024**3
    checks.append(("free_disk", free >= config.minimum_free_disk_gb, f"{free:.1f} GiB free"))
    for name, passed, detail in checks:
        print(f"{'PASS' if passed else 'FAIL'} {name}: {detail}")
    return 0 if all(passed for _, passed, _ in checks) else 1


def _status(context: RunContext, stages: dict[str, Stage]) -> None:
    state = context.read_state()
    for name in stages:
        row = state.get("stages", {}).get(name, {})
        print(f"{name:28s} {row.get('status', 'pending'):10s} "
              f"{row.get('completed_units', 0)}/{row.get('total_units', 0)} {row.get('current_unit', '') or ''}")


def export(context: RunContext, force: bool) -> None:
    if not context.stage_completed("07_report"):
        raise RuntimeError("Run 07_report before export")
    destination = context.config.export_root / context.analysis_id
    if destination.exists() and any(destination.iterdir()) and not force:
        raise FileExistsError(f"Export exists: {destination}; pass --force")
    destination.mkdir(parents=True, exist_ok=True)
    selected = [context.run_dir / name for name in (
        "design_lock.json", "input_fingerprints.json", "methods.md", "results.md",
        "limitations.md", "report.md", "assr_validation_conclusions.json",
    )]
    selected += sorted(context.table_dir.glob("*.csv")) + sorted(context.figure_dir.glob("*.png"))
    for source in selected:
        if not source.exists():
            continue
        folder = destination / ("tables" if source.suffix == ".csv" else "figures" if source.suffix == ".png" else "")
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, folder / source.name)
    manifest = {"analysis_id": context.analysis_id, "source": str(context.run_dir),
                "files": sorted(str(path.relative_to(destination)).replace("\\", "/")
                                for path in destination.rglob("*") if path.is_file())}
    (destination / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run independent positive ASSR estimator validation")
    parser.add_argument("--config", default="assr_validation_config.toml")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor")
    commands.add_parser("list")
    run = commands.add_parser("run")
    run.add_argument("stage")
    run.add_argument("--workers", type=int)
    run.add_argument("--force", action="store_true")
    run_all = commands.add_parser("run-all")
    run_all.add_argument("--workers", type=int)
    run_all.add_argument("--force-stage", action="append", default=[])
    status = commands.add_parser("status")
    status.add_argument("--watch", action="store_true")
    status.add_argument("--interval", type=float, default=2.0)
    export_parser = commands.add_parser("export")
    export_parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    stages = stage_registry()
    if args.command == "doctor":
        raise SystemExit(doctor(config))
    if args.command == "list":
        for stage in stages.values():
            print(f"{stage.name}: {stage.description} [depends: {', '.join(stage.dependencies) or 'none'}]")
        return
    context = RunContext.create(config, getattr(args, "workers", None))
    if args.command == "run":
        name = _resolve(args.stage, stages)
        forced = _descendants(name, stages) if args.force else set()
        if forced:
            context.invalidate(forced, f"forced rerun from {name}")
        _execute(context, name, stages, forced)
    elif args.command == "run-all":
        forced: set[str] = set()
        for value in args.force_stage:
            forced |= _descendants(_resolve(value, stages), stages)
        if forced:
            context.invalidate(forced, "forced run-all dependency invalidation")
        for name in stages:
            _execute(context, name, stages, forced)
    elif args.command == "export":
        export(context, args.force)
    else:
        try:
            while True:
                _status(context, stages)
                if not args.watch:
                    break
                time.sleep(max(0.25, args.interval))
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
