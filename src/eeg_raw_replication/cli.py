from __future__ import annotations

import argparse
from importlib import metadata
from pathlib import Path
import shutil
import sys
import time

from .config import load_config, normalize_branch, normalize_participant
from .events import discover_headers
from .runtime import RunContext
from .stages import Stage, stage_registry


def _csv(value: str | None, normalizer, defaults: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return defaults
    values = tuple(normalizer(item) for item in value.split(",") if item.strip())
    if not values or len(values) != len(set(values)):
        raise ValueError("Selection must contain unique comma-separated values")
    return values


def _resolve_stage(value: str, stages: dict[str, Stage]) -> str:
    if value in stages:
        return value
    matches = [name for name in stages if name.startswith(value)]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"Unknown or ambiguous stage {value!r}; choices: {', '.join(stages)}")


def _descendants(name: str, stages: dict[str, Stage]) -> set[str]:
    result = set()
    changed = True
    while changed:
        changed = False
        for candidate, stage in stages.items():
            if candidate != name and candidate not in result and any(dep == name or dep in result for dep in stage.dependencies):
                result.add(candidate)
                changed = True
    return result


def _execute(context: RunContext, name: str, stages: dict[str, Stage], forced: set[str], visiting=None) -> None:
    visiting = set() if visiting is None else visiting
    if name in visiting:
        raise RuntimeError(f"Circular dependency at {name}")
    visiting.add(name)
    for dependency in stages[name].dependencies:
        _execute(context, dependency, stages, forced, visiting)
    visiting.remove(name)
    force = name in forced
    if context.stage_completed(name) and not force:
        print(f"[{name}] already completed", flush=True)
        return
    try:
        stages[name].function(context, force)
    except BaseException as error:
        context.fail_stage(name, error)
        raise


def _selection(args, config):
    participants = _csv(getattr(args, "participants", None), normalize_participant, config.participants)
    unknown = sorted(set(participants) - set(config.participants))
    if unknown:
        raise ValueError(f"Participants outside the configured cohort: {unknown}")
    branches = _csv(getattr(args, "branches", None), normalize_branch, config.enabled_branches)
    return participants, branches


def _context(args, config) -> RunContext:
    participants, branches = _selection(args, config)
    return RunContext.create(config, participants, branches, getattr(args, "workers", None))


def _show_status(context: RunContext, stages: dict[str, Stage]) -> None:
    state = context.read_state()
    print(f"Cucu raw replication {context.analysis_id}")
    for name in stages:
        record = state.get("stages", {}).get(name, {})
        print(
            f"{name:25} {record.get('status', 'pending'):10} "
            f"{record.get('completed_units', 0)}/{record.get('total_units', 0)} "
            f"{record.get('current_unit') or ''}"
        )


def _doctor(config, participants: tuple[str, ...], branches: tuple[str, ...]) -> bool:
    problems = []
    print("Cucu raw replication doctor")
    for name in ("mne", "mne-icalabel", "onnxruntime"):
        try:
            version = metadata.version(name)
            print(f"[ok] {name} {version}")
        except metadata.PackageNotFoundError:
            needed = name == "mne" or (name in {"mne-icalabel", "onnxruntime"} and "iclabel" in branches)
            label = "missing (required)" if needed else "missing (unused by selection)"
            print(f"[{'error' if needed else 'note'}] {name}: {label}")
            if needed:
                problems.append(name)
    headers = discover_headers(config.paths.raw_root) if config.paths.raw_root.exists() else {}
    missing = sorted(set(participants) - set(headers))
    if missing:
        problems.append(f"missing raw headers: {missing}")
    else:
        print(f"[ok] {len(participants)} selected raw headers")
    for folder in (config.paths.preprocessed_root, config.paths.matlab_root, config.paths.original_repo_root):
        if not folder.exists():
            problems.append(f"missing path: {folder}")
        else:
            print(f"[ok] {folder}")
    disk = shutil.disk_usage(config.paths.project_root)
    requested = config.maximum_checkpoint_gb * 1024 ** 3
    if disk.free < requested:
        problems.append(f"free disk {disk.free / 1024**3:.1f} GB is below configured checkpoint budget")
    else:
        print(f"[ok] free disk {disk.free / 1024**3:.1f} GB")
    if problems:
        print("Doctor found blocking problems:")
        for problem in problems:
            print(f"- {problem}")
        return False
    print("Doctor passed.")
    return True


def _add_selection(parser: argparse.ArgumentParser, workers: bool = True) -> None:
    parser.add_argument("--participants", help="Comma-separated archive participant IDs")
    parser.add_argument("--branches", help="Comma-separated: primary, no_ica, iclabel, lowpass_no_ica")
    if workers:
        parser.add_argument("--workers", type=int)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Isolated Python/MNE raw replication of Cucu et al. (2022)")
    parser.add_argument("--config", default="raw_replication_config.toml")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List stages and dependencies")
    doctor = commands.add_parser("doctor", help="Validate dependencies, inputs, and disk space")
    _add_selection(doctor, workers=False)
    run = commands.add_parser("run", help="Run one stage and its incomplete prerequisites")
    run.add_argument("stage")
    _add_selection(run)
    run.add_argument("--force", action="store_true")
    run_all = commands.add_parser("run-all", help="Run the complete pipeline")
    _add_selection(run_all)
    run_all.add_argument("--force-stage", action="append", default=[])
    for name in ("status", "watch"):
        command = commands.add_parser(name, help=f"{name.title()} the persistent stage ledger")
        _add_selection(command)
        if name == "watch":
            command.add_argument("--interval", type=float, default=2.0)
    export = commands.add_parser("export", help="Explicitly export compact results and figures")
    _add_selection(export, workers=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    stages = stage_registry()
    if args.command == "list":
        for stage in stages.values():
            print(f"{stage.name}: {stage.description} [depends: {', '.join(stage.dependencies) or 'none'}]")
        return 0
    participants, branches = _selection(args, config)
    if args.command == "doctor":
        return 0 if _doctor(config, participants, branches) else 1
    context = _context(args, config)
    if args.command == "run":
        stage_name = _resolve_stage(args.stage, stages)
        forced = {stage_name} if args.force else set()
        if forced:
            context.mark_stale(_descendants(stage_name, stages), f"dependency {stage_name} was forced")
        _execute(context, stage_name, stages, forced)
    elif args.command == "run-all":
        forced = {_resolve_stage(value, stages) for value in args.force_stage}
        for name in forced:
            context.mark_stale(_descendants(name, stages), f"dependency {name} was forced")
        for stage_name in stages:
            _execute(context, stage_name, stages, forced)
    elif args.command in {"status", "watch"}:
        try:
            while True:
                _show_status(context, stages)
                if args.command == "status":
                    break
                time.sleep(max(0.25, args.interval))
        except KeyboardInterrupt:
            pass
    elif args.command == "export":
        if not (context.run_dir / "report.md").exists():
            raise RuntimeError("No completed report to export; run the pipeline first")
        print(context.export_compact_results())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
