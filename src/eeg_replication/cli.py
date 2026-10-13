from __future__ import annotations

import argparse
import sys
import time

from .config import load_config
from .runtime import RunContext
from .stages import Stage, stage_registry


def _resolve_stage(value: str, stages: dict[str, Stage]) -> str:
    if value in stages:
        return value
    matches = [name for name in stages if name.startswith(value)]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"Unknown or ambiguous stage {value!r}; choices: {', '.join(stages)}")


def _execute(
    context: RunContext,
    stage_name: str,
    stages: dict[str, Stage],
    force_stages: set[str],
    visiting: set[str] | None = None,
) -> None:
    visiting = set() if visiting is None else visiting
    if stage_name in visiting:
        raise RuntimeError(f"Circular stage dependency at {stage_name}")
    stage = stages[stage_name]
    visiting.add(stage_name)
    for dependency in stage.dependencies:
        _execute(context, dependency, stages, force_stages, visiting)
    visiting.remove(stage_name)
    force = stage_name in force_stages
    if context.stage_completed(stage_name) and not force:
        print(f"[{stage_name}] already completed; using checkpoints", flush=True)
        return
    try:
        stage.function(context, force)
    except BaseException as error:
        context.fail_stage(stage_name, error)
        raise


def _show_status(context: RunContext, stages: dict[str, Stage]) -> None:
    state = context.read_state()
    try:
        from rich.console import Console
        from rich.table import Table
        table = Table(title=f"Cucu replication {context.analysis_id}")
        table.add_column("Stage")
        table.add_column("Status")
        table.add_column("Progress", justify="right")
        table.add_column("Current unit")
        table.add_column("Last error")
        for name, stage in stages.items():
            record = state.get("stages", {}).get(name, {})
            completed = int(record.get("completed_units", 0))
            total = int(record.get("total_units", 0))
            progress = f"{completed}/{total}" if total else "-"
            error = str(record.get("error") or "").splitlines()[-1:]
            table.add_row(
                name,
                record.get("status", "pending"),
                progress,
                str(record.get("current_unit") or ""),
                error[0][:90] if error else "",
            )
        Console(file=sys.stdout, force_terminal=False, color_system=None).print(table)
    except ImportError:
        print(f"Cucu replication {context.analysis_id}")
        for name in stages:
            record = state.get("stages", {}).get(name, {})
            print(
                f"{name:24} {record.get('status', 'pending'):10} "
                f"{record.get('completed_units', 0)}/{record.get('total_units', 0)} "
                f"{record.get('current_unit') or ''}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Independent Cucu paper replication pipeline")
    parser.add_argument("--config", default="replication_config.toml")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List stages and dependencies")
    run = commands.add_parser("run", help="Run one stage and incomplete dependencies")
    run.add_argument("stage")
    run.add_argument("--workers", type=int, default=None)
    run.add_argument("--force", action="store_true")
    run_all = commands.add_parser("run-all", help="Run every stage in order")
    run_all.add_argument("--workers", type=int, default=None)
    run_all.add_argument("--force-stage", action="append", default=[])
    status = commands.add_parser("status", help="Show the persistent stage ledger")
    status.add_argument("--watch", action="store_true")
    status.add_argument("--interval", type=float, default=2.0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    stages = stage_registry()
    if args.command == "list":
        for stage in stages.values():
            dependencies = ", ".join(stage.dependencies) or "none"
            print(f"{stage.name}: {stage.description} [depends: {dependencies}]")
        return
    workers = getattr(args, "workers", None)
    context = RunContext.create(config, workers=workers)
    if args.command == "run":
        stage_name = _resolve_stage(args.stage, stages)
        _execute(context, stage_name, stages, {stage_name} if args.force else set())
    elif args.command == "run-all":
        forced = {_resolve_stage(name, stages) for name in args.force_stage}
        for stage_name in stages:
            _execute(context, stage_name, stages, forced)
    elif args.command == "status":
        try:
            while True:
                _show_status(context, stages)
                if not args.watch:
                    break
                time.sleep(max(0.25, args.interval))
        except KeyboardInterrupt:
            return


def run_script(stage_name: str, argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=f"Run replication stage {stage_name}")
    parser.add_argument("--config", default="replication_config.toml")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    context = RunContext.create(config, workers=args.workers)
    stages = stage_registry()
    _execute(context, stage_name, stages, {stage_name} if args.force else set())


if __name__ == "__main__":
    main()
