from __future__ import annotations

import argparse
import sys
import time
from .config import load_config
from .runtime import RunContext
from .stages import Stage, stage_registry


def _resolve(value: str, stages: dict[str, Stage]) -> str:
    if value in stages: return value
    matches = [name for name in stages if name.startswith(value)]
    if len(matches) == 1: return matches[0]
    raise ValueError(f"Unknown or ambiguous stage {value!r}; choices: {', '.join(stages)}")


def _execute(context: RunContext, name: str, stages: dict[str, Stage], forced: set[str], visiting=None) -> None:
    visiting = set() if visiting is None else visiting
    if name in visiting: raise RuntimeError(f"Circular dependency at {name}")
    visiting.add(name)
    for dependency in stages[name].dependencies: _execute(context, dependency, stages, forced, visiting)
    visiting.remove(name)
    if context.stage_completed(name) and name not in forced:
        print(f"[{name}] already completed; using checkpoints", flush=True); return
    try: stages[name].function(context, name in forced)
    except BaseException as error: context.fail_stage(name, error); raise


def _status(context: RunContext, stages: dict[str, Stage]) -> None:
    state = context.read_state()
    try:
        from rich.console import Console
        from rich.table import Table
        table = Table(title=f"Cucu improvement {context.analysis_id}")
        for column in ("Stage", "Status", "Progress", "Current unit", "Last error"): table.add_column(column)
        for name in stages:
            record = state.get("stages", {}).get(name, {})
            total, done = int(record.get("total_units", 0)), int(record.get("completed_units", 0))
            errors = str(record.get("error") or "").splitlines()
            table.add_row(name, record.get("status", "pending"), f"{done}/{total}" if total else "-",
                          str(record.get("current_unit") or ""), errors[-1][:90] if errors else "")
        Console(file=sys.stdout, force_terminal=False, color_system=None).print(table)
    except ImportError:
        for name in stages: print(name, state.get("stages", {}).get(name, {}).get("status", "pending"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Focused Cucu detector follow-up")
    parser.add_argument("--config", default="improvement_config.toml")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    run = commands.add_parser("run"); run.add_argument("stage"); run.add_argument("--workers", type=int); run.add_argument("--force", action="store_true")
    run_all = commands.add_parser("run-all"); run_all.add_argument("--workers", type=int); run_all.add_argument("--force-stage", action="append", default=[])
    status = commands.add_parser("status"); status.add_argument("--watch", action="store_true"); status.add_argument("--interval", type=float, default=2)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv); stages = stage_registry()
    if args.command == "list":
        for stage in stages.values(): print(f"{stage.name}: {stage.description} [depends: {', '.join(stage.dependencies) or 'none'}]")
        return
    context = RunContext.create(load_config(args.config), getattr(args, "workers", None))
    if args.command == "run": _execute(context, _resolve(args.stage, stages), stages, {_resolve(args.stage, stages)} if args.force else set())
    elif args.command == "run-all":
        forced = {_resolve(value, stages) for value in args.force_stage}
        for name in stages: _execute(context, name, stages, forced)
    else:
        try:
            while True:
                _status(context, stages)
                if not args.watch: break
                time.sleep(max(0.25, args.interval))
        except KeyboardInterrupt: pass


if __name__ == "__main__": main()
