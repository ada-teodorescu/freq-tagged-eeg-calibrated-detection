from __future__ import annotations

import argparse
from importlib import metadata
import json
import shutil
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


def doctor(config) -> int:
    checks = []
    for package in ("numpy", "scipy", "pandas", "matplotlib", "rich", "mne"):
        try: checks.append((f"dependency:{package}", True, metadata.version(package)))
        except metadata.PackageNotFoundError: checks.append((f"dependency:{package}", package == "mne", "missing (needed only for stage 05)"))
    public = config.data_root / "Pre-processed EEGLAB datasets"
    checks.extend([
        ("public_set_fdt", len(list(public.glob("*.set"))) == 25 and len(list(public.glob("*.fdt"))) == 25, str(public)),
        ("confirmatory_baseline", (config.baseline_confirmatory_run / "confirmatory_conclusions.json").exists(), str(config.baseline_confirmatory_run)),
        ("external_baseline", (config.baseline_external_run / "external_validation_conclusions.json").exists(), str(config.baseline_external_run)),
        ("chronological_events", config.raw_events.exists(), str(config.raw_events)),
    ])
    free = shutil.disk_usage(config.project_root).free / 1024**3
    checks.append(("free_disk", free >= config.minimum_free_disk_gb, f"{free:.1f} GiB free"))
    for name, passed, detail in checks: print(f"{'PASS' if passed else 'FAIL'} {name}: {detail}")
    return 0 if all(passed for _, passed, _ in checks) else 1


def _status(context: RunContext, stages: dict[str, Stage]) -> None:
    state = context.read_state()
    for name in stages:
        row = state.get("stages", {}).get(name, {})
        print(f"{name:25s} {row.get('status', 'pending'):10s} {row.get('completed_units', 0)}/{row.get('total_units', 0)}")


def export(context: RunContext, force: bool) -> None:
    if not context.stage_completed("07_report"): raise RuntimeError("Run 07_report before export")
    destination = context.config.export_root / context.analysis_id
    if destination.exists() and any(destination.iterdir()) and not force:
        raise FileExistsError(f"Export exists: {destination}; pass --force")
    destination.mkdir(parents=True, exist_ok=True)
    selected = [context.run_dir / name for name in (
        "report.md", "methods.md", "results.md", "limitations.md", "design_lock.json",
        "validation_extension_conclusions.json", "assr_positive_protocol.md", "input_fingerprints.json",
    )]
    selected += sorted(context.table_dir.glob("*.csv")) + sorted(context.figure_dir.glob("*.png"))
    for source in selected:
        if not source.exists(): continue
        folder = destination / ("figures" if source.suffix == ".png" else "tables" if source.suffix == ".csv" else "")
        folder.mkdir(parents=True, exist_ok=True); shutil.copy2(source, folder / source.name)
    manifest = {"analysis_id": context.analysis_id, "source": str(context.run_dir),
                "files": sorted(str(path.relative_to(destination)).replace("\\", "/") for path in destination.rglob("*") if path.is_file())}
    (destination / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Cucu robustness and generalization extension")
    parser.add_argument("--config", default="validation_extension_config.toml")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor"); commands.add_parser("list")
    run = commands.add_parser("run"); run.add_argument("stage"); run.add_argument("--workers", type=int); run.add_argument("--force", action="store_true")
    all_parser = commands.add_parser("run-all"); all_parser.add_argument("--workers", type=int); all_parser.add_argument("--force-stage", action="append", default=[])
    status = commands.add_parser("status"); status.add_argument("--watch", action="store_true"); status.add_argument("--interval", type=float, default=2)
    export_parser = commands.add_parser("export"); export_parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv); config = load_config(args.config); stages = stage_registry()
    if args.command == "doctor": raise SystemExit(doctor(config))
    if args.command == "list":
        for stage in stages.values(): print(f"{stage.name}: {stage.description} [depends: {', '.join(stage.dependencies) or 'none'}]")
        return
    context = RunContext.create(config, getattr(args, "workers", None))
    if args.command == "run":
        name = _resolve(args.stage, stages); _execute(context, name, stages, {name} if args.force else set())
    elif args.command == "run-all":
        forced = {_resolve(value, stages) for value in args.force_stage}
        for name in stages: _execute(context, name, stages, forced)
    elif args.command == "export": export(context, args.force)
    else:
        try:
            while True:
                _status(context, stages)
                if not args.watch: break
                time.sleep(max(0.25, args.interval))
        except KeyboardInterrupt: pass


if __name__ == "__main__": main()
