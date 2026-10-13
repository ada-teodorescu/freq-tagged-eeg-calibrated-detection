"""Run/resume restricted-input controls without modifying source studies."""
from __future__ import annotations

import argparse
import json

from .analysis import coefficients, features, folds, report
from .runtime import Run, read_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Static restricted-input Hilbert comparison")
    parser.add_argument("--config", default="support_study_config.toml")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Read source metadata and validate inputs without computing coefficients")
    commands.add_parser("status", help="Read saved progress; does not start analysis")
    commands.add_parser("run-all", help="Run/resume all stages")
    stage = commands.add_parser("run")
    stage.add_argument("stage", choices=("coefficients", "features", "folds", "report"))
    args = parser.parse_args(argv)
    run = Run(read_config(args.config))
    print(f"Analysis {run.identifier}: {run.path}", flush=True)
    if args.command == "doctor":
        print(f"PASS: {len(run.epochs)} EEG participants, 5000 samples/epoch, 1000 Hz; source checkpoints available.")
        print("Source coefficient agreement is checked during the coefficients stage.")
        return
    if args.command == "status":
        state = run.path / "state.json"
        print(state.read_text(encoding="utf-8") if state.exists() else "Not started for this input/code identity.")
        return
    run.initialize()
    functions = {"coefficients": coefficients, "features": features, "folds": folds, "report": report}
    target = args.stage if args.command == "run" else "report"
    for name, function in functions.items():
        try:
            function(run)
        except BaseException as error:
            path = run.path / "state.json"
            state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            run.progress(name, state.get("completed", 0), state.get("total", 0),
                         f"{type(error).__name__}: {error}", "failed")
            raise
        if name == target:
            break


if __name__ == "__main__":
    main()
