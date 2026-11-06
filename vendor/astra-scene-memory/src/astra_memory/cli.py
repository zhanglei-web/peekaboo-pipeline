"""Explicit stage, resumable run, confirmation, validation and batch commands."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .config import load_config
from .io import require, run_lock, save
from .pipeline import check, prepare_inputs, run_s1


def report_inside(config, path, result):
    report = Path(path).resolve()
    require(report.is_relative_to(Path(config["output"])), "report must be within run output")
    save(report, result)
    return {"report": str(report), "validation": result.get("validation", result.get("status"))}


def execute(args, config):
    progress = lambda text: print(text, file=sys.stderr, flush=True)
    if args.command == "s2":
        from .s2 import run_s2
        return run_s2(config, args.s1_source, args.allow_partial, args.through_frame, progress=progress)
    if args.command in {"s2-query", "s3-query"}:
        if args.command == "s2-query":
            from .s2 import query_s2 as query
        else:
            from .workflow import query_s3 as query
        result = query(config, args.at_frame)
        return report_inside(config, args.report, result) if args.report else result
    if args.command == "s3":
        from .s3 import run_s3
        return run_s3(config, args.allow_partial, progress=progress)
    if args.command == "confirm-template":
        from .confirmations import submission_template
        result = submission_template(config)
        path = Path(args.report).resolve() if args.report else Path(config["output"])/"human_submission.template.json"
        require(path.is_relative_to(Path(config["output"])), "template must be within output directory")
        require(not path.exists(), "template already exists; choose a new --report path to preserve human edits")
        save(path, result)
        return {"status": "awaiting_explicit_human_submission", "template": str(path), "accepted_as_confirmation": False}
    if args.command == "confirm-import":
        from .confirmations import import_confirmation
        return import_confirmation(config, args.input, args.actor)
    if args.command == "export":
        from .final_export import export_episode
        return export_episode(config, progress)
    if args.command == "validate":
        from .final_export import validate_export
        return validate_export(config)
    if args.command == "status":
        from .workflow import status
        return status(config)
    if args.command == "run":
        from .workflow import run_pipeline
        return run_pipeline(config, args.s1_source, progress)
    if args.command == "check":
        _, events, result = check(config)
        save(Path(config["output"])/"preflight.json", result)
        from .inputs import confirmation_todo
        path = Path(config["output"])/"confirmation_todo.json"
        if not path.exists():
            save(path, confirmation_todo(config, events))
        return result
    frames = prepare_inputs(config)
    if args.command == "s1":
        return run_s1(config, frames, args.max_frames, progress=progress)
    return {"prepared_bag_frames": len(frames), "output": config["output"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Independent single-model bag-to-search-episode pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["check", "prepare", "s1", "s2", "s2-query", "s3", "s3-query", "confirm-template", "confirm-import", "export", "validate", "status", "run"]:
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)
        if name == "s1":
            limits = command.add_mutually_exclusive_group(required=True)
            limits.add_argument("--max-frames", type=int, help="Process only this causal prefix")
            limits.add_argument("--all-bag-frames", action="store_true")
        if name in {"s2", "run"}:
            command.add_argument("--s1-source", help="Read a validated complete S1 run without rewriting it")
        if name in {"s2", "s3"}:
            command.add_argument("--allow-partial", action="store_true", help="Preview only; partial builds cannot be exported")
        if name == "s2":
            command.add_argument("--through-frame", type=int, help="Stop at this original video frame")
        if name in {"s2-query", "s3-query"}:
            command.add_argument("--at-frame", type=int, required=True)
            command.add_argument("--report", help="Save historical projection inside output")
        if name == "confirm-template":
            command.add_argument("--report", help="Write a fresh submission template inside output")
        if name == "confirm-import":
            command.add_argument("--input", required=True, help="Explicit human submission JSON")
            command.add_argument("--actor", required=True, help="Person explicitly submitting these decisions")
        if name == "run":
            command.add_argument("--all-bag-frames", action="store_true", required=True, help="Authorize full baseline; may issue many model requests")
    batch = commands.add_parser("batch")
    batch.add_argument("--configs", nargs="+", required=True)
    batch.add_argument("--all-bag-frames", action="store_true", required=True)
    batch.add_argument("--continue-on-error", action="store_true")
    batch.add_argument("--report", required=True, help="Explicit path for batch status JSON")
    args = parser.parse_args(argv)
    try:
        if args.command == "batch":
            from .workflow import run_pipeline
            configs = [load_config(p) for p in args.configs]
            require(len({c["output"] for c in configs}) == len(configs), "batch output directories must be distinct")
            report_path = Path(args.report).resolve()
            for c in configs:
                for protected in [Path(c["source_bag"]), Path(c["event_images"]), Path(__file__).resolve().parent]:
                    require(not report_path.is_relative_to(protected), "batch report cannot overwrite input/source files")
            require(not report_path.exists(), "batch report exists; use a fresh report path")
            results = []
            for config in configs:
                try:
                    with run_lock(config["output"]):
                        result = run_pipeline(config, progress=lambda t: print(t, file=sys.stderr, flush=True))
                    results.append({"episode_id": config["episode_id"], "status": "complete", "result": result})
                except Exception as exc:
                    results.append({"episode_id": config["episode_id"], "status": "failed", "detail": str(exc)})
                    if not args.continue_on_error:
                        break
                finally:
                    save(report_path, {"episodes": results, "requested_episode_count": len(configs), "completed_episode_count": sum(r["status"] == "complete" for r in results)})
            result = {"report": str(report_path), "episodes": results}
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if len(results) == len(configs) and all(r["status"] == "complete" for r in results) else 1
        config = load_config(args.config)
        with run_lock(config["output"]):
            result = execute(args, config)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
