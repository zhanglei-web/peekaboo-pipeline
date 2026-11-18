"""python -m spatial_memory <initialize|prepare-event|select|prepare-geometry|score|assemble-event|replay>"""
import argparse
from pathlib import Path

from .candidates import prepare_event
from .catalog import initialize
from .geometry import score_job
from .geometry_inputs import prepare_geometry
from .io import read_json, write_json
from .memory import replay
from .selection import select


def _relative(value, base):
    path = Path(value)
    return str(path if path.is_absolute() else (base / path).resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("initialize", help="Stage-one catalog to empty spatial memory")
    init.add_argument("--catalog", required=True)
    init.add_argument("--output", required=True)
    prep = sub.add_parser("prepare-event", help="Export event/action/reference frames from source video")
    prep.add_argument("--spec", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--max-frames", type=int, default=20)
    pick = sub.add_parser("select", help="Qwen3.8-8B selects Pre, PNR, Post and region associations")
    pick.add_argument("--candidates", required=True)
    pick.add_argument("--catalog", required=True)
    pick.add_argument("--output", required=True)
    pick.add_argument("--base-url", default="http://127.0.0.1:8089/v1")
    pick.add_argument("--model", default="Qwen3.8-8B")
    pick.add_argument("--image-max-edge", type=int, default=768)
    score = sub.add_parser("score", help="Three-frame fixed-reference geometry for one associated unit")
    score.add_argument("--job", required=True)
    score.add_argument("--selection", required=True)
    score.add_argument("--output", required=True)
    geo = sub.add_parser("prepare-geometry", help="Package external depth/masks and register selected RGB frames")
    geo.add_argument("--sources", required=True)
    geo.add_argument("--selection", required=True)
    geo.add_argument("--output", required=True)
    assemble = sub.add_parser("assemble-event", help="Freeze one event for cumulative memory replay")
    assemble.add_argument("--candidates", required=True)
    assemble.add_argument("--selection", required=True)
    assemble.add_argument("--assessments", nargs="+", required=True)
    assemble.add_argument("--available-after-s", type=float, required=True)
    assemble.add_argument("--output", required=True)
    build = sub.add_parser("replay", help="Chronological event replay from stage-one catalog")
    build.add_argument("--catalog", required=True)
    build.add_argument("--event-records", nargs="*", default=[])
    build.add_argument("--cutoff-s", type=float)
    build.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "initialize":
        result = initialize(read_json(args.catalog))
    elif args.command == "prepare-event":
        spec = read_json(args.spec)
        spec["left_video"] = _relative(spec["left_video"], Path(args.spec).resolve().parent)
        result = prepare_event(spec, args.output, max_frames=args.max_frames)
        return
    elif args.command == "select":
        event = read_json(args.candidates)
        for frame in event["candidates"]:
            frame["image"] = _relative(frame["image"], Path(args.candidates).resolve().parent)
        select(event, read_json(args.catalog), args.output, args.base_url, model=args.model,
               image_max_edge=args.image_max_edge)
        return
    elif args.command == "score":
        job = read_json(args.job)
        root = Path(args.job).resolve().parent
        if job.get("scope") == "original_support_footprint":
            job["reference_npz"] = _relative(job["reference_npz"], root)
            job["frames"] = {role: _relative(path, root) for role, path in job["frames"].items()}
        result = score_job(job, read_json(args.selection), args.output)
        return
    elif args.command == "prepare-geometry":
        config = read_json(args.sources)
        root = Path(args.sources).resolve().parent
        for key in ("left_image", "depth_npz", "object_mask", "support_mask", "dynamic_mask"):
            config["reference"][key] = _relative(config["reference"][key], root)
        for frame in config["frames"].values():
            for key in ("left_image", "depth_npz", "foreground_mask", "dynamic_mask"):
                frame[key] = _relative(frame[key], root)
        prepare_geometry(config, read_json(args.selection), args.output)
        return
    elif args.command == "assemble-event":
        event = read_json(args.candidates)
        selection = read_json(args.selection)
        if event["event_id"] != selection["event_id"] or event["episode_id"] != selection["episode_id"]:
            raise ValueError("Candidate event and selection disagree")
        result = {"episode_id": event["episode_id"], "event_id": event["event_id"],
                  "event_end_time_s": event["event_end_time_s"],
                  "available_after_s": args.available_after_s,
                  "catalog_version": selection["catalog_version"],
                  "selection": selection,
                  "assessments": [read_json(path) for path in args.assessments]}
    elif args.command == "replay":
        events = [read_json(path) for path in args.event_records]
        result = replay(read_json(args.catalog), events, args.cutoff_s)
    write_json(args.output, result)


if __name__ == "__main__":
    main()
