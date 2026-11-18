"""Current detector/action/cycles event frontend. External weights are required."""
import argparse
import json
from pathlib import Path

from .action_cycles import cycle_config
from .action_events import segment_action_events
from .actions import action_config, extract_actions
from .detector import HandObjectDetector
from .event_signals import extract_event_signals
from .events import event_config
from .video import detect_video, read_jsonl, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    detect = sub.add_parser("detect")
    detect.add_argument("--session", required=True)
    detect.add_argument("--output", required=True)
    detect.add_argument("--repository", required=True)
    detect.add_argument("--checkpoint", required=True)
    detect.add_argument("--device", default="cuda")
    detect.add_argument("--max-frames", type=int)
    act = sub.add_parser("recognize-actions")
    act.add_argument("--run", required=True)
    act.add_argument("--output", required=True)
    act.add_argument("--checkpoint", required=True)
    act.add_argument("--labels-dir", required=True)
    act.add_argument("--device", default="cuda")
    act.add_argument("--view", choices=("center", "contact_context"), default="contact_context")
    signal = sub.add_parser("extract-signals")
    signal.add_argument("--run", required=True)
    signal.add_argument("--output", required=True)
    signal.add_argument("--clip-checkpoint", required=True)
    signal.add_argument("--device", default="cuda")
    segment = sub.add_parser("segment-events")
    segment.add_argument("--run", required=True)
    segment.add_argument("--actions", required=True)
    segment.add_argument("--signals", required=True)
    segment.add_argument("--output", required=True)
    clip = sub.add_parser("clip-event")
    clip.add_argument("--run", required=True)
    clip.add_argument("--output", required=True)
    clip.add_argument("--start-frame", required=True, type=int)
    clip.add_argument("--end-frame", required=True, type=int)
    clip.add_argument("--clip-id", required=True)
    objects = sub.add_parser("recognize-objects")
    objects.add_argument("--clip", required=True)
    objects.add_argument("--output", required=True)
    objects.add_argument("--weights", required=True)
    objects.add_argument("--device", default="cuda")
    clusters = sub.add_parser("prepare-contact-cluster")
    clusters.add_argument("--clip", required=True)
    clusters.add_argument("--objects", required=True)
    clusters.add_argument("--output", required=True)
    rank = sub.add_parser("rank-candidates")
    rank.add_argument("--clusters", required=True)
    rank.add_argument("--clip", required=True)
    rank.add_argument("--output", required=True)
    rank.add_argument("--clip-checkpoint", required=True)
    rank.add_argument("--device", default="cuda")
    export = sub.add_parser("export-specs")
    export.add_argument("--run", required=True)
    export.add_argument("--actions", required=True)
    export.add_argument("--events", required=True)
    export.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "detect":
        detector = HandObjectDetector(Path(args.repository), Path(args.checkpoint), .3, .3,
                                      args.device, 4)
        result = detect_video(Path(args.session), Path(args.output), detector, max_frames=args.max_frames)
    elif args.command == "recognize-actions":
        result = extract_actions(args.run, args.output, args.checkpoint, args.labels_dir,
                                 action_config(), args.device, args.view)
    elif args.command == "extract-signals":
        result = extract_event_signals(args.run, args.output, args.clip_checkpoint,
                                       event_config(), args.device)
    elif args.command == "segment-events":
        result = segment_action_events(args.run, args.signals, args.actions, args.output,
                                       action_config(), event_config(), "cycles", cycle_config())
    elif args.command == "clip-event":
        from .clip import scope_detection
        result = scope_detection(args.run, args.output, args.start_frame, args.end_frame,
                                 args.clip_id)
    elif args.command == "recognize-objects":
        from .objects import YoloContactRecognizer, object_config, recognize_objects
        config = object_config()
        result = recognize_objects(args.clip, args.output,
                                   YoloContactRecognizer(args.weights, config, args.device), config)
    elif args.command == "prepare-contact-cluster":
        from .clusters import pipeline_config, prepare_cluster_candidates
        result = prepare_cluster_candidates(args.clip, args.objects, args.output,
                                            pipeline_config())
    elif args.command == "rank-candidates":
        from .features import feature_config, rank_cluster_frames
        result = rank_cluster_frames(args.clusters, args.clip, args.output,
                                     args.clip_checkpoint, args.device, feature_config())
    else:
        run = Path(args.run)
        manifest = json.loads((run / "manifest.json").read_text())
        index = read_jsonl(run / "frame_index.jsonl")
        actions = read_jsonl(Path(args.actions) / "actions.jsonl")
        predictions = json.loads((Path(args.events) / "predictions.json").read_text())
        video = manifest["source"]["video"]
        episode_id = Path(manifest["source"].get("session", video)).name
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        for row in predictions["predictions"]:
            spec = {"episode_id": episode_id, "event_id": row["event_id"],
                    "left_video": video,
                    "start_source_frame_id": row["start_source_frame_id"],
                    "end_source_frame_id": row["end_source_frame_id"],
                    "frame_index": [{"source_frame_id": x["source_frame_id"],
                                     "timestamp_s": x["relative_seconds"]} for x in index],
                    "action_rows": [{"source_frame_id": x["source_frame_id"],
                                     "action_label": x["action_label"],
                                     "confidence": (x.get("top_verbs") or [{}])[0].get("probability")}
                                    for x in actions],
                    "event_detection_provenance": {"prediction_status": row.get("status"),
                                                   "boundary_reason": row.get("boundary_reason"),
                                                   "review_required": row.get("review_required", False)}}
            save_json(out / f"{row['event_id']}.json", spec)
        result = {"exported": len(predictions["predictions"]), "output": str(out)}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
