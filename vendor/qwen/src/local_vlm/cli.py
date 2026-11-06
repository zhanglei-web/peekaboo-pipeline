import argparse
import sys
from .client import Client, new_output, save_json
from .config import load_config, resolve
from .media import extract_video, load_manifest

def main(kind):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--task", choices=["describe", "event_annotation"], default="describe" if kind == "image" else "event_annotation")
    parser.add_argument("--prompt")
    parser.add_argument("--max-tokens", type=int)
    if kind == "image":
        parser.add_argument("--image", required=True)
    elif kind == "frames":
        parser.add_argument("--manifest", required=True)
    else:
        parser.add_argument("--video", required=True)
        parser.add_argument("--start", type=float, default=0)
        parser.add_argument("--end", type=float)
        parser.add_argument("--num-frames", type=int, default=4)
    args = parser.parse_args()
    out = new_output()
    print(f"Output: {out}", flush=True)
    try:
        cfg = load_config(args.config)
        if args.max_tokens is not None:
            if args.max_tokens < 1:
                raise ValueError("max-tokens must be positive")
            cfg["client"]["max_tokens"] = args.max_tokens
        if kind == "image":
            path = resolve(args.image)
            manifest = {"clip_id": path.stem, "frames": [{"frame_id": "f0", "path": str(path), "timestamp_s": 0.0}]}
        elif kind == "frames":
            manifest = load_manifest(resolve(args.manifest), cfg["client"]["max_frames"])
        else:
            if args.num_frames > cfg["client"]["max_frames"]:
                raise ValueError("Requested video frame count exceeds max_frames; configure explicitly")
            path = extract_video(resolve(args.video), out, args.start, args.end, args.num_frames)
            manifest = load_manifest(path, cfg["client"]["max_frames"])
        _, result = Client(cfg).run(manifest, args.task, args.prompt, out)
        import json
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as exc:
        if not (out / "metadata.json").exists():
            save_json(out / "metadata.json", {"status": "error", "error_type": type(exc).__name__, "error": str(exc), "stage": "input_validation"})
            save_json(out / "error.json", {"status": "error", "detail": str(exc)})
        print(f"ERROR: {exc}\nEvidence: {out}", file=sys.stderr)
        return 1
    return 0
