"""python3 -m online_annotation initialize --session PATH --output PATH"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

from PIL import Image

from .contracts import KEYFRAMES, PROPOSALS, build_catalog, validate_keyframes, validate_proposals
from .frames import sample_scene, contact_sheet
from .model import JsonModel, save_json
from .report import write_report


ROOT = Path(__file__).resolve().parent


def initialize(args):
    started = time.perf_counter()
    config = json.loads(Path(args.config).read_text())
    brief_mode = getattr(args, "candidate_mode", "full") == "brief"
    multiscale_mode = getattr(args, "candidate_mode", "full") == "multiscale"
    if brief_mode:
        config["candidate_mode"] = "brief_two_full_visual_passes_v2"
    if multiscale_mode:
        config["candidate_mode"] = "single_multiscale_request_v1"
    if args.observation_seconds is not None:
        config["observation_seconds"] = args.observation_seconds
    if args.base_url:
        config["base_url"] = args.base_url
    if args.model_root:
        config["model_root"] = args.model_root
    for key in ["observation_seconds", "sample_hz", "temporal_bin_seconds",
                "max_shortlist", "max_semantic_keyframes", "model_image_max_edge"]:
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if config["max_semantic_keyframes"] > 4:
        raise ValueError("Current response contract supports at most four semantic keyframes")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    run = {"session": str(Path(args.session).resolve()), "config": config}
    run_path = output / "run_config.json"
    if run_path.exists() and json.loads(run_path.read_text()) != run:
        raise ValueError("Existing run configuration differs; use a new output directory")
    save_json(run_path, run)
    sampled, shortlist, audit = sample_scene(args.session, output / "candidates", config)
    save_json(output / "sampled_frames.json", sampled)
    save_json(output / "input_audit.json", audit)
    save_json(output / "shortlist.json", shortlist)
    contact_sheet(sampled, output / "sampled_contact_sheet.jpg", "观察前缀 · 全部采样帧", columns=4, width=384)
    contact_sheet(shortlist, output / "shortlist.jpg", "质量与视角去冗余 · 模型筛选输入")
    print(json.dumps({"stage": "sampled", "sampled": len(sampled),
                      "shortlist": [row["frame_index"] for row in shortlist]}, ensure_ascii=False), flush=True)
    if args.prepare_only:
        return
    model = JsonModel(config["model_root"], config.get("base_url"))
    ids = [row["frame_index"] for row in shortlist]
    key_prompt = (
        "图中每格标有F开头的原始帧号。这些图像全部来自同一段交互前观察。"
        "为建立候选搜索区域目录，选择清晰且有互补空间信息的最少帧；"
        "若全部展示同一场景且几乎重复，保留最清楚的一帧即可，"
        "不要按时间平均强行选满。保留可辨认的承托面、覆盖物和容器结构。"
        f"只能选择帧号{ids}中的1到{config['max_semantic_keyframes']}帧。"
        "selected_frame_indices填写整数零基原帧号，reason用中文解释。"
    )
    chosen, key_receipt = model.infer(output / "model/keyframes", key_prompt,
                                     [output / "shortlist.jpg"], KEYFRAMES, 800)
    validate_keyframes(chosen, ids)
    if len(chosen["selected_frame_indices"]) > config["max_semantic_keyframes"]:
        raise ValueError("Model exceeded configured semantic frame limit")
    selected = [row for row in shortlist if row["frame_index"] in chosen["selected_frame_indices"]]
    contact_sheet(selected, output / "selected_keyframes.jpg", "最终语义关键帧")
    model_images = []
    model_image_dir = output / "model_images"
    model_image_dir.mkdir(exist_ok=True)
    for row in selected:
        image = Image.open(row["image"]).convert("RGB")
        image.thumbnail((config["model_image_max_edge"], config["model_image_max_edge"]))
        path = model_image_dir / f"left_f{row['frame_index']:06d}.jpg"
        image.save(path, quality=95)
        model_images.append(path)
    mapping = "\n".join(f"图像{i}：原始frame_index={row['frame_index']}，"
                        f"相对时间{row['relative_seconds']:.3f}秒"
                        for i, row in enumerate(selected, 1))
    prompt = (ROOT / "prompts/regions.txt").read_text() + "\n\n本次图像顺序：\n" + mapping
    print(json.dumps({"stage": "region_inference", "keyframes": chosen}, ensure_ascii=False), flush=True)
    review_template = (ROOT / "prompts/region_review.txt").read_text()
    if multiscale_mode:
        from .multiscale import propose
        proposals, region_receipt = propose(model, output, selected)
        reviewed = proposals
        review_receipt = {"seconds": 0, "model": model.model, "usage": None,
                          "executed": False, "reason": "single complete multiscale request"}
    elif brief_mode:
        from .brief import propose
        proposals, reviewed, region_receipt, review_receipt = propose(
            model, output, model_images, selected, prompt, review_template)
    else:
        proposals, region_receipt = model.infer(output / "model/regions", prompt, model_images, PROPOSALS, 6500)
        validate_proposals(proposals, [row["frame_index"] for row in selected])
        review_prompt = (prompt + "\n\n" + review_template + "\n" + json.dumps(proposals, ensure_ascii=False))
        print(json.dumps({"stage": "region_review", "initial_regions": len(proposals["regions"])},
                         ensure_ascii=False), flush=True)
        reviewed, review_receipt = model.infer(output / "model/region_review", review_prompt,
                                             model_images, PROPOSALS, 6500)
    validate_proposals(reviewed, [row["frame_index"] for row in selected])
    catalog = build_catalog(reviewed, Path(args.session).name, selected, audit, model.model)
    catalog["model_review"] = {
        "initial_region_count": len(proposals["regions"]),
        "reviewed_region_count": len(reviewed["regions"]),
        "initial_names": [row["name"] for row in proposals["regions"]],
        "reviewed_names": [row["name"] for row in reviewed["regions"]],
        "independent_ground_truth": False,
    }
    if multiscale_mode:
        catalog["model_review"]["review_mode"] = "single_multiscale_request"
        catalog["model_review"]["separate_review_executed"] = False
    catalog["source_method"] = "docs/stage1_method.md"
    source_files = [ROOT / filename for filename in ["__main__.py", "frames.py", "contracts.py",
                    "model.py", "prompts/regions.txt", "prompts/region_review.txt"]]
    if brief_mode or multiscale_mode:
        source_files += [ROOT / "brief.py", ROOT / "compact.py", ROOT / "prompts/brief_transport.txt",
                         ROOT / "prompts/brief_review.txt"]
    if multiscale_mode:
        source_files += [ROOT / "multiscale.py", ROOT / "prompts/multiscale_regions.txt"]
    save_json(output / "implementation.json", {
        "source_hashes": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in source_files}})
    save_json(output / "catalog.v0.json", catalog)
    catalog_ready_seconds = time.perf_counter() - started
    receipts = {"keyframes": key_receipt, "regions": region_receipt,
                "region_review": review_receipt, "keyframe_reason": chosen["reason"]}
    save_json(output / "receipts.json", receipts)
    write_report(output, catalog, audit, selected, receipts)
    if brief_mode or multiscale_mode:
        save_json(output / "performance.json", {
            "measurement": "raw_session_to_catalog; model_loaded_before_start; non_streaming_replay",
            "catalog_ready_seconds": catalog_ready_seconds,
            "including_report_seconds": time.perf_counter() - started,
            "target_seconds": 120,
            "within_target": catalog_ready_seconds <= 120,
            "semantic_quality_verified": False,
            "model_seconds": sum(r["seconds"] for r in (key_receipt, region_receipt, review_receipt)),
            "result_cache_present_before_call": model.cache_observations,
        })
    print(json.dumps({"stage": "complete", "regions": len(catalog["regions"]),
                      "report": str(output / "report.html")}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("initialize", help="Build initial semantic candidate proposals")
    init.add_argument("--session", required=True)
    init.add_argument("--output", required=True)
    init.add_argument("--config", default=str(ROOT / "config.json"))
    init.add_argument("--observation-seconds", type=float)
    init.add_argument("--base-url")
    init.add_argument("--model-root", help="Directory containing the local Qwen service config and credential file")
    init.add_argument("--prepare-only", action="store_true")
    init.add_argument("--candidate-mode", choices=["full", "brief", "multiscale"], default="full",
                      help="full retains the v3 default; brief tests compact output with two full visual passes")
    args = parser.parse_args()
    try:
        initialize(args)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
