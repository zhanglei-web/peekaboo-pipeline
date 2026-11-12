"""Closed configuration, with secrets supplied only through the named environment."""
from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from jsonschema import Draft202012Validator

from .io import load, require


def closed(properties, required=None):
    return {"type": "object", "properties": properties,
            "required": list(properties) if required is None else required, "additionalProperties": False}


TEXT = {"type": "string", "minLength": 1}
CONFIG_SCHEMA = closed({
    "schema": {"const": "astra_memory.run_config.v1"},
    "episode_id": {"type": "string", "pattern": "^[A-Za-z0-9][A-Za-z0-9_.-]*$"},
    "scene_id": TEXT, "source_bag": TEXT, "output": TEXT, "event_images": TEXT,
    "fps": {"type": "number", "exclusiveMinimum": 0},
    "target": closed({"description": TEXT, "category": {"type": ["string", "null"]}}),
    "success_policy": {"const": "manual_confirmation_only"},
    "selection": closed({"mode": {"const": "all_bag_frames_baseline"}}),
    "model": closed({"name": {"enum": ["gpt-6-astra", "gpt-6.1-sol", "local-vlm"]}, "base_url": TEXT, "api_key_env": TEXT,
        "transport": {"enum": ["responses", "chat_completions"]},
        "reasoning_effort": {"enum": [None, "low", "medium", "high"]},
        "max_output_tokens": {"type": "integer", "minimum": 512},
        "timeout_s": {"type": "number", "exclusiveMinimum": 0},
        "transport_retries": {"type": "integer", "minimum": 0, "maximum": 3},
        "contract_retries": {"type": "integer", "minimum": 0, "maximum": 2},
        "past_images": {"type": "integer", "minimum": 0, "maximum": 2},
        "max_candidate_objects": {"type": "integer", "minimum": 1, "maximum": 100}}),
    "association": closed({"minimum_bbox_iou": {"type": "number", "minimum": 0, "maximum": 1}}),
})

CONFIG_SCHEMA["properties"]["model"]["properties"]["local_llama"] = closed({
    "context_size": {"type": "integer", "minimum": 8192},
    "image_max_edge": {"type": "integer", "minimum": 32},
    "image_max_tokens": {"type": "integer", "minimum": 1},
    "max_images": {"type": "integer", "minimum": 1, "maximum": 4},
    "server_record": TEXT, "model_lock_sha256": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
})
CONFIG_SCHEMA["properties"]["model"]["properties"]["local_llama"]["properties"].update({
    "parallel": {"type": "integer", "minimum": 1, "maximum": 2},
    "cache_type_k": {"enum": ["f16", "q8_0"]}, "cache_type_v": {"enum": ["f16", "q8_0"]},
})
CONFIG_SCHEMA["properties"]["execution"] = closed({"s2_prefetch": {"type": "boolean"}})
CONFIG_SCHEMA["properties"]["selection"] = {"oneOf": [
    closed({"mode": {"const": "all_bag_frames_baseline"}}),
    closed({"mode": {"const": "adaptive_keyframes_v1"},
        "max_frames": {"type": "integer", "minimum": 2},
        "max_gap_s": {"type": "number", "exclusiveMinimum": 0},
        "min_spacing_s": {"type": "number", "minimum": 0},
        "visual_change_threshold": {"type": "number", "minimum": 0, "maximum": 1}}),
]}

# Optional role budgets; every role uses the same explicitly configured model.
CONFIG_SCHEMA["properties"]["s2"] = closed({
    "knowledge_max_output_tokens": {"type": "integer", "minimum": 512, "maximum": 8192},
    "unit_max_output_tokens": {"type": "integer", "minimum": 512, "maximum": 16384},
})
CONFIG_SCHEMA["properties"]["s3"] = closed({
    "max_identity_reviews": {"type": "integer", "minimum": 0, "maximum": 200},
    "review_max_output_tokens": {"type": "integer", "minimum": 512, "maximum": 8192},
    "event_max_output_tokens": {"type": "integer", "minimum": 512, "maximum": 16384},
})


def load_config(path):
    path = Path(path).resolve()
    config = load(path)
    Draft202012Validator(CONFIG_SCHEMA).validate(config)
    for key in ["source_bag", "output", "event_images"]:
        p = Path(config[key]).expanduser()
        config[key] = str((p if p.is_absolute() else path.parent / p).resolve())
    url = urlparse(config["model"]["base_url"])
    require(url.scheme in {"http", "https"} and url.netloc and not url.username and
            not url.password and not url.query and not url.fragment, "invalid model service URL")
    require(url.scheme == "https" or url.hostname in {"localhost", "127.0.0.1", "::1"},
            "unencrypted model transport is only supported on loopback")
    if config["model"]["name"] == "local-vlm":
        require("local_llama" in config["model"] and url.hostname in {"127.0.0.1", "::1"},
                "local-vlm requires explicit local_llama settings and a numeric loopback URL")
        require(config["model"]["transport"] == "chat_completions" and
                config["model"]["reasoning_effort"] is None, "local-vlm requires chat completions with thinking disabled")
    else:
        require("local_llama" not in config["model"], "local_llama is reserved for local-vlm")
    output, source, images = map(Path, [config["output"], config["source_bag"], config["event_images"]])
    for input_root in [source, images, Path(__file__).resolve().parent]:
        require(output != input_root and not output.is_relative_to(input_root) and not input_root.is_relative_to(output),
                "output must be separate from inputs and source code")
    return config
