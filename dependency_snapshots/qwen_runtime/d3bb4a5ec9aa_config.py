from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]

def resolve(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path

def load_config(path=None):
    path = resolve(path or "configs/default.yaml")
    cfg = yaml.safe_load(path.read_text())
    for group, keys in [("server", ["port", "ctx_size", "parallel", "batch_size", "ubatch_size", "image_max_tokens", "startup_timeout"]),
                        ("client", ["max_tokens", "image_max_edge", "max_frames", "timeout", "resource_interval_s"])]:
        for key in keys:
            if cfg[group][key] <= 0:
                raise ValueError(f"{group}.{key} must be positive")
    cfg["client"]["base_url"] = f"http://{cfg['server']['host']}:{cfg['server']['port']}/v1"
    return cfg
