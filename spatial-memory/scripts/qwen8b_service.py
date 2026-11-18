"""Manage a verified local Qwen3-VL-8B-Instruct GGUF service, never a model alias swap."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess

import requests

MODEL = "Qwen3-VL-8B-Instruct"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "status", "stop"))
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--llama-server", type=Path)
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--state", type=Path, default=Path("runs/qwen8b_service"))
    args = parser.parse_args()
    state = args.state.resolve()
    root = args.model_root.resolve()
    state.mkdir(parents=True, exist_ok=True)
    pid = None
    try:
        candidate = int((state / "server.pid").read_text())
        command = Path(f"/proc/{candidate}/cmdline").read_bytes().decode().split("\0")
        if command[command.index("--alias") + 1] == MODEL and str(root) in command[command.index("--model") + 1]:
            pid = candidate
    except (ValueError, OSError):
        pass
    if args.action == "stop":
        if pid:
            os.kill(pid, signal.SIGTERM)
        print(json.dumps({"stopped_pid": pid}))
        return
    if args.action == "status":
        session = requests.Session()
        session.trust_env = False
        try:
            health = session.get(f"http://127.0.0.1:{args.port}/health", timeout=3).json()
        except requests.RequestException:
            health = {"status": "unavailable"}
        print(json.dumps({"pid": pid, "model": MODEL, "health": health}))
        return
    if pid:
        print(json.dumps({"already_running": pid}))
        return
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["upstream"] != "Qwen/Qwen3-VL-8B-Instruct":
        raise ValueError("Model manifest is not the requested official 8B vision model")
    verified = []
    for record in manifest["files"]:
        path = root / record["rfilename"]
        if Path(str(path) + ".aria2").exists() or path.stat().st_size != record["size"]:
            raise ValueError(f"Incomplete weight file: {path.name}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for data in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(data)
        actual = digest.hexdigest()
        if actual != record["lfs"]["sha256"]:
            raise ValueError(f"Weight hash mismatch: {path.name}")
        verified.append({"file": path.name, "sha256": actual})
    (state / "verified_weights.json").write_text(json.dumps({"upstream": manifest["upstream"],
        "revision": manifest["revision"], "files": verified}, indent=2) + "\n")
    # Credentials live in the model directory, excluded from the source repository.
    key = root / "secrets/api-key.txt"
    key.parent.mkdir(parents=True, exist_ok=True)
    if not key.exists():
        key.write_text("EMPTY\n")
        key.chmod(0o600)
    config = {"host": "127.0.0.1", "port": args.port, "model_alias": MODEL,
              "api_key_file": "secrets/api-key.txt"}
    (root / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    if args.llama_server is None:
        raise ValueError("--llama-server is required to start")
    binary = args.llama_server.resolve()
    model = next(r["rfilename"] for r in manifest["files"] if not r["rfilename"].startswith("mmproj"))
    projector = next(r["rfilename"] for r in manifest["files"] if r["rfilename"].startswith("mmproj"))
    command = [str(binary), "--model", str(root / model), "--mmproj", str(root / projector),
               "--alias", MODEL, "--host", "127.0.0.1", "--port", str(args.port),
               "--n-gpu-layers", "all", "--parallel", "2", "--ctx-size", "65536",
               "--kv-unified-per-slot", "32768", "--flash-attn", "on", "--cont-batching", "--jinja",
               "--image-min-tokens", "1024", "--image-max-tokens", "1024",
               "--chat-template-kwargs", '{"enable_thinking":false}',
               "--api-key-file", str(key)]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(binary.parent) + ":/usr/local/cuda-13.1/lib64:" + env.get("LD_LIBRARY_PATH", "")
    with (state / "server.log").open("a") as log:
        process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=log, start_new_session=True)
    (state / "server.pid").write_text(str(process.pid) + "\n")
    print(json.dumps({"pid": process.pid, "model": MODEL, "verified": verified,
                      "health": "check status before inference"}))


if __name__ == "__main__":
    main()
