import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import psutil
import requests
from .config import ROOT, load_config, resolve

PIDFILE = ROOT / "logs/server.pid.json"

def identity(record):
    try:
        proc = psutil.Process(record["pid"])
        return (abs(proc.create_time() - record["create_time"]) < 0.01
                and proc.status() != psutil.STATUS_ZOMBIE
                and str(Path(proc.exe()).resolve()) == record["binary"]
                and proc.cmdline() == record["command"])
    except (psutil.Error, KeyError, OSError):
        return False

def environment():
    env = {k: v for k, v in os.environ.items() if not k.startswith("LLAMA_ARG_")}
    lib = ROOT / ".toolchains/cuda-12.8/lib"
    if lib.is_dir():
        env["LD_LIBRARY_PATH"] = str(lib) + ":" + env.get("LD_LIBRARY_PATH", "")
    return env

def command(cfg):
    s = cfg["server"]
    binary = resolve(s["binary"])
    if not binary.is_file():
        raise FileNotFoundError(f"Build backend first: {binary}")
    for key in ["model", "mmproj"]:
        with resolve(s[key]).open("rb") as f:
            if f.read(4) != b"GGUF":
                raise ValueError(f"Invalid GGUF: {s[key]}")
    cmd = [str(binary), "-m", str(resolve(s["model"])), "--mmproj", str(resolve(s["mmproj"]))]
    for flag, key in [("--host", "host"), ("--port", "port"), ("--alias", "alias"),
                      ("--ctx-size", "ctx_size"), ("--parallel", "parallel"), ("--n-gpu-layers", "n_gpu_layers"),
                      ("--batch-size", "batch_size"), ("--ubatch-size", "ubatch_size"),
                      ("--flash-attn", "flash_attn"), ("--image-max-tokens", "image_max_tokens"),
                      ("--log-verbosity", "log_verbosity"), ("--reasoning", "reasoning"), ("--cors-origins", "cors_origins")]:
        cmd.extend([flag, str(s[key])])
    for flag, key in [("--threads", "threads"), ("--threads-batch", "threads_batch"),
                      ("--cache-ram", "cache_ram")]:
        if key in s:
            if not isinstance(s[key], int) or isinstance(s[key], bool) or s[key] <= 0:
                raise ValueError(f"server.{key} must be a positive integer")
            cmd.extend([flag, str(s[key])])
    for flag, key in [("--cache-type-k", "cache_type_k"), ("--cache-type-v", "cache_type_v")]:
        if key in s:
            if s[key] not in {"f16", "q8_0"}:
                raise ValueError(f"unsupported server.{key}: {s[key]}")
            cmd.extend([flag, s[key]])
    cmd.extend(["--fit", "off", "--no-context-shift", "--jinja", "--reasoning-format", "deepseek"])
    if not s["reasoning_preserve"]:
        cmd.append("--no-reasoning-preserve")
    if not s["mmproj_offload"]:
        cmd.append("--no-mmproj-offload")
    help_text = subprocess.run([str(binary), "--help"], env=environment(), capture_output=True, text=True, check=True).stdout
    for arg in cmd:
        if arg.startswith("--") and arg not in help_text:
            raise RuntimeError(f"Unsupported backend flag: {arg}")
    return cmd

def stop():
    if not PIDFILE.exists():
        print("No project-owned server PID file")
        return
    record = json.loads(PIDFILE.read_text())
    if not identity(record):
        PIDFILE.unlink()
        print("Stale PID record removed; no process signalled")
        return
    proc = psutil.Process(record["pid"])
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except psutil.TimeoutExpired:
        if identity(record):
            proc.kill()
            proc.wait(timeout=10)
    PIDFILE.unlink(missing_ok=True)
    print(f"Stopped project server PID {record['pid']}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["start", "stop", "status"])
    parser.add_argument("--foreground", action="store_true")
    parser.add_argument("--config")
    args = parser.parse_args()
    (ROOT / "logs").mkdir(exist_ok=True)
    with (ROOT / "logs/server-control.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.action == "stop":
            stop()
            return
        cfg = load_config(args.config)
        base = cfg["client"]["base_url"].removesuffix("/v1")
        if args.action == "status":
            if not PIDFILE.exists() or not identity(json.loads(PIDFILE.read_text())):
                raise RuntimeError("No confirmed project-owned server")
            for endpoint in ["/health", "/v1/models"]:
                response = requests.get(base + endpoint, timeout=10)
                response.raise_for_status()
                print(json.dumps(response.json(), ensure_ascii=False))
            return
        if PIDFILE.exists() and identity(json.loads(PIDFILE.read_text())):
            raise RuntimeError("Project server already running; use status or stop")
        s = cfg["server"]
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((s["host"], s["port"]))
        cmd = command(cfg)
        if args.foreground:
            def interrupted(signum, frame):
                raise KeyboardInterrupt
            signal.signal(signal.SIGTERM, interrupted)
        stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{time.time_ns() % 1_000_000_000:09d}"
        logfile = ROOT / "logs" / f"server-{stamp}.log"
        with logfile.open("a") as log:
            proc = subprocess.Popen(cmd, cwd=ROOT, env=environment(), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        record = {"pid": proc.pid, "create_time": psutil.Process(proc.pid).create_time(), "binary": str(resolve(s["binary"]).resolve()),
                  "command": cmd, "log": str(logfile), "config": cfg}
        PIDFILE.write_text(json.dumps(record, indent=2))
        (ROOT / "logs" / f"server-{stamp}.json").write_text(json.dumps(record, indent=2))
        print(f"Starting PID {proc.pid}; log: {logfile}", flush=True)
        try:
            deadline = time.monotonic() + s["startup_timeout"]
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(f"Server exited {proc.returncode}; inspect {logfile}")
                try:
                    r = requests.get(base + "/health", timeout=2)
                    if r.ok and r.json().get("status") == "ok":
                        print(f"Healthy: {base}/v1; PID {proc.pid}", flush=True)
                        break
                except (requests.RequestException, ValueError):
                    pass
                time.sleep(1)
            else:
                raise TimeoutError(f"Startup timeout; inspect {logfile}")
            fcntl.flock(lock, fcntl.LOCK_UN)
            if args.foreground:
                sys.exit(proc.wait())
        except BaseException:
            if proc.poll() is None and identity(record):
                proc.terminate()
                proc.wait(timeout=30)
            if PIDFILE.exists() and json.loads(PIDFILE.read_text()).get("pid") == proc.pid:
                PIDFILE.unlink()
            raise

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted; project-owned child stopped.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
