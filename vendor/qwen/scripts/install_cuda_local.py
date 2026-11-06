#!/usr/bin/env python3
"""Install only NVIDIA's compiler/runtime/BLAS/headers, without root or drivers."""
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://developer.download.nvidia.com/compute/cuda/redist/"
META = "redistrib_12.8.1.json"
target = ROOT / ".toolchains/cuda-12.8"
cache = ROOT / ".toolchains/downloads"
target.mkdir(parents=True, exist_ok=True)
cache.mkdir(parents=True, exist_ok=True)
with urllib.request.urlopen(BASE + META, timeout=60) as r:
    meta = json.load(r)
(ROOT / "docs/cuda-source.json").write_text(json.dumps(meta, indent=2))
for component in ["cuda_nvcc", "cuda_cudart", "cuda_cccl", "libcublas"]:
    spec = meta[component]["linux-x86_64"]
    archive = cache / Path(spec["relative_path"]).name
    complete = False
    if archive.exists() and archive.stat().st_size == int(spec["size"]):
        with archive.open("rb") as f:
            complete = hashlib.file_digest(f, "sha256").hexdigest() == spec["sha256"]
        if not complete:
            raise RuntimeError(f"Existing full archive has incorrect SHA256: {archive}")
    if not complete:
        print(f"Downloading {component}: {spec['size']} bytes", flush=True)
        subprocess.run(["curl", "-fL", "--retry", "5", "--connect-timeout", "30", "-C", "-", BASE + spec["relative_path"], "-o", str(archive)], check=True)
    with archive.open("rb") as f:
        sha = hashlib.file_digest(f, "sha256").hexdigest()
    if sha != spec["sha256"]:
        raise RuntimeError(f"SHA256 mismatch: {archive}")
    subprocess.run(["tar", "-xJf", str(archive), "--strip-components=1", "-C", str(target)], check=True)
if not (target / "lib64").exists():
    (target / "lib64").symlink_to("lib")
subprocess.run([str(target / "bin/nvcc"), "--version"], check=True)
