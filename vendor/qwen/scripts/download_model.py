#!/usr/bin/env python3
"""Pinned, resumable HF download; verify size, magic and full SHA256."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "models/qwen3.8-27b"
LOCK = ROOT / "configs/model.lock.json"
REPO = "ggml-org/Qwen3.8-27B-GGUF"
FILES = ["Qwen3.8-27B-Q4_K_M.gguf", "mmproj-Qwen3.8-27B-Q8_0.gguf"]

def main():
    DEST.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        lock = json.loads(LOCK.read_text())
    else:
        info = HfApi().model_info(REPO, files_metadata=True)
        siblings = {s.rfilename: s for s in info.siblings}
        lock = {"repo_id": REPO, "revision": info.sha, "files": []}
        for name in FILES:
            s = siblings[name]
            lock["files"].append({"name": name, "size": s.size, "sha256": s.lfs.sha256, "blob_id": s.blob_id})
        LOCK.write_text(json.dumps(lock, indent=2) + "\n")
    manifest = {**lock, "verified_at": datetime.now(timezone.utc).isoformat(), "verified_files": []}
    for spec in lock["files"]:
        path = DEST / spec["name"]
        if not path.exists() or path.stat().st_size != spec["size"]:
            print(f"Downloading {spec['name']} @ {lock['revision']}", flush=True)
            path = Path(hf_hub_download(repo_id=lock["repo_id"], filename=spec["name"], revision=lock["revision"], local_dir=DEST))
        with path.open("rb") as f:
            if f.read(4) != b"GGUF":
                raise RuntimeError(f"Not a GGUF file: {path}")
            f.seek(0)
            sha = hashlib.file_digest(f, "sha256").hexdigest()
        if path.stat().st_size != spec["size"] or sha != spec["sha256"]:
            raise RuntimeError(f"Verification failed: {path}; retained for diagnosis, do not load")
        manifest["verified_files"].append({**spec, "actual_bytes": path.stat().st_size, "actual_sha256": sha})
        print(f"Verified {path.name}: {path.stat().st_size} bytes, SHA256 {sha}", flush=True)
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

if __name__ == "__main__":
    main()
