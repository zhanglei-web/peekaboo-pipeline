#!/usr/bin/env bash
set -u
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/docs" "$ROOT/logs"
{
  date --iso-8601=seconds
  cat /etc/os-release
  uname -a
  lscpu
  free -h
  df -h "$ROOT"
  nvidia-smi
  nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,compute_cap --format=csv
  for command in python3 git cmake gcc g++ nvcc; do
    if command -v "$command" >/dev/null; then "$command" --version; else echo "$command: not on PATH"; fi
  done
  if [[ -x "$ROOT/.toolchains/cuda-12.8/bin/nvcc" ]]; then "$ROOT/.toolchains/cuda-12.8/bin/nvcc" --version; fi
  curl -IL --connect-timeout 15 --max-time 30 https://huggingface.co/api/models/ggml-org/Qwen3.8-27B-GGUF
} 2>&1 | tee "$ROOT/logs/environment-$(date +%Y%m%d-%H%M%S).log"
