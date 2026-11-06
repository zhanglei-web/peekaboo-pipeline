#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="$ROOT/third_party/llama.cpp"
mkdir -p "$ROOT/logs"
COMMIT="$(cat "$ROOT/configs/backend.commit")"
if [[ ! -d "$SOURCE/.git" ]]; then
  git clone --no-checkout https://github.com/ggml-org/llama.cpp.git "$SOURCE"
  git -C "$SOURCE" checkout --detach "$COMMIT"
fi
[[ "$(git -C "$SOURCE" rev-parse HEAD)" == "$COMMIT" ]] || { echo 'Backend commit differs from lock; refusing implicit update' >&2; exit 1; }
if [[ -x "$ROOT/.toolchains/cuda-12.8/bin/nvcc" ]]; then
  export CUDACXX="$ROOT/.toolchains/cuda-12.8/bin/nvcc"
  export PATH="$ROOT/.toolchains/cuda-12.8/bin:$PATH"
  export LD_LIBRARY_PATH="$ROOT/.toolchains/cuda-12.8/lib:${LD_LIBRARY_PATH:-}"
elif command -v nvcc >/dev/null; then
  export CUDACXX="$(command -v nvcc)"
else
  echo 'Missing CUDA Toolkit; run python3 scripts/install_cuda_local.py' >&2; exit 1
fi
# Query the actual device, rather than guessing an architecture from its marketing name.
ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.[:space:]')"
[[ "$ARCH" =~ ^[0-9]+$ ]] || { echo 'Cannot determine CUDA compute capability' >&2; exit 1; }
"$CUDACXX" --list-gpu-code | grep -qx "sm_$ARCH" || { echo "Toolkit does not support sm_$ARCH" >&2; exit 1; }
{
  echo "Commit: $COMMIT; architecture sm_$ARCH (nvidia-smi query)"
  "$CUDACXX" --version
  cmake -S "$SOURCE" -B "$SOURCE/build" -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$ARCH" -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
  cmake --build "$SOURCE/build" --config Release -j "${BUILD_JOBS:-8}" --target llama-server
  "$SOURCE/build/bin/llama-server" --version
  "$SOURCE/build/bin/llama-server" --list-devices
  "$SOURCE/build/bin/llama-server" --help > "$ROOT/logs/backend-help.txt"
} 2>&1 | tee "$ROOT/logs/backend-build.log"
