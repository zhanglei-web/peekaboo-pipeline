#!/usr/bin/env bash
set -eo pipefail
PIPELINE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source /opt/ros/jazzy/setup.bash
exec /usr/bin/python3 "$PIPELINE_ROOT/pipeline.py" "$@"
