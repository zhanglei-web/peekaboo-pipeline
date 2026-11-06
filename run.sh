#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/jazzy/setup.bash
exec /home/baai/zl/egomemory_ws/.venv/bin/python -u /home/baai/zl/todo/11.4/processing/run_batch.py
