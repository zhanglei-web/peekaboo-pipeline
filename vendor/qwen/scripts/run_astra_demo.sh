#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/.venv/bin/python" -I - <<'PY'
import json, subprocess
from local_vlm.config import ROOT, load_config
from local_vlm.server import PIDFILE, identity
expected = load_config('configs/astra-pipeline.yaml')['server']
record = json.loads(PIDFILE.read_text()) if PIDFILE.exists() else None
if record and identity(record):
    if record['config']['server'] != expected:
        raise SystemExit('An existing local server uses different settings. Finish its requests before switching to configs/astra-pipeline.yaml.')
else:
    subprocess.run([str(ROOT/'scripts/start_server.sh'), '--config', 'configs/astra-pipeline.yaml'], check=True)
PY
python3 "$ROOT/integrations/astra_pipeline/run.py" run \
  --config "$ROOT/integrations/astra_pipeline/config/demo_5.local.json" --all-selected-frames
python3 "$ROOT/integrations/astra_pipeline/run.py" validate \
  --config "$ROOT/integrations/astra_pipeline/config/demo_5.local.json"
