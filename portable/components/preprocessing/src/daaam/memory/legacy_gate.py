"""Explicit opt-in for historical entrypoints outside the frozen comparison."""
import os


def require_legacy_opt_in(name: str) -> None:
    if os.environ.get("DAAAM_ENABLE_ISOLATED_LEGACY") != "1":
        raise SystemExit(
            f"{name} is isolated from the Ego causal-memory freeze. "
            "Use scripts/run_frozen_ego_memory.py for comparisons. "
            "Historical runs require DAAAM_ENABLE_ISOLATED_LEGACY=1 and are "
            "not certified cutoff-safe or budget-comparable."
        )
