"""Prefetch exact S2 requests from committed S1 records while S1 advances.

Only validated request caches are produced. The regular S2 stage still replays
the complete, verified S1 and commits its artifacts with unchanged validators.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from threading import Lock

from .contracts import KNOWLEDGE_SCHEMA, UNIT_SCHEMA_V2, validate_knowledge, validate_units
from .io import require, safe_file, save
from .pipeline import implementation_hash
from .prompts import KNOWLEDGE_VERSION, ROLE_C, ROLE_D
from .provider import Provider
from .s2 import unit_context
from .s2_state import TemporalMemory
from .s2_units import UnitRegistry


class S2Prefetch:
    def __init__(self, config, progress=print):
        self.config, self.progress = config, progress
        self.output = Path(config["output"])
        self.memory = TemporalMemory(config["episode_id"])
        self.units = UnitRegistry(config["episode_id"])
        self.knowledge = set()
        budgets = config.get("s2", {"knowledge_max_output_tokens": 2048, "unit_max_output_tokens": 4096})
        code = implementation_hash()
        self.providers = {role: Provider({**config["model"], "max_output_tokens": min(config["model"]["max_output_tokens"], budgets[key])},
            self.output.parent/"_category_knowledge_cache" if role == "C" else self.output/"cache_s2", code)
            for role, key in [("C", "knowledge_max_output_tokens"), ("D", "unit_max_output_tokens")]}
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="s2-prefetch")
        self.futures = []
        self.lock = Lock()
        self.submitted, self.completed = 0, 0
        self.failure = None
        self.state = "running"

    def report(self):
        with self.lock:
            save(self.output/"s2_prefetch.json", {"status": self.state, "submitted_frames": self.submitted,
                "completed_frames": self.completed, "new_transport_attempts": sum(p.new_transport_attempts for p in self.providers.values()),
                "failure": None if self.failure is None else {"type": type(self.failure).__name__, "detail": str(self.failure)[:1000]},
                "semantics": "exact_causal_S2_request_cache_only; final_S2_requires_full_S1_verification"})

    def submit(self, record, evidence):
        if self.failure is not None:
            raise self.failure
        with self.lock:
            order = self.submitted
            self.submitted += 1
        self.futures.append(self.pool.submit(self.process, order, deepcopy(record), deepcopy(evidence)))
        self.report()

    def process(self, order, record, evidence):
        if self.failure is not None:
            raise self.failure
        try:
            require(order == self.completed, "S2 prefetch lost causal ordering")
            require(record["image_ref"] == evidence["image_ref"], "S2 prefetch evidence mismatch")
            new_ids = self.memory.ingest(record)
            for category in sorted({o["category"] for o in record["observation"]["objects"]}):
                if category in self.knowledge:
                    continue
                self.progress(f"parallel C category={category}")
                self.providers["C"].observe(ROLE_C, KNOWLEDGE_SCHEMA,
                    {"role": "C", "category": category, "knowledge_version": KNOWLEDGE_VERSION}, [],
                    lambda doc, category=category: validate_knowledge(doc, category))
                self.knowledge.add(category)
            changes = {c["kind"] for c in record["observation"]["changes"] if c["epistemic_status"] != "unknown"}
            triggered = order == 0 or bool(new_ids) or bool(changes & {"opened", "closed", "revealed", "hidden"})
            if triggered and record["observation"]["objects"]:
                self.progress(f"parallel D frame={record['original_video_frame_index']}")
                context, entities, existing, facts = unit_context(self.memory, record, {record["image_ref"]: evidence}, self.units)
                images = [{"image_ref": record["image_ref"], "path": str(safe_file(self.output, evidence["image_path"])),
                           "sha256": evidence["image_sha256"]}]
                provider = self.providers["D"]
                result = provider.observe(ROLE_D, UNIT_SCHEMA_V2, context, images,
                    lambda doc: validate_units(doc, record["image_ref"], entities, existing, facts))
                self.units.ingest(result.document, self.memory.frames[-1], str((provider.root/result.cache_key/"ledger.json").resolve()))
            with self.lock:
                self.completed += 1
            self.report()
        except Exception as exc:
            self.failure, self.state = exc, "failed"
            self.report()
            raise

    def __enter__(self):
        self.report()
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc is not None:
            self.failure, self.state = exc, "failed"
        try:
            if exc is None:
                for future in self.futures:
                    future.result()
                self.state = "complete"
        finally:
            self.pool.shutdown(wait=True, cancel_futures=self.state == "failed")
            self.report()
        return False
