"""Experimental TSM phase evidence fused with existing contact/visual boundaries.

EPIC labels do not define our event units. A take->put or open->close sequence
can be one interaction: only a completed phase followed by a new acquisition
is proposed as a restart. This is an explicit heuristic, not a trained segmenter.
"""
from pathlib import Path
import json

import numpy as np

from .actions import action_config
from .detector import sha256_file
from .events import contact_islands, event_config, segment_from_signals
from .video import read_jsonl, save_json, write_jsonl


PHASE_GROUPS = {"take": "acquire", "remove": "acquire", "lift": "acquire",
                "put": "place", "insert": "place", "drop": "place", "lower": "place",
                "open": "open", "close": "close"}
PHASE_MAXIMUM_GAP_SECONDS = .4
RESTART_MAXIMUM_GAP_SECONDS = 2.5
OPEN_CLOSE_MAXIMUM_SECONDS = 12.


def stable_action_phases(actions, config):
    phases, current = [], []
    def flush():
        if not current:
            return
        span = current[-1]["relative_seconds"] - current[0]["relative_seconds"]
        if len(current) >= 3 and span + 1e-6 >= config["minimum_action_persistence_seconds"]:
            phases.append({"phase": PHASE_GROUPS[current[0]["action_label"]],
                           "start_source_frame_id": current[0]["source_frame_id"],
                           "end_source_frame_id": current[-1]["source_frame_id"],
                           "start_seconds": current[0]["relative_seconds"],
                           "end_seconds": current[-1]["relative_seconds"],
                           "observed_windows": len(current),
                           "mean_top_verb_probability": float(np.mean([r["top_verbs"][0]["probability"] for r in current])),
                           "observed_labels": sorted(set(r["action_label"] for r in current))})
        current.clear()
    for row in actions:
        if current and row["relative_seconds"] - current[-1]["relative_seconds"] > PHASE_MAXIMUM_GAP_SECONDS + 1e-6:
            flush()
        label = row["action_label"]
        if label not in PHASE_GROUPS:
            # An unrelated confident label ends the phase. Uncertainty may only
            # bridge a short gap, and never contributes an observed window.
            if label is not None:
                flush()
            continue
        if current and PHASE_GROUPS[label] != PHASE_GROUPS[current[-1]["action_label"]]:
            flush()
        current.append(row)
    flush()
    return phases


def _observed_release(rows, start, end, hand_threshold):
    local = [r for r in rows if start <= r["relative_seconds"] <= end]
    longest, active, previous_time = 0., None, None
    contacts = []
    for row in local:
        t = row["relative_seconds"]
        has_contact = row["status"] == "ok" and any(
            h["score"] >= hand_threshold and h["contact_state_id"] in (3, 4)
            and h.get("matched_object_id") is not None for h in row["hands"])
        contacts.append(has_contact)
        # Missing detections/errors cannot prove that a hand released a target.
        has_hand = row["status"] == "ok" and any(h["score"] >= hand_threshold for h in row["hands"])
        if has_hand and not has_contact:
            if active is None or (previous_time is not None and t - previous_time > .1):
                active = t
            longest = max(longest, t - active)
        else:
            active = None
        previous_time = t
    # Also require actual contact observations on both sides of this region.
    return longest if any(contacts[:len(contacts)//2]) and any(contacts[len(contacts)//2:]) else 0.


def action_boundary_candidates(actions, detections, visual_decisions, config, visual_config):
    phases = stable_action_phases(actions, config)
    protected, candidates = [], []
    for i, phase in enumerate(phases):
        if phase["phase"] == "close":
            opening = next((p for p in reversed(phases[:i]) if p["phase"] == "open"), None)
            if opening and phase["end_seconds"] - opening["start_seconds"] <= OPEN_CLOSE_MAXIMUM_SECONDS:
                reacquired = any(p["phase"] == "acquire" and opening["start_seconds"] < p["start_seconds"] < phase["start_seconds"]
                                 for p in phases)
                if not reacquired:
                    protected.append({"start_source_frame_id": opening["start_source_frame_id"],
                                      "end_source_frame_id": phase["end_source_frame_id"],
                                      "reason": "open_through_close_without_stable_reacquisition"})
        if i == 0:
            continue
        previous = phases[i - 1]
        restart = previous["phase"] in ("place", "close") and phase["phase"] in ("acquire", "open")
        if not restart or phase["start_seconds"] - previous["end_seconds"] > RESTART_MAXIMUM_GAP_SECONDS:
            continue
        t = phase["start_seconds"]
        radius = config["boundary_contact_window_seconds"]
        release = _observed_release(detections, t - radius, t + radius, visual_config["hand_score_min"])
        resets = [d for d in visual_decisions if abs(d["relative_seconds"] - t) <= radius and d["reset"]]
        supported = bool(resets or release + 1e-6 >= config["minimum_observed_release_seconds"])
        candidates.append({"source_frame_id": phase["start_source_frame_id"], "relative_seconds": t,
                           "previous_phase": previous["phase"], "next_phase": phase["phase"],
                           "observed_release_seconds": release, "hand_reset_support": bool(resets),
                           "supported": supported,
                           "score": min(previous["mean_top_verb_probability"], phase["mean_top_verb_probability"]),
                           "reason": "tsm_completed_phase_then_new_operation"})
    return phases, protected, candidates


def fuse_action_events(samples, detections, actions, width, height, config, visual_config):
    _, visual_decisions = segment_from_signals(samples, detections, width, height, visual_config)
    phases, protected, candidates = action_boundary_candidates(actions, detections, visual_decisions, config, visual_config)
    boundaries = []
    for row in visual_decisions:
        if row["selected"]:
            boundaries.append({"source_frame_id": row["source_frame_id"], "relative_seconds": row["relative_seconds"],
                               "score": row["target_cosine_distance"], "reason": "original_visual_boundary"})
    boundaries += [dict(c) for c in candidates if c["supported"]]
    vetoed, kept = [], []
    for b in sorted(boundaries, key=lambda r: (-r["score"], r["source_frame_id"])):
        cycle = next((p for p in protected if p["start_source_frame_id"] < b["source_frame_id"] <= p["end_source_frame_id"]), None)
        if cycle:
            vetoed.append(dict(b, veto_reason=cycle["reason"]))
        elif all(abs(b["relative_seconds"] - old["relative_seconds"]) >= visual_config["minimum_boundary_gap_seconds"] for old in kept):
            kept.append(b)
    events, selected = [], []
    for island in contact_islands(detections, visual_config):
        start, end = island[0]["source_frame_id"], island[-1]["source_frame_id"]
        chosen, previous = [], start
        for b in sorted(kept, key=lambda r: r["source_frame_id"]):
            f = b["source_frame_id"]
            if not start < f <= end:
                continue
            before = [r for r in island if previous <= r["source_frame_id"] < f]
            after = [r for r in island if f <= r["source_frame_id"] <= end]
            if (len(before) >= visual_config["minimum_contact_frames"] and len(after) >= visual_config["minimum_contact_frames"]
                    and before[-1]["relative_seconds"] - before[0]["relative_seconds"] >= visual_config["minimum_event_seconds"]
                    and after[-1]["relative_seconds"] - after[0]["relative_seconds"] >= visual_config["minimum_event_seconds"]):
                chosen.append(b)
                selected.append(b)
                previous = f
        edges = [start] + [b["source_frame_id"] for b in chosen] + [end + 1]
        for i, (first, following) in enumerate(zip(edges, edges[1:])):
            rows = [r for r in island if first <= r["source_frame_id"] < following]
            phase_labels = sorted(set(p["phase"] for p in phases if p["end_source_frame_id"] >= first and p["start_source_frame_id"] < following))
            events.append({"event_id": f"event_{len(events)+1:03d}",
                           "start_source_frame_id": rows[0]["source_frame_id"],
                           "end_source_frame_id": rows[-1]["source_frame_id"],
                           "start_seconds": rows[0]["relative_seconds"], "end_seconds": rows[-1]["relative_seconds"],
                           "contact_frame_count": len(rows), "action_phases": phase_labels,
                           "boundary_reason": "contact_island_start" if i == 0 else chosen[i-1]["reason"],
                           "status": "experimental_tsm_contact_hypothesis", "manually_verified": False})
    diagnostics = {"phases": phases, "protected_cycles": protected, "action_boundary_candidates": candidates,
                   "vetoed_boundaries": vetoed, "selected_boundaries": selected}
    return events, diagnostics


def segment_action_events(run, evidence, actions_run, output, config=None, visual_config=None,
                          decoder="cycles", decoder_config=None):
    run, evidence, actions_run, output = [Path(p).resolve() for p in (run, evidence, actions_run, output)]
    config, visual_config = config or action_config(), visual_config or event_config()
    from .action_cycles import cycle_config, fuse_cycle_events
    if decoder not in ("legacy", "cycles"):
        raise ValueError("Unknown action-event decoder")
    decoder_config = decoder_config or cycle_config()
    if output in (run, evidence, actions_run):
        raise ValueError("Use a separate action-event output directory")
    am = json.loads((actions_run / "manifest.json").read_text())
    ar = json.loads((actions_run / "report.json").read_text())
    em = json.loads((evidence / "manifest.json").read_text())
    er = json.loads((evidence / "report.json").read_text())
    for m in (am, em):
        for field, filename in (("detection_manifest_sha256", "manifest.json"), ("detections_sha256", "detections.jsonl"),
                                ("frame_index_sha256", "frame_index.jsonl")):
            if m[field] != sha256_file(run / filename):
                raise ValueError("Action or visual evidence belongs to a different recording")
        if m.get("reference_annotations_used") or m.get("vlm_used"):
            raise ValueError("Expected reference-free, VLM-free evidence")
    if ar["actions_sha256"] != sha256_file(actions_run / "actions.jsonl") or er["signals_sha256"] != sha256_file(evidence / "signals.jsonl"):
        raise ValueError("Action or visual evidence changed")
    if ar["labels_sha256"] != sha256_file(actions_run / "labels.json"):
        raise ValueError("Action labels changed")
    if any(am["configuration"][k] != config[k] for k in config):
        raise ValueError("Action configuration differs from extracted scores")
    if any(em["configuration"][k] != visual_config[k] for k in em["configuration"]):
        raise ValueError("Visual evidence configuration differs")
    identity = {"stage": "experimental_tsm_contact_event_segmentation", "reference_annotations_used": False,
                "vlm_used": False, "completion_signals_used": False, "action_configuration": config,
                "visual_configuration": visual_config, "phase_groups": PHASE_GROUPS,
                "decoder": decoder, "decoder_configuration": decoder_config if decoder == "cycles" else None,
                "phase_maximum_gap_seconds": PHASE_MAXIMUM_GAP_SECONDS,
                "restart_maximum_gap_seconds": RESTART_MAXIMUM_GAP_SECONDS,
                "open_close_maximum_seconds": OPEN_CLOSE_MAXIMUM_SECONDS,
                "actions_sha256": ar["actions_sha256"], "visual_signals_sha256": er["signals_sha256"],
                "actions_manifest_sha256": sha256_file(actions_run / "manifest.json"),
                "visual_manifest_sha256": sha256_file(evidence / "manifest.json"),
                "implementation_sha256": {p: sha256_file(Path(__file__).with_name(p))
                    for p in ("action_events.py", "events.py", "action_cycles.py")}}
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists() and json.loads((output / "manifest.json").read_text()) != identity:
        raise ValueError("Action-event inputs changed; use a new output directory")
    save_json(output / "manifest.json", identity)
    rows = read_jsonl(run / "detections.jsonl")
    index = read_jsonl(run / "frame_index.jsonl")
    if [r["source_frame_id"] for r in rows] != [r["source_frame_id"] for r in index]:
        raise ValueError("Detection/frame-index mismatch")
    arguments = (read_jsonl(evidence / "signals.jsonl"), rows, read_jsonl(actions_run / "actions.jsonl"),
                 index[0]["width"], index[0]["height"], config, visual_config)
    events, diagnostics = (fuse_cycle_events(*arguments, decoder_config) if decoder == "cycles"
                           else fuse_action_events(*arguments))
    save_json(output / "predictions.json", {"reference_annotations_used": False, "vlm_used": False,
              "event_source": "tsm_contact_fusion", "predictions": events})
    save_json(output / "action_diagnostics.json", diagnostics)
    report = {"event_count": len(events), "source_frame_count": len(rows), "reference_annotations_used": False,
              "vlm_used": False, "completion_signals_used": False,
              "detection_manifest_sha256": sha256_file(run / "manifest.json"),
              "detections_sha256": sha256_file(run / "detections.jsonl"),
              "predictions_sha256": sha256_file(output / "predictions.json"), "manually_verified": False}
    save_json(output / "report.json", report)
    return report
