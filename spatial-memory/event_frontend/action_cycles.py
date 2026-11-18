"""Conservative action-cycle decoder with explicit observed withdrawal evidence.

TSM contextual views improve action visibility. Visual target changes alone do
not terminate an operation. Missing hands/processing failures remain unknown.
The bottom-entry withdrawal rule is a configurable image-plane cue for these
ego cameras, not a reconstructed three-dimensional lifting action.
"""
from pathlib import Path
import json
import math

import numpy as np

from .action_events import _observed_release, action_boundary_candidates
from .events import contact_islands, primary_hand_trajectory, segment_from_signals


def cycle_config(path=None):
    result = json.loads(Path(__file__).with_name("action_cycle_config.json").read_text())
    if path:
        supplied = json.loads(Path(path).read_text())
        if set(supplied) - set(result):
            raise ValueError("Unknown action-cycle configuration keys")
        result.update(supplied)
    for key, value in result.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid action-cycle setting: {key}")
    for key in ("minimum_close_probability", "withdrawal_edge_center_y", "minimum_reapproach_y_distance",
                "minimum_gap_hand_observation_fraction"):
        if result[key] > 1:
            raise ValueError(f"Action-cycle fraction must be at most one: {key}")
    return result


def withdrawal_reapproaches(samples, detections, width, height, phases, config, visual_config):
    positions, confident = primary_hand_trajectory(samples, detections, width, height,
                                                  visual_config["hand_score_min"])
    # Convert associated hand centers back to each actual image: accumulated
    # camera transforms may move stabilized coordinates outside [0, 1].
    raw = np.full_like(positions, np.nan)
    for i, (point, sample) in enumerate(zip(positions, samples)):
        if confident[i] and sample.get("camera_fit_inlier_fraction", 0.) >= .5:
            mapped = np.asarray(sample["camera_transform_from_first"]) @ np.r_[point * [width, height], 1.]
            raw[i] = mapped[:2] / [width, height]
    candidates, group, pending = [], [], None
    edge = config["withdrawal_edge_center_y"]
    for i, sample in enumerate(samples):
        t = sample["relative_seconds"]
        valid = np.isfinite(raw[i]).all()
        if group and t - samples[group[-1]]["relative_seconds"] > config["withdrawal_maximum_observation_gap_seconds"] + 1e-6:
            group.clear()
        if valid and raw[i, 1] >= edge:
            group.append(i)
            if len(group) >= 4 and t - samples[group[0]]["relative_seconds"] >= config["minimum_withdrawal_seconds"]:
                pending = list(group)
            continue
        if not pending:
            continue
        last = pending[-1]
        if t - samples[last]["relative_seconds"] > config["maximum_reapproach_seconds"]:
            pending, group = None, []
            continue
        if not valid or raw[i, 1] > edge - config["minimum_reapproach_y_distance"]:
            continue
        # A manipulation can include withdrawing and then closing a drawer.
        # Do not split the approach into a reliably observed closing phase.
        closing = any(p["phase"] == "close"
                      and p["mean_top_verb_probability"] >= config["minimum_close_probability"]
                      and p["start_seconds"] - config["close_guard_seconds"] <= t <= p["end_seconds"] + config["close_guard_seconds"]
                      for p in phases)
        after = [s for s in samples[i:i+4] if s["relative_seconds"] - t <= .31]
        actual_contact = sum(s.get("contact") and s["status"] == "ok" for s in after) >= 2
        reliable_target_fraction = np.mean([samples[j].get("reliable_target", False) for j in pending])
        # A small reliably tracked target may still be held while moving at the
        # border; this cannot establish hand withdrawal from that target.
        held_target_ambiguous = reliable_target_fraction >= .7
        supported = actual_contact and not closing and not held_target_ambiguous
        candidates.append({"source_frame_id": sample["source_frame_id"], "relative_seconds": t,
                           "reason": "observed_border_withdrawal_then_reapproach", "supported": bool(supported),
                           "score": .85, "withdrawal_start_source_frame_id": samples[pending[0]]["source_frame_id"],
                           "withdrawal_end_source_frame_id": samples[last]["source_frame_id"],
                           "observed_border_samples": len(pending),
                           "image_plane_y_displacement": float(np.median(raw[pending, 1]) - raw[i, 1]),
                           "closing_phase_guard": closing, "held_target_ambiguous": bool(held_target_ambiguous),
                           "actual_contact_on_reapproach": actual_contact,
                           "missing_hand_is_release": False})
        pending, group = None, []
    return candidates


def action_supported_islands(detections, phases, boundaries, config, visual_config):
    """Bridge an ongoing action across a short contact dropout, not a restart.

Only real contacts remain in the returned islands. Intermediate frames need
observed hands; a video/detector failure alone cannot create a bridge.
"""
    islands, bridges = [], []
    for island in contact_islands(detections, visual_config):
        if not islands:
            islands.append(list(island))
            continue
        before, after = islands[-1][-1], island[0]
        a, b = before["relative_seconds"], after["relative_seconds"]
        active = [p for p in phases if p["phase"] in ("acquire", "open") and
                  p["start_seconds"] <= a <= p["end_seconds"] and
                  b - p["end_seconds"] <= config["maximum_action_continuation_seconds"]]
        gap = [r for r in detections if a < r["relative_seconds"] < b]
        observed = sum(r["status"] == "ok" and any(h["score"] >= visual_config["hand_score_min"]
                       for h in r["hands"]) for r in gap) / max(1, len(gap))
        restarting = any(a < p["relative_seconds"] <= b for p in boundaries)
        completed = any(p["phase"] in ("close", "place") and a < p["end_seconds"] < b for p in phases)
        if (b - a <= config["maximum_action_supported_contact_gap_seconds"] and active
                and observed >= config["minimum_gap_hand_observation_fraction"] and not restarting and not completed):
            bridges.append({"before_source_frame_id": before["source_frame_id"],
                            "after_source_frame_id": after["source_frame_id"],
                            "gap_seconds": b-a, "observed_hand_fraction": observed,
                            "ongoing_action": active[-1], "gap_frames_counted_as_contacts": False})
            islands[-1].extend(island)
        else:
            islands.append(list(island))
    return islands, bridges


def fuse_cycle_events(samples, detections, actions, width, height, config, visual_config, decoder_config):
    _, visual_decisions = segment_from_signals(samples, detections, width, height, visual_config)
    phases, protected, action_candidates = action_boundary_candidates(actions, detections, visual_decisions, config, visual_config)
    protected = [p for p in protected if any(phase["phase"] == "close"
                 and phase["end_source_frame_id"] == p["end_source_frame_id"]
                 and phase["mean_top_verb_probability"] >= decoder_config["minimum_close_probability"]
                 for phase in phases)]
    # A weak "close" classification is not proof that the current event ended.
    for candidate in action_candidates:
        previous = [p for p in phases if p["phase"] == candidate["previous_phase"]
                    and p["end_seconds"] <= candidate["relative_seconds"]]
        strong_close = (candidate["previous_phase"] != "close" or (previous and
                        previous[-1]["mean_top_verb_probability"] >= decoder_config["minimum_close_probability"]))
        candidate["close_confidence_sufficient"] = bool(strong_close)
        candidate["supported"] = candidate["supported"] and bool(strong_close)
    withdrawal = withdrawal_reapproaches(samples, detections, width, height, phases, decoder_config, visual_config)
    # A separate observed retreat/reapproach defeats broad open-close pairing.
    protected = [p for p in protected if not any(w["supported"] and
                 p["start_source_frame_id"] < w["source_frame_id"] <= p["end_source_frame_id"] for w in withdrawal)]
    boundaries = [dict(c) for c in action_candidates + withdrawal if c["supported"]]
    rejected = []
    for row in visual_decisions:
        if not row["selected"]:
            continue
        radius = config["boundary_contact_window_seconds"]
        release = _observed_release(detections, row["relative_seconds"] - radius,
                                     row["relative_seconds"] + radius, visual_config["hand_score_min"])
        candidate = {"source_frame_id": row["source_frame_id"], "relative_seconds": row["relative_seconds"],
                     "score": row["target_cosine_distance"], "reason": "visual_change_with_observed_release",
                     "observed_release_seconds": release}
        if release + 1e-6 >= config["minimum_observed_release_seconds"]:
            boundaries.append(candidate)
        else:
            rejected.append(dict(candidate, veto_reason="appearance_or_stationary_hand_without_observed_completion"))
    kept, vetoed = [], []
    for b in sorted(boundaries, key=lambda r: (-r["score"], r["source_frame_id"])):
        cycle = next((p for p in protected if p["start_source_frame_id"] < b["source_frame_id"] <= p["end_source_frame_id"]), None)
        if cycle:
            vetoed.append(dict(b, veto_reason=cycle["reason"]))
        elif all(abs(b["relative_seconds"] - old["relative_seconds"]) >= visual_config["minimum_boundary_gap_seconds"] for old in kept):
            kept.append(b)
    events, selected = [], []
    islands, bridges = action_supported_islands(detections, phases, kept, decoder_config, visual_config)
    for island in islands:
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
            duration = rows[-1]["relative_seconds"] - rows[0]["relative_seconds"]
            events.append({"event_id": f"event_{len(events)+1:03d}",
                           "start_source_frame_id": rows[0]["source_frame_id"],
                           "end_source_frame_id": rows[-1]["source_frame_id"],
                           "start_seconds": rows[0]["relative_seconds"], "end_seconds": rows[-1]["relative_seconds"],
                           "contact_frame_count": len(rows), "action_phases": phase_labels,
                           "boundary_reason": "contact_island_start" if i == 0 else chosen[i-1]["reason"],
                           "review_required": duration >= decoder_config["ambiguous_cycle_seconds"],
                           "review_reason": "long_continuous_operation_may_contain_multiple_events" if duration >= decoder_config["ambiguous_cycle_seconds"] else None,
                           "status": "experimental_tsm_cycle_hypothesis", "manually_verified": False})
    diagnostics = {"decoder": "cycles_v2", "phases": phases, "protected_cycles": protected,
                   "supported_contact_gap_bridges": bridges,
                   "action_boundary_candidates": action_candidates, "withdrawal_candidates": withdrawal,
                   "rejected_visual_boundaries": rejected, "vetoed_boundaries": vetoed,
                   "selected_boundaries": selected}
    return events, diagnostics
