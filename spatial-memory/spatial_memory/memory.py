"""Chronological, idempotence-checked Q=max updates with unknown-aware hierarchy."""
from copy import deepcopy
import math

from .catalog import initialize


def _summary(values):
    if not values:
        return {"Q": None, "known_mean": None, "unknown_count": 0,
                "count": 0, "lower_bound": None, "upper_bound": None}
    known = [v for v in values if v is not None]
    count = len(values)
    unknown = count - len(known)
    total = sum(known)
    return {"Q": total / count if unknown == 0 else None,
            "known_mean": total / len(known) if known else None,
            "unknown_count": unknown, "count": count,
            "lower_bound": total / count,
            "upper_bound": (total + unknown) / count}


def summarize(state):
    objects = {}
    for unit in state["units"].values():
        if unit["confirmed"]:
            objects.setdefault(unit["parent_id"], []).append(unit["Q"])
    state["object_summary"] = {pid: _summary(values) for pid, values in objects.items()}
    macros = {}
    for pid, group in state["object_summary"].items():
        macro_ids = {u["macro_id"] for u in state["units"].values()
                     if u["parent_id"] == pid and u["confirmed"]}
        if len(macro_ids) != 1:
            raise ValueError("One object group must belong to exactly one macro group")
        macros.setdefault(next(iter(macro_ids)), []).append(group["Q"])
    state["macro_summary"] = {mid: _summary(values) for mid, values in macros.items()}


def _q_snapshot(state):
    return {uid: {"Q": unit["Q"], "last_q": unit["last_q"],
                  "confirmed": unit["confirmed"], "visit_count": unit["visit_count"]}
            for uid, unit in state["units"].items()}


def apply_event(state, event):
    """Return a new state. A revision must be handled by replaying from catalog."""
    state = deepcopy(state)
    if event["episode_id"] != state["episode_id"] or event["catalog_version"] != state["catalog_version"]:
        raise ValueError("Event episode or catalog version mismatch")
    eid = event["event_id"]
    if any(x["event_id"] == eid for x in state["applied_events"]):
        raise ValueError("Duplicate event ID; replay from the original catalog for a revision")
    end = float(event["event_end_time_s"])
    available = float(event["available_after_s"])
    if not math.isfinite(end) or not math.isfinite(available) or available < end:
        raise ValueError("Memory cannot be available before the event ends")
    if state["applied_events"] and (end < state["applied_events"][-1]["event_end_time_s"]
                                    or available < state["applied_events"][-1]["available_after_s"]):
        raise ValueError("Events must be committed in chronological availability order")
    selection = event["selection"]
    if selection["status"] != "ok" or selection["event_id"] != eid:
        raise ValueError("Event needs a confirmed three-frame selection")
    if not selection["pre"] < selection["pnr"] < selection["post"]:
        raise ValueError("Pre, PNR, Post source frames must be ordered")
    if selection["catalog_version"] != state["catalog_version"]:
        raise ValueError("Selection refers to another catalog version")
    before = _q_snapshot(state)
    for new in event.get("new_units", []):
        uid = new["search_unit_id"]
        if uid in state["units"]:
            raise ValueError("New search unit ID already exists")
        if not new.get("confirmation_basis") or not new.get("boundary_description"):
            raise ValueError("New unit needs explicit identity and boundary evidence")
        state["units"][uid] = {"search_unit_id": uid, "candidate_id": None,
                               "parent_id": new["parent_id"],
                               "macro_id": new.get("macro_id", f"{state['episode_id']}:scene"),
                               "name": new["name"], "region_type": new["region_type"],
                               "identity_anchor": new["identity_anchor"],
                               "boundary_description": new["boundary_description"],
                               "catalog_version": state["catalog_version"],
                               "source": "event_discovery", "confirmed": True,
                               "searched": False, "visit_count": 0, "last_q": None,
                               "Q": 0.0, "history": [],
                               "confirmation_basis": new["confirmation_basis"]}
    associated = {a["candidate_id"] for a in selection["associations"]}
    if any(uid not in state["units"] for uid in associated):
        raise ValueError("Selected region is not a memory unit")
    assessments = event.get("assessments", [])
    ids = [a["search_unit_id"] for a in assessments]
    if len(ids) != len(set(ids)) or set(ids) != associated:
        raise ValueError("Provide exactly one geometry assessment per associated unit")
    for assessment in assessments:
        uid = assessment["search_unit_id"]
        if (assessment["event_id"] != eid
                or assessment.get("episode_id") != state["episode_id"]
                or assessment["score_type"] != "geometric_coverage"
                or not assessment.get("scope")):
            raise ValueError("Assessment identity or numeric semantics mismatch")
        q = assessment.get("q")
        if q is not None and (isinstance(q, bool) or not isinstance(q, (int, float))
                              or not math.isfinite(q) or not 0 <= q <= 1):
            raise ValueError("Geometric q must be finite and in [0,1]")
        if assessment["status"] not in ("valid", "uncomputable"):
            raise ValueError("Unknown geometry status")
        if assessment["status"] == "valid" and q is None:
            raise ValueError("Valid geometry needs q")
        if assessment["status"] == "valid" and assessment["scope"] != "original_support_footprint":
            raise ValueError("Numeric geometry has no implemented scope")
        if assessment["status"] != "valid" and q is not None:
            raise ValueError("Uncomputable geometry must not write a numeric q")
        unit = state["units"][uid]
        unit["confirmed"] = True
        unit["searched"] = True
        unit["visit_count"] += 1
        unit["last_q"] = q
        if q is not None:
            unit["Q"] = q if unit["Q"] is None else max(unit["Q"], q)
        unit["history"].append({"event_id": eid, "q": q,
                                 "status": assessment["status"],
                                 "reason": assessment.get("reason"),
                                 "scope": assessment.get("scope"),
                                 "available_after_s": available})
    summarize(state)
    state["applied_events"].append({"event_id": eid, "event_end_time_s": end,
                                     "available_after_s": available,
                                     "pre": selection["pre"], "pnr": selection["pnr"],
                                     "post": selection["post"],
                                     "before": before, "after": _q_snapshot(state),
                                     "object_summary": deepcopy(state["object_summary"]),
                                     "macro_summary": deepcopy(state["macro_summary"])})
    return state


def replay(catalog, events, cutoff_s=None):
    state = initialize(catalog)
    for event in events:
        if cutoff_s is not None and event["available_after_s"] > cutoff_s:
            continue
        state = apply_event(state, event)
    return state
