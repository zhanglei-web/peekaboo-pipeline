"""Final selected triplet -> new stereo/masks/reference -> geometric q -> memory."""
from copy import deepcopy
from pathlib import Path
import time

import cv2
import numpy as np

from online_annotation.model import JsonModel
from .geometry import _backproject, make_reference, score_job
from .estimation import ensure_numeric
from .geometry_inputs import prepare_geometry
from .grounding import ObjectMasks, ground_triplet
from .io import read_json, write_json
from .memory import apply_event
from .registration import estimate_pose
from .selection import MODEL
from .stereo import FastStereoDepth, StereoSequence
from .support import plane_hypotheses


def static_check(points, frame, dynamic):
    """Validate registration on static support points, without using coverage q."""
    with np.load(frame, allow_pickle=False) as f:
        if not bool(f["pose_valid"]):
            return {"accepted": False, "reason": "pose_failed"}
        T, K, depth, valid = f["T"], f["K"], f["depth"], f["valid"].astype(bool)
    xyz = points @ T[:3, :3].T + T[:3, 3]
    z = xyz[:, 2]
    safe = np.where(z > 0, z, 1.)
    x = np.rint(xyz[:, 0] / safe * K[0, 0] + K[0, 2]).astype(int)
    y = np.rint(xyz[:, 1] / safe * K[1, 1] + K[1, 2]).astype(int)
    h, w = depth.shape
    inside = (z > 0) & (x >= 0) & (x < w) & (y >= 0) & (y < h)
    ids = np.flatnonzero(inside)
    ids = ids[valid[y[ids], x[ids]] & ~dynamic[y[ids], x[ids]]]
    error = depth[y[ids], x[ids]] - z[ids]
    if len(error) < 80:
        return {"accepted": False, "reason": "too_few_static_validation_points", "points": len(error)}
    agreement = float(np.mean(abs(error) <= .02))
    median = float(np.median(error))
    return {"accepted": bool(abs(median) <= .012 and agreement >= .6),
            "points": len(error), "agreement_fraction": agreement,
            "median_signed_depth_error_m": median}


def _overlap_on_plane(a, b, normal):
    axis = np.cross(normal, [0., 0., 1.])
    if np.linalg.norm(axis) < .1:
        axis = np.cross(normal, [0., 1., 0.])
    axis /= np.linalg.norm(axis)
    other = np.cross(normal, axis)
    ca = cv2.convexHull(np.column_stack((a @ axis, a @ other)).astype("float32"))
    cb = cv2.convexHull(np.column_stack((b @ axis, b @ other)).astype("float32"))
    intersection, _ = cv2.intersectConvexConvex(ca, cb)
    return float(intersection / max(cv2.contourArea(ca) + cv2.contourArea(cb) - intersection, 1e-12))


class GeometryCompiler:
    def __init__(self, config, model_root, base_url):
        self.config = config
        self.model = JsonModel(model_root, base_url)
        if self.model.model != MODEL:
            raise ValueError(f"This real test requires {MODEL}; configured model is {self.model.model}")
        self.depth = None
        self.segmenter = None

    def _models(self):
        if self.depth is None:
            self.depth = FastStereoDepth(self.config["stereo_repository"], self.config["stereo_checkpoint"])
        if self.segmenter is None:
            self.segmenter = ObjectMasks(self.config["fastsam_weights"],
                                         pythonpath=self.config.get("ultralytics_pythonpath"))

    def _reuse_reference(self, registry, episode, candidate, current, points, normal):
        matches = []
        for previous in registry:
            if previous["episode_id"] != episode:
                continue
            with np.load(previous["reference_npz"], allow_pickle=False) as ref:
                old_points, _ = make_reference(ref)
                old_depth, old_valid = ref["depth"], ref["valid"]
            old = previous["reference"]
            T, quality = estimate_pose(cv2.imread(old["left_image"]), cv2.imread(current["left_image"]),
                old_depth, old_valid, cv2.imread(old["dynamic_mask"], 0) > 0,
                cv2.imread(current["dynamic_mask"], 0) > 0, np.array(previous["K"]))
            if T is None:
                continue
            transformed = old_points @ T[:3, :3].T + T[:3, 3]
            distance = abs(float(np.median((transformed - points.mean(axis=0)) @ normal)))
            overlap = _overlap_on_plane(points, transformed, normal)
            if distance <= .02 and overlap >= .85:
                matches.append((overlap, previous, quality))
        if len(matches) > 1:
            raise ValueError("Ambiguous stable-reference association")
        return matches[0] if matches else None

    def compile(self, session, event, selection, catalog, state, output_dir, registry=None,
                detections=None):
        """A registry preserves fixed references for verified geometric revisits."""
        best_effort = self.config.get("score_policy", "best_effort") == "best_effort"
        if selection["model"] != MODEL or selection["status"] not in (("ok", "uncertain") if best_effort else ("ok",)):
            raise ValueError("Requires a confirmed triplet from the requested 8B model")
        if not all(isinstance(selection.get(role), int) for role in ("pre", "pnr", "post")):
            raise ValueError("The VLM must select all three actual frame references before scoring")
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        registry = deepcopy(registry or [])
        geometry_setup_failure = None
        try:
            stereo = StereoSequence(session)
            frames = stereo.prepare(selection, output_dir / "stereo")
            if detections is None:
                raise ValueError("Automatic geometry requires source-frame hand detections")
            hand_masks = stereo.hand_exclusions(selection, detections, output_dir / "hand_evidence.json")
        except (ValueError, KeyError, OSError, RuntimeError) as exc:
            if not best_effort:
                raise
            geometry_setup_failure = f"{type(exc).__name__}: {exc}"
            by_id = {c["source_frame_id"]: c for c in event["candidates"]}
            frames = {role: {"left": by_id[selection[role]]["image"]} for role in ("pre", "pnr", "post")}
        geometry_selection = deepcopy(selection)
        geometry_selection["associations"] = []
        assessments, new_units = [], []
        new_parents = []
        event_references = []
        regions = {r["candidate_id"]: r for r in catalog["regions"]}
        associations = deepcopy(selection["associations"])
        if not associations and best_effort:
            uid = f"{catalog['episode_id']}:unresolved_scope:{event['event_id']}"
            parent = uid + ":object"
            regions[uid] = {"candidate_id": uid, "parent_id": parent, "name": "unresolved event search scope",
                            "boundary_description": "Inspection scope visible in this event; catalog association unresolved"}
            associations = [{"candidate_id": uid, "reason": "numeric estimate with unresolved spatial association",
                             "confidence": "low"}]
            new_parents.append({"parent_id": parent, "name": "unresolved event context", "source": "unresolved_event"})
            new_units.append({"search_unit_id": uid, "parent_id": parent, "name": regions[uid]["name"],
                "boundary_description": regions[uid]["boundary_description"], "region_type": "unresolved_visual_scope",
                "identity_anchor": "event_local", "confirmation_basis": "operational_estimate_only_unresolved_association"})
        for association_index, association in enumerate(associations):
            source_id = association["candidate_id"]
            region = regions[source_id]
            folder = output_dir / f"unit_{association_index:03d}"
            folder.mkdir(exist_ok=True)
            uid = source_id
            scope = "unestablished"
            failure = None
            job = None
            try:
                if selection["status"] != "ok":
                    raise ValueError("VLM selected best-effort frames with uncertain semantic roles")
                if geometry_setup_failure:
                    raise ValueError(geometry_setup_failure)
                if source_id not in {r["candidate_id"] for r in catalog["regions"]}:
                    raise ValueError("Event has no resolved catalog region association")
                grounding = ground_triplet(self.model, frames, region, folder)
                scope = grounding["scope"]
                if scope != "original_support_footprint":
                    raise ValueError("No external support-footprint inspection established by this triplet")
                if grounding["pre"]["object_box"] is None:
                    raise ValueError("Initial operated object cannot be grounded")
                self._models()
                sources, masks = {}, {}
                for role in ("pre", "pnr", "post"):
                    frame = frames[role]
                    mask_dir = folder / f"{role}_masks"
                    masks[role] = self.segmenter.compute(frame["left"], grounding[role], mask_dir,
                                                         hand_mask=hand_masks[role])
                    depth_path = output_dir / "stereo" / f"{role}_depth.npz"
                    if not depth_path.exists():
                        timing = self.depth.compute(frame, stereo.K, stereo.baseline_m, depth_path)
                        write_json(depth_path.with_suffix(".timing.json"), timing)
                    sources[role] = {"source_frame_id": frame["source_frame_id"],
                        "left_image": frame["left"], "depth_npz": str(depth_path),
                        "foreground_mask": str(mask_dir / "dynamic.png"),
                        "dynamic_mask": str(mask_dir / "dynamic.png")}
                with np.load(sources["pre"]["depth_npz"], allow_pickle=False) as pre:
                    xyz = _backproject(pre["depth"], stereo.K)
                    normal_hint = stereo.support_normal_hint(selection["pre"])
                    write_json(folder / "support_orientation.json", {"hint": normal_hint})
                    candidates, plane_diag = plane_hypotheses(xyz, pre["valid"].astype(bool),
                        masks["pre"][0].astype("uint8"), masks["pre"][2].astype("uint8"),
                        prefer_adjacent_component=True, support_normal_hint=normal_hint)
                    if not candidates:
                        raise ValueError("No adjacent connected support plane")
                    plane = candidates[0]
                    current_reference = {**sources["pre"],
                        "object_mask": str(folder / "pre_masks/object.png"),
                        "support_mask": str(folder / "support.png")}
                    cv2.imwrite(current_reference["support_mask"], plane["support_mask"].astype("uint8") * 255)
                    current_points, _ = make_reference({"depth": pre["depth"], "valid": pre["valid"],
                        "object_mask": masks["pre"][0], "support_mask": plane["support_mask"], "K": stereo.K})
                duplicate = next((entry for entry in event_references
                    if abs(float(np.median((entry["points"] - current_points.mean(0)) @ plane["normal"]))) <= .02
                    and _overlap_on_plane(entry["points"], current_points, plane["normal"]) >= .9), None)
                if duplicate:
                    prior = assessments[duplicate["assessment_index"]]
                    prior.setdefault("source_region_ids", [prior["source_region_id"]]).append(source_id)
                    if duplicate["parent_id"] != region["parent_id"]:
                        prior["region_association_status"] = "ambiguous_parent_region_links"
                    write_json(folder / "duplicate_scope.json", {"search_unit_id": prior["search_unit_id"],
                        "reason": "same_registered_local_footprint_already_scored_in_this_event",
                        "source_region_ids": prior["source_region_ids"]})
                    continue
                reuse = self._reuse_reference(registry, catalog["episode_id"], source_id,
                                              current_reference, current_points, plane["normal"])
                reference = current_reference
                if reuse:
                    _, previous, match_quality = reuse
                    uid, reference = previous["search_unit_id"], previous["reference"]
                    write_json(folder / "stable_reference_match.json", {"overlap": reuse[0], "pose": match_quality,
                                                                         "search_unit_id": uid})
                else:
                    uid = source_id + ":support_footprint:" + event["event_id"]
                    new_units.append({"search_unit_id": uid, "parent_id": region["parent_id"],
                        "macro_id": region.get("macro_id", f"{catalog['episode_id']}:scene"),
                        "name": grounding["object_name"] + " original support footprint",
                        "region_type": "occluder_footprint", "identity_anchor": "original_footprint",
                        "boundary_description": "Frozen projected initial-object footprint; excludes full parent region",
                        "confirmation_basis": {"source_region_id": source_id,
                            "geometry_reference": str(folder / "inputs/reference.npz"),
                            "grounding": str(folder / "grounding.json"), "independent_human_review": False}})
                    # Geometry establishes a new operated-object grouping. It does
                    # not assign this local coverage to an entire language region.
                    object_group = uid + ":object"
                    new_units[-1]["parent_id"] = object_group
                    new_units[-1]["source_region_id"] = source_id
                    new_parents.append({"parent_id": object_group, "name": grounding["object_name"],
                        "source": "event_operated_object_with_fixed_geometric_footprint",
                        "independent_human_review": False})
                quality_reasons = []
                if normal_hint is None:
                    quality_reasons.append("support_orientation_unverified")
                if len(candidates) > 1 and candidates[1]["support_pixels"] / plane["support_pixels"] >= .8:
                    quality_reasons.append("competing_support_plane_hypotheses")
                config = {"K": stereo.K.tolist(), "search_unit_id": uid, "scope": scope,
                          "reference": reference, "frames": sources}
                job = prepare_geometry(config, selection_for_unit(selection, uid), folder / "inputs")
                with np.load(job["reference_npz"], allow_pickle=False) as ref:
                    support_xyz = _backproject(ref["depth"], ref["K"])
                    support_points = support_xyz[ref["support_mask"].astype(bool) & ref["valid"].astype(bool)][::5]
                checks = {role: static_check(support_points, job["frames"][role], masks[role][2])
                          for role in ("pre", "pnr", "post")}
                if not all(c["accepted"] for c in checks.values()):
                    quality_reasons.append("static_surface_cross_frame_inconsistent")
                job["quality_audit"] = {"accepted": not quality_reasons, "reasons": quality_reasons,
                                         "plane_generation": plane_diag, "static_checks": checks,
                                         "support_normal_hint": normal_hint,
                                         "selected_support_normal": plane["normal"].tolist(),
                                         "source": "automatic_geometry_compiler_before_scoring"}
                write_json(folder / "inputs/job.json", job)
                write_json(folder / "geometry_sources.json", config)
                assessment = score_job(job, selection_for_unit(selection, uid), folder / "score")
                event_references.append({"points": current_points, "parent_id": region["parent_id"],
                                         "assessment_index": len(assessments)})
                if not reuse and not quality_reasons:
                    registry.append({"episode_id": catalog["episode_id"], "source_region_id": source_id,
                        "search_unit_id": uid, "reference": reference, "K": stereo.K.tolist(),
                        "reference_npz": job["reference_npz"]})
            except (ValueError, KeyError, OSError, RuntimeError) as exc:
                failure = f"{type(exc).__name__}: {exc}"
                assessment = {"episode_id": catalog["episode_id"], "event_id": event["event_id"],
                    "search_unit_id": uid, "score_type": "geometric_coverage", "scope": scope,
                    "q": None, "status": "uncomputable", "reason": failure}
                write_json(folder / "assessment_failure.json", assessment)
            if best_effort:
                estimate_region = region
                if uid != source_id:
                    estimate_region = {"name": grounding["object_name"] + " original support footprint",
                        "boundary_description": "Only the fixed initial operated-object support footprint, not the entire parent region"}
                assessment = ensure_numeric(assessment, job, self.model, frames, estimate_region, folder / "numeric_score")
            assessment["source_region_id"] = source_id
            assessments.append(assessment)
            geometry_selection["associations"].append({**association, "candidate_id": uid,
                                                        "source_region_id": source_id})
        previous_ready = (state["applied_events"][-1]["available_after_s"] if state["applied_events"]
                          else state.get("available_after_s", 0.))
        completed_after = max(previous_ready, max(c["timestamp_s"] for c in event["candidates"])) + (
            selection.get("model_seconds", 0) + time.perf_counter() - start)
        record = {"episode_id": catalog["episode_id"], "event_id": event["event_id"],
            "catalog_version": catalog["catalog_version"], "event_end_time_s": event["event_end_time_s"],
            "available_after_s": completed_after,
            "availability_note": "offline conservative serial replay: last input arrival + selection + geometry wall time",
            "selection": geometry_selection, "assessments": assessments, "new_units": new_units,
            "new_parents": new_parents, "score_policy": "best_effort" if best_effort else "strict"}
        if best_effort:
            record["q_event_estimate"] = float(np.mean([a["q"] for a in assessments]))
            record["q_event_semantics"] = "operational_best_available_sufficiency_estimate"
        memory = apply_event(state, record)
        write_json(output_dir / "committed_event.json", record)
        write_json(output_dir / "memory.json", memory)
        write_json(output_dir / "reference_registry.json", registry)
        return memory, registry, record


def selection_for_unit(selection, uid):
    result = deepcopy(selection)
    result["associations"] = [{"candidate_id": uid}]
    return result
