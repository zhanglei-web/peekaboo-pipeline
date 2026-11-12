"""All geometric confidence is separate from semantic identity confidence."""
import numpy as np
from scipy import ndimage


def box_iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0., x1-x0) * max(0., y1-y0)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0.


def nms(detections, threshold):
    kept = []
    for d in sorted(detections, key=lambda x: -x["score"]):
        if all(box_iou(d["box"], k["box"]) < threshold for k in kept):
            kept.append(d)
    return kept


def mask_box(mask):
    y, x = np.nonzero(mask)
    return [int(x.min()), int(y.min()), int(x.max())+1, int(y.max())+1] if len(x) else None


def interior_anchor(mask):
    labels, components = ndimage.label(mask)
    if not components:
        raise ValueError("Cannot anchor an empty mask")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    component = labels == int(sizes.argmax())
    # Padding treats image boundary as background even for a full-frame mask.
    distance = ndimage.distance_transform_edt(np.pad(component, 1))[1:-1, 1:-1]
    y, x = np.unravel_index(int(distance.argmax()), distance.shape)
    return {"pixel_xy": [float(x)+.5, float(y)+.5], "method": "largest_component_max_interior_distance",
            "boundary_clearance_px": float(distance[y, x]), "connected_components": int(components)}


def choose_mask(box, masks, cfg, descriptors=None):
    ranking = []
    if descriptors is None:
        descriptors=[(int(m.sum()),mask_box(m)) for m in masks]
    for i, mask in enumerate(masks):
        area,mb = descriptors[i]
        if area < cfg["min_mask_pixels"]:
            continue
        iou = box_iou(box, mb)
        x0, y0 = int(np.floor(box[0])), int(np.floor(box[1]))
        x1, y1 = int(np.ceil(box[2])), int(np.ceil(box[3]))
        containment = float(mask[y0:y1, x0:x1].sum()) / area
        if iou >= cfg["mask_box_iou"] and containment >= cfg["mask_containment"]:
            ranking.append({"mask_index": i, "bbox_iou": iou, "mask_containment": containment,
                            "match_score": .7*iou + .3*containment})
    ranking.sort(key=lambda x: -x["match_score"])
    if not ranking:
        return None, "no_matching_mask", []
    if len(ranking) > 1 and ranking[0]["match_score"] - ranking[1]["match_score"] < cfg["mask_score_margin"]:
        return None, "ambiguous_masks", ranking[:3]
    return ranking[0], "matched", ranking[:3]


def ground_frame(frame, detections, masks, cfg):
    width, height = frame["width"], frame["height"]
    if masks.ndim != 3 or masks.shape[1:] != (height, width):
        raise ValueError("Masks must be in the original image coordinate system")
    descriptors=[(int(m.sum()),mask_box(m)) for m in masks]
    anchor_cache={}
    objects, outcomes = [], []
    for candidate in frame["candidates"]:
        ds = []
        for d in detections.get(candidate["prompt"], []):
            a = np.asarray(d["box"], dtype=float)
            if a.shape != (4,) or not np.isfinite(a).all() or not np.isfinite(d["score"]):
                raise ValueError("Nonfinite/invalid model detection")
            a[[0,2]] = a[[0,2]].clip(0, width)
            a[[1,3]] = a[[1,3]].clip(0, height)
            if a[2] <= a[0] or a[3] <= a[1] or d["score"] < cfg["detection_threshold"]:
                continue
            ds.append({"box": a.tolist(), "score": float(d["score"])})
        ds = nms(ds, cfg["nms_iou"])
        truncated = len(ds) > cfg["max_instances_per_prompt"]
        ds = ds[:cfg["max_instances_per_prompt"]]
        refs = []
        for d in ds:
            box = d["box"]
            match, status, ranking = choose_mask(box, masks, cfg, descriptors)
            aid = frame["state_id"] + ":anchor_%03d" % (len(objects)+1)
            refs.append(aid)
            flags = []
            if len(ds) > 1:
                flags.append("multiple_instances_for_source_reference")
            if truncated:
                flags.append("instance_limit_reached")
            mask = masks[match["mask_index"]] if match else None
            if match:
                index=match["mask_index"]
                if index not in anchor_cache:anchor_cache[index]=interior_anchor(mask)
                anchor = dict(anchor_cache[index])
                anchor["inside_segmented_region"] = True
                if anchor["connected_components"] > 1:
                    flags.append("disconnected_mask")
            else:
                anchor = {"pixel_xy": [(box[0]+box[2])/2, (box[1]+box[3])/2],
                          "method": "bbox_center_fallback", "inside_segmented_region": None}
                flags.append(status)
            anchor["normalized_xy"] = [anchor["pixel_xy"][0]/width, anchor["pixel_xy"][1]/height]
            touch = (box[0] <= 1 or box[1] <= 1 or box[2] >= width-1 or box[3] >= height-1)
            if touch:
                flags.append("image_boundary_contact_possible_truncation")
            objects.append({"anchor_id": aid, "source_candidate_id": candidate["candidate_id"],
                "source_ref": candidate["source_ref"], "reference_kind": candidate["reference_kind"],
                "source_label": candidate["source_label"], "detection_prompt": candidate["prompt"],
                "is_frame_focus": candidate["is_frame_focus"],
                "scene_entity_candidates": candidate["scene_entity_candidates"],
                "identity_status": "text_grounded_candidate_not_verified_physical_identity",
                "track_id": None, "bbox_xyxy_pixels": box,
                "bbox_xyxy_normalized": [box[0]/width, box[1]/height, box[2]/width, box[3]/height],
                "detection_score": d["score"], "score_is_calibrated_probability": False,
                "segmentation_status": status, "mask_index": match["mask_index"] if match else None,
                "mask_match": match, "mask_alternatives": ranking,
                "anchor": anchor, "visibility": "occlusion_not_estimated",
                "apparent_size": {"bbox_width_px": box[2]-box[0], "bbox_height_px": box[3]-box[1],
                    "bbox_width_fraction": (box[2]-box[0])/width,
                    "bbox_height_fraction": (box[3]-box[1])/height,
                    "visible_mask_area_px": int(mask.sum()) if match else None,
                    "visible_mask_area_fraction": float(mask.mean()) if match else None,
                    "metric_dimensions": None, "scope": "image_projection_not_physical_size"},
                "review_flags": flags, "conflicting_anchor_ids": [], "human_reviewed": False})
            for key in ['selection_scope','description_event_id','description_evidence','description_role','selection_review_status']:
                if key in candidate:objects[-1][key]=candidate[key]
        outcomes.append({**candidate, "status": "detected" if refs else "not_detected",
                         "anchor_ids": refs, "absence_confirmed": False,
                         "instance_limit_reached": truncated})
    # Do not silently merge two existing IDs even if they refer to the same visible mask.
    for i, a in enumerate(objects):
        for b in objects[i+1:]:
            if a["source_candidate_id"] == b["source_candidate_id"]:
                continue
            same_mask = a["mask_index"] is not None and a["mask_index"] == b["mask_index"]
            if same_mask or box_iou(a["bbox_xyxy_pixels"], b["bbox_xyxy_pixels"]) >= cfg["identity_conflict_iou"]:
                for obj, other in [(a,b),(b,a)]:
                    if "competing_source_references" not in obj["review_flags"]:
                        obj["review_flags"].append("competing_source_references")
                    obj["conflicting_anchor_ids"].append(other["anchor_id"])
    for obj in objects:
        obj["geometry_status"] = "needs_review" if obj["review_flags"] else "mask_matched_unreviewed"
    result={"schema": "ego.frame_anchors.v1", "state_id": frame["state_id"],
            "image_sha256": frame["image_sha256"], "image_size": [width, height],
            "coordinate_system": "native_raster_xy_top_left; boxes_xyxy_half_open; anchor_pixel_centers; normalized_by_width_height",
            "coverage": "existing_json_candidates_only_not_exhaustive_visual_inventory",
            "objects": objects, "candidate_outcomes": outcomes}
    if frame.get('candidate_scope')=='event_description':
        result['coverage']='explicit_event_description_mentions_queried_independently_in_each_triplet_frame'
        result['excluded_description_mentions']=frame['excluded_description_mentions']
    return result
