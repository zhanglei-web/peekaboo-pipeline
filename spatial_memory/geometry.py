"""Fixed-reference, three-frame geometric coverage for an external support footprint."""
from pathlib import Path

import cv2
import numpy as np

from .io import write_json


EPSILON_M = 0.02


def _array(bundle, key, shape=None):
    if key not in bundle:
        raise ValueError(f"Missing geometry field: {key}")
    value = np.asarray(bundle[key])
    if shape is not None and value.shape != shape:
        raise ValueError(f"Wrong shape for {key}: {value.shape}, expected {shape}")
    return value


def _backproject(depth, K):
    h, w = depth.shape
    y, x = np.indices((h, w))
    return np.stack(((x - K[0, 2]) * depth / K[0, 0],
                     (y - K[1, 2]) * depth / K[1, 1], depth), axis=-1)


def make_reference(bundle):
    """Fit adjacent support from a supplied mask, then sample projected object hull.

    The object/support masks and stereo depth are inputs from upstream perception.
    This routine never selects the plane using later-frame coverage.
    """
    depth = _array(bundle, "depth").astype(float)
    valid = _array(bundle, "valid", depth.shape).astype(bool) & np.isfinite(depth) & (depth > 0)
    obj = _array(bundle, "object_mask", depth.shape).astype(bool) & valid
    support = _array(bundle, "support_mask", depth.shape).astype(bool) & valid
    K = _array(bundle, "K", (3, 3)).astype(float)
    if np.count_nonzero(obj) < 50 or np.count_nonzero(support) < 100:
        raise ValueError("Insufficient valid object or support depth")
    if np.any(~np.isfinite(K)) or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("Invalid camera intrinsics")
    xyz = _backproject(depth, K)
    support_points = xyz[support]
    center = support_points.mean(axis=0)
    _, _, vh = np.linalg.svd(support_points - center, full_matrices=False)
    n = vh[-1]
    if n @ center > 0:
        n = -n
    offset = -float(n @ center)
    residual = np.abs(support_points @ n + offset)
    if np.median(residual) > 0.008 or np.quantile(residual, 0.9) > 0.02:
        raise ValueError("Support mask does not fit a stable plane")
    object_points = xyz[obj]
    above_fraction = float(np.mean(object_points @ n + offset >= -0.008))
    if above_fraction < 0.9:
        raise ValueError("Reference plane cuts through the object")
    projected = object_points - (object_points @ n + offset)[:, None] * n
    axis_u = np.cross(n, [0., 0., 1.])
    if np.linalg.norm(axis_u) < 0.1:
        axis_u = np.cross(n, [0., 1., 0.])
    axis_u /= np.linalg.norm(axis_u)
    axis_v = np.cross(n, axis_u)
    origin = projected.mean(axis=0)
    xy = np.column_stack(((projected - origin) @ axis_u, (projected - origin) @ axis_v))
    lo, hi = np.quantile(xy, [0.02, 0.98], axis=0)
    xy = xy[np.all((xy >= lo) & (xy <= hi), axis=1)]
    hull = cv2.convexHull(xy.astype(np.float32)).reshape(-1, 2)
    area = abs(float(cv2.contourArea(hull)))
    if not 0.0001 < area < 1.0:
        raise ValueError("Projected footprint area is implausible")
    minimum, maximum = hull.min(axis=0), hull.max(axis=0)
    step = max(0.003, np.sqrt(area / 1800.0))
    sx, sy = np.meshgrid(np.arange(minimum[0], maximum[0] + step, step),
                         np.arange(minimum[1], maximum[1] + step, step))
    samples = np.column_stack((sx.ravel(), sy.ravel()))
    inside = np.array([cv2.pointPolygonTest(hull, tuple(map(float, p)), False) >= 0
                       for p in samples])
    samples = samples[inside]
    if len(samples) < 30:
        raise ValueError("Empty or undersampled reference footprint")
    points = origin + samples[:, 0, None] * axis_u + samples[:, 1, None] * axis_v
    return points, {"reference_area_m2": area, "reference_points": len(points),
                    "support_median_residual_m": float(np.median(residual)),
                    "support_p90_residual_m": float(np.quantile(residual, 0.9)),
                    "object_above_support_fraction": above_fraction,
                    "normal": n.tolist(), "offset": offset}


def visibility(points, bundle, epsilon_m=EPSILON_M):
    depth = _array(bundle, "depth").astype(float)
    valid = _array(bundle, "valid", depth.shape).astype(bool)
    foreground = _array(bundle, "foreground", depth.shape).astype(bool)
    K = _array(bundle, "K", (3, 3)).astype(float)
    T = _array(bundle, "T", (4, 4)).astype(float)
    if not bool(np.asarray(bundle["pose_valid"]).item()) or not np.isfinite(T).all():
        raise ValueError("Pose registration was not validated")
    xyz = points @ T[:3, :3].T + T[:3, 3]
    z = xyz[:, 2]
    safe_z = np.where(z > 0, z, 1)
    x = np.rint(K[0, 0] * xyz[:, 0] / safe_z + K[0, 2]).astype(int)
    y = np.rint(K[1, 1] * xyz[:, 1] / safe_z + K[1, 2]).astype(int)
    h, w = depth.shape
    inside = (z > 0) & (x >= 0) & (x < w) & (y >= 0) & (y < h)
    seen = np.zeros(len(points), bool)
    known = np.zeros(len(points), bool)
    ix = np.flatnonzero(inside)
    if len(ix):
        observed = depth[y[ix], x[ix]]
        usable = valid[y[ix], x[ix]] & np.isfinite(observed) & (observed > 0)
        known[ix[usable]] = True
        seen[ix[usable]] = (np.abs(observed[usable] - z[ix[usable]]) <= epsilon_m) & ~foreground[y[ix[usable]], x[ix[usable]]]
    return seen, known


def score_job(job, selection, output_dir, *, epsilon_m=EPSILON_M, max_unknown_fraction=0.25):
    """Score one selected search unit. Input NPZs must identify the chosen source frames."""
    if selection["status"] != "ok":
        raise ValueError("Three confirmed frame roles are required")
    uid = job["search_unit_id"]
    if uid not in [a["candidate_id"] for a in selection["associations"]]:
        raise ValueError("Geometry job has no selected-event region association")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {"episode_id": selection["episode_id"], "event_id": selection["event_id"],
              "search_unit_id": uid, "score_type": "geometric_coverage",
              "scope": job["scope"], "q": None, "status": "uncomputable",
              "reason": None, "epsilon_m": epsilon_m,
              "source_frame_ids": {r: selection[r] for r in ("pre", "pnr", "post")}}
    if job.get("scope") != "original_support_footprint":
        result["reason"] = "unsupported_geometry_scope"
        write_json(output_dir / "assessment.json", result)
        return result
    try:
        with np.load(job["reference_npz"], allow_pickle=False) as ref:
            points, reference = make_reference(ref)
        visible = []
        known = []
        for role in ("pre", "pnr", "post"):
            with np.load(job["frames"][role], allow_pickle=False) as frame:
                if int(np.asarray(frame["source_frame_id"]).item()) != selection[role]:
                    raise ValueError(f"{role} depth does not match selected source frame")
                v, k = visibility(points, frame, epsilon_m)
            visible.append(v)
            known.append(k)
        union = np.logical_or.reduce(visible)
        ever_known = np.logical_or.reduce(known)
        unknown_fraction = float(np.mean(~ever_known))
        result.update(reference)
        result["raw_q"] = float(np.mean(union))
        result["unknown_fraction"] = unknown_fraction
        result["per_frame_visible_fraction"] = {r: float(np.mean(v)) for r, v in zip(("pre", "pnr", "post"), visible)}
        np.savez_compressed(output_dir / "point_evidence.npz", reference_points=points,
                            visible=np.stack(visible), known=np.stack(known), union=union)
        if unknown_fraction > max_unknown_fraction:
            result["reason"] = "too_many_reference_points_without_valid_depth"
        else:
            result["q"] = result["raw_q"]
            result["status"] = "valid"
    except (ValueError, KeyError, OSError, np.linalg.LinAlgError) as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
    write_json(output_dir / "assessment.json", result)
    return result
