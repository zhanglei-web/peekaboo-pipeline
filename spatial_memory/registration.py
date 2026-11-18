"""Static-background RGB-D registration from frozen reference to selected frame."""
import cv2
import numpy as np


def estimate_pose(reference_bgr, target_bgr, reference_depth, reference_valid,
                  reference_dynamic, target_dynamic, K):
    """Return T_target_from_reference and quality; reject weak PnP results."""
    if reference_bgr.shape[:2] != target_bgr.shape[:2] or reference_depth.shape != reference_bgr.shape[:2]:
        raise ValueError("RGB and depth shapes differ")
    sift = cv2.SIFT_create(nfeatures=2200)
    a = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)
    ka, da = sift.detectAndCompute(a, (~reference_dynamic.astype(bool)).astype("uint8") * 255)
    kb, db = sift.detectAndCompute(b, (~target_dynamic.astype(bool)).astype("uint8") * 255)
    if da is None or db is None or len(db) < 2:
        return None, {"accepted": False, "reason": "no_static_features"}
    matches = [m for pair in cv2.BFMatcher().knnMatch(da, db, k=2) if len(pair) == 2
               for m, n in [pair] if m.distance < 0.7 * n.distance]
    if len(matches) < 12:
        return None, {"accepted": False, "reason": "few_rgb_matches", "matches": len(matches)}
    p = np.float32([ka[m.queryIdx].pt for m in matches])
    q = np.float32([kb[m.trainIdx].pt for m in matches])
    uv = np.rint(p).astype(int)
    h, w = reference_depth.shape
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    p, q, uv = p[inside], q[inside], uv[inside]
    z = reference_depth[uv[:, 1], uv[:, 0]]
    valid = reference_valid[uv[:, 1], uv[:, 0]].astype(bool) & np.isfinite(z) & (z > 0)
    p, q, z = p[valid], q[valid], z[valid]
    if len(z) < 12:
        return None, {"accepted": False, "reason": "few_valid_rgbd_matches", "matches": len(z)}
    X = np.column_stack(((p[:, 0] - K[0, 2]) * z / K[0, 0],
                         (p[:, 1] - K[1, 2]) * z / K[1, 1], z)).astype("float64")
    q = q.astype("float64")
    ok, r, t, inliers = cv2.solvePnPRansac(X, q, K, np.zeros(5),
                                            iterationsCount=500, reprojectionError=2.0,
                                            confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers) < 6:
        return None, {"accepted": False, "reason": "pnp_failed", "matches": len(X)}
    index = inliers.ravel()
    r, t = cv2.solvePnPRefineLM(X[index], q[index], K, np.zeros(5), r, t)
    projected, _ = cv2.projectPoints(X[index], r, t, K, np.zeros(5))
    error = np.linalg.norm(projected.reshape(-1, 2) - q[index], axis=1)
    quality = {"accepted": bool(len(index) >= 15 and len(index) / len(X) > 0.35
                                and np.median(error) < 1.5),
               "matches": len(X), "inliers": len(index),
               "inlier_fraction": len(index) / len(X),
               "median_reprojection_error_px": float(np.median(error))}
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(r)[0]
    T[:3, 3] = t.ravel()
    return (T if quality["accepted"] else None), quality
