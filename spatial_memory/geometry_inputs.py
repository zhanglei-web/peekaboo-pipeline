"""Package selected RGB-D observations into a reproducible geometry job."""
from pathlib import Path

import cv2
import numpy as np

from .io import write_json
from .registration import estimate_pose


def _image(path, grayscale=False):
    value = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
    if value is None:
        raise FileNotFoundError(path)
    return value


def _depth(path):
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["depth"]), np.asarray(data["valid"])


def prepare_geometry(config, selection, output_dir):
    """Depth/masks are external evidence; registration and role checks happen here."""
    if selection["status"] != "ok":
        raise ValueError("Geometry requires three selected frames")
    if config["search_unit_id"] not in [a["candidate_id"] for a in selection["associations"]]:
        raise ValueError("Geometry unit was not associated with this event")
    if config["scope"] != "original_support_footprint":
        raise ValueError("This input builder supports only original support footprints")
    K = np.asarray(config["K"], dtype=float)
    if K.shape != (3, 3):
        raise ValueError("Camera matrix K must be 3x3")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = config["reference"]
    ref_image = _image(reference["left_image"])
    ref_depth, ref_valid = _depth(reference["depth_npz"])
    ref_object = _image(reference["object_mask"], True)
    ref_support = _image(reference["support_mask"], True)
    ref_dynamic = _image(reference["dynamic_mask"], True)
    shape = ref_depth.shape
    if any(x.shape != shape for x in (ref_image[:, :, 0], ref_valid, ref_object,
                                      ref_support, ref_dynamic)):
        raise ValueError("Reference image, depth, and masks must align")
    np.savez_compressed(output_dir / "reference.npz", depth=ref_depth,
                        valid=ref_valid, object_mask=ref_object,
                        support_mask=ref_support, K=K)
    frames = {}
    diagnostics = {}
    for role in ("pre", "pnr", "post"):
        source = config["frames"][role]
        fid = int(source["source_frame_id"])
        if fid != selection[role]:
            raise ValueError(f"{role} geometry does not match Qwen selection")
        image = _image(source["left_image"])
        depth, valid = _depth(source["depth_npz"])
        foreground = _image(source["foreground_mask"], True)
        dynamic = _image(source["dynamic_mask"], True)
        if any(x.shape != depth.shape for x in (image[:, :, 0], valid, foreground, dynamic)):
            raise ValueError(f"{role} image, depth, and masks must align")
        if fid == int(reference["source_frame_id"]):
            T = np.eye(4)
            quality = {"accepted": True, "identity": True}
        else:
            T, quality = estimate_pose(ref_image, image, ref_depth, ref_valid,
                                       ref_dynamic, dynamic, K)
        target = output_dir / f"{role}.npz"
        np.savez_compressed(target, source_frame_id=fid, depth=depth, valid=valid,
                            foreground=foreground, K=K,
                            T=np.eye(4) if T is None else T,
                            pose_valid=T is not None)
        frames[role] = str(target)
        diagnostics[role] = quality
    job = {"search_unit_id": config["search_unit_id"], "scope": config["scope"],
           "reference_npz": str((output_dir / "reference.npz").resolve()),
           "frames": {role: str(Path(path).resolve()) for role, path in frames.items()},
           "quality_audit": config.get("quality_audit", {
               "accepted": False, "reasons": ["support_and_cross_frame_quality_audit_not_provided"]})}
    write_json(output_dir / "registration.json", diagnostics)
    write_json(output_dir / "job.json", job)
    return job
