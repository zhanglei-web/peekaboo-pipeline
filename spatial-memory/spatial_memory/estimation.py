"""Always-numeric sufficiency with explicit measured, approximate and default provenance."""
import math
from pathlib import Path

import numpy as np
from jsonschema import ValidationError

from online_annotation.contracts import object_schema
from .geometry import make_reference, visibility
from .io import write_json


ESTIMATE_SCHEMA = object_schema({
    "q": {"type": "number", "minimum": 0, "maximum": 1},
    "reason": {"type": "string"},
    "scope_description": {"type": "string"},
    "evidence_strength": {"type": "string", "enum": ["low", "medium", "high"]},
})


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1


def approximate_geometry(assessment, job, output):
    """Retain measured geometry when possible; skip failed poses instead of discarding all views."""
    if job is None or job.get("scope") != "original_support_footprint":
        return None
    reasons = job.get("quality_audit", {}).get("reasons", [])
    raw = assessment.get("raw_q")
    if numeric(raw) and "static_surface_cross_frame_inconsistent" not in reasons:
        return {**assessment, "q": float(raw), "status": "estimated",
                "estimation_method": "geometric_union_with_quality_warnings",
                "evidence_strength": "low", "confidence_calibrated": False,
                "strict_geometry_accepted": False, "fallback_reason": assessment.get("reason"),
                "warnings": sorted(set(assessment.get("warnings", []) + reasons))}
    try:
        with np.load(job["reference_npz"], allow_pickle=False) as reference:
            points, _ = make_reference(reference)
        checks = job.get("quality_audit", {}).get("static_checks", {})
        visible, roles = [], []
        for role in ("pre", "pnr", "post"):
            if role in checks and not checks[role]["accepted"]:
                continue
            with np.load(job["frames"][role], allow_pickle=False) as frame:
                if not bool(frame["pose_valid"]):
                    continue
                view, _ = visibility(points, frame)
            visible.append(view)
            roles.append(role)
        if not visible:
            return None
        union = np.logical_or.reduce(visible)
        np.savez_compressed(Path(output) / "approximate_point_evidence.npz", reference_points=points,
                            visible=np.stack(visible), union=union)
        return {**assessment, "q": float(union.mean()), "score_type": "geometric_coverage",
                "status": "estimated", "estimation_method": "available_registered_views_lower_bound",
                "available_views": roles, "evidence_strength": "low", "confidence_calibrated": False,
                "strict_geometry_accepted": False, "fallback_reason": assessment.get("reason"),
                "warnings": sorted(set(reasons + ["incomplete_three_view_geometry"]))}
    except (ValueError, KeyError, OSError, np.linalg.LinAlgError):
        return None


def ensure_numeric(assessment, job, model, frames, region, output):
    """Priority: valid geometry -> approximate geometry -> continuous visual estimate -> declared default."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "strict_assessment.json", assessment)
    if assessment.get("status") == "valid" and numeric(assessment.get("q")):
        result = {**assessment, "estimation_method": "validated_geometric_union",
                  "strict_geometry_accepted": True, "confidence_calibrated": False}
    else:
        result = approximate_geometry(assessment, job, output)
    if result is not None:
        write_json(output / "assessment.json", result)
        return result
    scope = region.get("boundary_description", "event's inspected visual context")
    prompt = (
        "你在给搜索事件输出一个相对可信的充分性估计。三张图依次为Pre、PNR、Post。"
        "几何计算未能完成，现在根据实际图像估计指定范围在本次检查中被有效查看的比例q，取0到1连续值。"
        "这是视觉近似估计，不是几何测量或已校准概率。请结合移开物体后暴露的范围、观察视角、手部遮挡、"
        "容器开口及内部可见范围判断。发生动作或接触物体本身不等于充分检查。"
        "不要把有效/部分/无效机械编码为1/0.5/0；尽量给出具体比例和简短图像依据。"
        "前态缺失或定位较弱时仍给最合理的数值，并将evidence_strength设为low。"
        "完全没有可用观察依据时q=0并说明无依据，不能编造可见区域。"
        f"\n区域名称：{region.get('name', '')}。估计范围：{scope}。"
        f"\n几何失败原因：{assessment.get('reason')}。只返回规定JSON，理由不超过120字。"
    )
    errors = []
    images = [frames[role]["left"] for role in ("pre", "pnr", "post") if role in frames]
    for attempt in range(2):
        try:
            value, receipt = model.infer(output / f"visual_estimate_{attempt + 1}", prompt,
                                         images, ESTIMATE_SCHEMA, 900)
            if not numeric(value["q"]):
                raise ValueError("Visual estimate is not a finite value in [0,1]")
            result = {**assessment, "q": float(value["q"]), "status": "estimated",
                      "score_type": "visual_sufficiency_estimate",
                      "scope": assessment.get("scope") if job else "candidate_visual_scope",
                      "estimation_method": "qwen_continuous_visual_estimate", "model": model.model,
                      "reason": value["reason"], "estimated_scope_description": value["scope_description"],
                      "evidence_strength": value["evidence_strength"], "confidence_calibrated": False,
                      "strict_geometry_accepted": False, "fallback_reason": assessment.get("reason"),
                      "model_receipt": receipt}
            break
        except (ValueError, KeyError, OSError, RuntimeError, ValidationError) as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    else:
        result = {**assessment, "q": 0.0, "status": "fallback", "score_type": "default_sufficiency",
                  "scope": "unresolved_visual_scope", "estimation_method": "no_evidence_operational_default",
                  "evidence_strength": "unsupported", "confidence_calibrated": False,
                  "strict_geometry_accepted": False, "reason": "numeric_output_required_but_no_usable_estimate",
                  "fallback_reason": assessment.get("reason"), "estimation_errors": errors,
                  "default_is_geometric_measurement": False}
    write_json(output / "assessment.json", result)
    return result
