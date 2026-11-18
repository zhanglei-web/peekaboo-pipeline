"""Inference-only replacements for the upstream pre-PyTorch-2 ROI extension.

No learned layers are changed. Legacy ROIAlign has no half-pixel offset and
legacy NMS uses inclusive pixel boxes. The adapter is installed before upstream
model imports, leaving the checked-out upstream source untouched.
"""
import sys
import types


def inclusive_nms(boxes, scores, threshold):
    from torchvision.ops import nms

    continuous = boxes.clone()
    continuous[:, 2:] += 1
    return nms(continuous, scores, threshold)


def install_roi_compat():
    from torch import nn
    from torchvision.ops import RoIPool, roi_align, roi_pool

    class LegacyROIAlign(nn.Module):
        def __init__(self, output_size, spatial_scale, sampling_ratio):
            super().__init__()
            self.output_size = output_size
            self.spatial_scale = spatial_scale
            self.sampling_ratio = sampling_ratio

        def forward(self, features, rois):
            return roi_align(features, rois, self.output_size,
                             self.spatial_scale, self.sampling_ratio, aligned=False)

    def legacy_roi_align(features, rois, output_size, spatial_scale, sampling_ratio):
        return roi_align(features, rois, output_size, spatial_scale,
                         sampling_ratio, aligned=False)

    if "model.roi_layers" in sys.modules:
        raise RuntimeError("ROI compatibility must be installed before model imports")
    module = types.ModuleType("model.roi_layers")
    module.nms = inclusive_nms
    module.ROIAlign = LegacyROIAlign
    module.ROIPool = RoIPool
    module.roi_align = legacy_roi_align
    module.roi_pool = roi_pool
    sys.modules[module.__name__] = module


DESCRIPTION = {
    "backend": "torchvision.ops",
    "roi_align_aligned": False,
    "roi_align_sampling_ratio": 0,
    "nms_boxes": "inclusive_xyxy; add 1 to x2,y2 before torchvision NMS",
    "scope": "inference only; learned modules and checkpoint keys unchanged",
    "limitations": "Tie ordering may differ; no compiled-upstream numerical parity claim",
}
