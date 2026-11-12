"""Model adapters use fixed local weights; no remote code or implicit downloads."""
from contextlib import contextmanager
from pathlib import Path
import os
import subprocess
import time
import numpy as np
from .common import locked


@contextmanager
def gpu_lease(cfg):
    if cfg["device"] != "cuda:0":
        raise ValueError("This pilot supports one explicitly locked CUDA device: cuda:0")
    with locked(cfg["gpu_lock"]):
        pids = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-compute-apps=pid",
                                       "--format=csv,noheader,nounits"], text=True).strip()
        if pids:
            raise RuntimeError("GPU has a compute owner; leave it untouched and retry later: " + pids)
        free = int(subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=memory.free",
                                           "--format=csv,noheader,nounits"], text=True).strip())
        if free < cfg["min_free_gpu_mib"]:
            raise RuntimeError("Insufficient free GPU memory; no CPU/model fallback")
        yield


class Models:
    def __init__(self, cfg):
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        # Explicitly isolate Ultralytics settings from existing project settings.
        os.environ["YOLO_CONFIG_DIR"] = str(Path(cfg["output_dir"]) / "runtime" / "ultralytics")
        from ultralytics import FastSAM
        self.torch, self.cfg = torch, cfg
        torch.set_num_threads(4)
        self.processor = AutoProcessor.from_pretrained(cfg["grounding_dino_path"], local_files_only=True)
        self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(
            cfg["grounding_dino_path"], local_files_only=True, use_safetensors=True).to(cfg["device"]).eval()
        self.segmenter = FastSAM(cfg["fastsam_weights"])

    def infer(self, image, prompts):
        torch, cfg = self.torch, self.cfg
        result = {p: [] for p in prompts}
        tick = time.perf_counter()
        for start in range(0, len(prompts), cfg["prompts_per_pass"]):
            chunk = prompts[start:start+cfg["prompts_per_pass"]]
            caption = ""
            spans = []
            for phrase in chunk:
                begin = len(caption)
                caption += phrase + ". "
                spans.append((begin, begin+len(phrase)))
            inputs = self.processor(images=image, text=caption, return_tensors="pt")
            offsets = self.processor.tokenizer(caption, return_offsets_mapping=True)["offset_mapping"]
            if len(offsets) != inputs["input_ids"].shape[-1] or len(offsets) > 256:
                raise ValueError("Prompt tokenization mismatch or model text limit exceeded")
            inputs = inputs.to(cfg["device"])
            with torch.inference_mode():
                outputs = self.detector(**inputs)
            probs = outputs.logits[0].sigmoid().detach().cpu().numpy()
            boxes = outputs.pred_boxes[0].detach().cpu().numpy()
            # Score each phrase over its own tokens, avoiding decoded-label substring matching.
            for phrase, (lo, hi) in zip(chunk, spans):
                token_indices = [i for i,(a,b) in enumerate(offsets) if b>a and a<hi and b>lo]
                if not token_indices:
                    raise ValueError("No detector tokens for noun phrase")
                scores = probs[:,token_indices].mean(axis=1)
                for j in np.flatnonzero(scores >= cfg["detection_threshold"]):
                    cx,cy,w,h = boxes[j]
                    result[phrase].append({"box": [float((cx-w/2)*image.width), float((cy-h/2)*image.height),
                                                   float((cx+w/2)*image.width), float((cy+h/2)*image.height)],
                                           "score": float(scores[j]), "query_index": int(j),
                                           "prompt_chunk": start//cfg["prompts_per_pass"]})
        detection_seconds = time.perf_counter()-tick
        tick = time.perf_counter()
        with torch.inference_mode():
            prediction = self.segmenter.predict(source=image, device=cfg["device"],
                imgsz=cfg["fastsam_imgsz"], conf=cfg["fastsam_conf"], iou=cfg["fastsam_iou"],
                retina_masks=True, verbose=False, save=False, max_det=cfg["max_masks"])[0]
        if prediction.masks is None:
            masks = np.zeros((0,image.height,image.width), dtype=bool)
        else:
            masks = prediction.masks.data.detach().cpu().numpy() > .5
            if masks.shape[1:] != (image.height,image.width):
                raise ValueError("FastSAM masks do not match original raster dimensions")
        return result, masks, {"detection_seconds": detection_seconds,
            "segmentation_seconds": time.perf_counter()-tick, "prompt_count": len(prompts),
            "mask_count": len(masks), "mask_limit_reached": len(masks)>=cfg["max_masks"]}
