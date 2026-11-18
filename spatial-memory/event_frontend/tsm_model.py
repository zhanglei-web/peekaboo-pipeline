"""Inference-only EPIC-KITCHENS RGB TSM, compatible with the official checkpoint.

Temporal shift and ResNet block placement follow Ji Lin, Chuang Gan and Song
Han's TSM, as distributed by epic-kitchens/C1-Action-Recognition-TSN-TRN-TSM.
The upstream repository and license are retained in third_party/epic_tsm.
No training framework, optical-flow model, or network download is needed here.
"""
from pathlib import Path
import csv

import numpy as np
import torch
from torch import nn
from torchvision import models, transforms

from .detector import sha256_file


CHECKPOINT_SHA256 = "b56b6e7210c89e7dfc93daadafa90fda9d4ea73dfec7e9d3399240bca664139c"
LABEL_SHA256 = {
    "verb": "aab8bf210d0234b6facb9b191ae402c03a5f8965d123586b4762d3468a85033e",
    "noun": "40f0a81c5bd06b33050df9f0bd17cd3c2fccd759f9581e4e5e04fec7562be893",
}


def load_labels(folder):
    labels = {}
    for task, count in (("verb", 97), ("noun", 300)):
        path = Path(folder) / f"EPIC_100_{task}_classes.csv"
        if sha256_file(path) != LABEL_SHA256[task]:
            raise ValueError(f"Unexpected EPIC {task} class file")
        rows = list(csv.DictReader(path.open()))
        if [int(r["id"]) for r in rows] != list(range(count)):
            raise ValueError(f"Invalid EPIC {task} class ordering")
        labels[task] = [r["key"] for r in rows]
    return labels


class TemporalShift(nn.Module):
    def __init__(self, net, segments=8, fold_div=8):
        super().__init__()
        self.net, self.segments, self.fold_div = net, segments, fold_div

    def forward(self, tensor):
        nt, channels, height, width = tensor.shape
        if nt % self.segments:
            raise ValueError("Frames must be grouped into complete TSM windows")
        sequence = tensor.reshape(-1, self.segments, channels, height, width)
        fold = channels // self.fold_div
        shifted = torch.zeros_like(sequence)
        shifted[:, :-1, :fold] = sequence[:, 1:, :fold]
        shifted[:, 1:, fold:2 * fold] = sequence[:, :-1, fold:2 * fold]
        shifted[:, :, 2 * fold:] = sequence[:, :, 2 * fold:]
        return self.net(shifted.reshape(nt, channels, height, width))


class TSMResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = models.resnet50(weights=None)
        for stage in (self.base_model.layer1, self.base_model.layer2,
                      self.base_model.layer3, self.base_model.layer4):
            for block in stage:
                block.conv1 = TemporalShift(block.conv1)
        self.base_model.fc = nn.Dropout(.7)
        self.new_fc = nn.Linear(2048, 397)

    def forward(self, windows):
        if windows.ndim != 5 or windows.shape[1:3] != (8, 3):
            raise ValueError("Expected B x 8 x 3 x H x W")
        vectors = self.base_model(windows.flatten(0, 1))
        return self.new_fc(vectors).reshape(len(windows), 8, 397).mean(1)


class TSMRecognizer:
    def __init__(self, checkpoint, labels_dir, device="cuda", threads=4):
        checkpoint = Path(checkpoint)
        if sha256_file(checkpoint) != CHECKPOINT_SHA256:
            raise ValueError("Unexpected TSM checkpoint; expected official EPIC RGB weights")
        self.labels = load_labels(labels_dir)
        torch.set_num_threads(threads)
        self.device = device
        self.model = TSMResNet50()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        cfg = payload["hyper_parameters"]
        expected = {"frame_count": 8, "segment_length": 1}
        if cfg["modality"] != "RGB" or any(cfg["data"][k] != v for k, v in expected.items()):
            raise ValueError("Unsupported TSM checkpoint configuration")
        self.preprocessing = cfg["data"]["preprocessing"]
        self.model.load_state_dict({k.removeprefix("model."): v
                                   for k, v in payload["state_dict"].items()}, strict=True)
        self.model.to(device).eval()
        # This official checkpoint was trained with std == mean. Do not replace
        # it with the usual ImageNet std, or silently stretch a nonsquare image.
        self.preprocess = transforms.Compose([
            transforms.Resize(cfg["data"]["test_augmentation"]["rescale_size"]),
            transforms.CenterCrop(self.preprocessing["input_size"]),
            transforms.ToTensor(),
            transforms.Normalize(self.preprocessing["mean"], self.preprocessing["std"]),
        ])

    def encode(self, windows):
        tensor = torch.stack([torch.stack([self.preprocess(im) for im in ims])
                              for ims in windows]).to(self.device)
        with torch.inference_mode():
            logits = self.model(tensor)
            verbs = logits[:, :97].softmax(-1).cpu().numpy()
            nouns = logits[:, 97:].softmax(-1).cpu().numpy()
        if not np.isfinite(verbs).all() or not np.isfinite(nouns).all():
            raise ValueError("Nonfinite TSM probabilities")
        return verbs, nouns
