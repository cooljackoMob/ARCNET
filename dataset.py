"""
dataset.py
==========

A folder-style ImageFolder dataset that returns two augmented views per
image, plus a thin wrapper for the ultra-FGVC benchmark layout
(`<root>/<split>/<class>/<image>`).
"""

from __future__ import annotations

import os
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset


class TwoViewImageFolder(Dataset):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,   # returns (x_A, x_B)
        extensions=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"),
    ):
        self.root = root
        self.transform = transform
        classes = sorted(d.name for d in os.scandir(root) if d.is_dir())
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.classes = classes
        self.samples: list[tuple[str, int]] = []
        for cls_name in classes:
            cls_dir = os.path.join(root, cls_name)
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith(extensions):
                    self.samples.append(
                        (os.path.join(cls_dir, fname), self.class_to_idx[cls_name])
                    )
        if not self.samples:
            raise RuntimeError(f"No images found under {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is None:
            return img, label
        out = self.transform(img)
        if isinstance(out, tuple):  # two-view transform
            x_A, x_B = out
            return x_A, x_B, label
        return out, label
