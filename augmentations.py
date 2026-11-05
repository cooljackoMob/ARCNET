"""
augmentations.py
================

Two-view augmentations used by ARC-Net:

  * View A: standard fine-grained augmentation (random crop, flip,
    colour jitter).
  * View B: random patch shuffling and random patch masking, as in
    MaskCOV / CLA-Net. Patch grid size and mask ratio are configurable.
"""

from __future__ import annotations

import random

import torch
from torch import Tensor
from torchvision import transforms as T


def _split_into_patches(img: Tensor, grid: int) -> list[Tensor]:
    """Split a (C, H, W) tensor into grid*grid contiguous patches."""
    C, H, W = img.shape
    assert H % grid == 0 and W % grid == 0, "H and W must be divisible by grid."
    ph, pw = H // grid, W // grid
    patches = []
    for i in range(grid):
        for j in range(grid):
            patches.append(img[:, i * ph:(i + 1) * ph, j * pw:(j + 1) * pw])
    return patches


def _stitch_patches(patches: list[Tensor], grid: int) -> Tensor:
    """Inverse of _split_into_patches."""
    rows = [torch.cat(patches[i * grid:(i + 1) * grid], dim=-1) for i in range(grid)]
    return torch.cat(rows, dim=-2)


class ShuffleAndMask:
    """View-B transform: shuffle patches with prob `p_shuffle` and mask each
    patch independently with prob `p_mask`."""

    def __init__(self, grid: int = 7, p_shuffle: float = 1.0, p_mask: float = 0.3):
        self.grid = grid
        self.p_shuffle = p_shuffle
        self.p_mask = p_mask

    def __call__(self, img: Tensor) -> Tensor:
        patches = _split_into_patches(img, self.grid)
        if random.random() < self.p_shuffle:
            random.shuffle(patches)
        if self.p_mask > 0:
            patches = [
                torch.zeros_like(p) if random.random() < self.p_mask else p
                for p in patches
            ]
        return _stitch_patches(patches, self.grid)


class TwoViewTransform:
    """Wraps a base transform and returns two views per input."""

    def __init__(self, image_size: int = 384, grid: int = 8, p_mask: float = 0.3):
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        self.view_A = T.Compose([
            T.Resize((image_size + 56, image_size + 56)),
            T.RandomCrop(image_size),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.2, 0.2, 0.2, 0.05),
            T.ToTensor(),
            normalize,
        ])
        self.view_B_pre = T.Compose([
            T.Resize((image_size + 56, image_size + 56)),
            T.RandomCrop(image_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ])
        self.view_B_shuffle = ShuffleAndMask(grid=grid, p_mask=p_mask)
        self.view_B_norm = normalize

    def __call__(self, img):
        x_A = self.view_A(img)
        x_B = self.view_B_pre(img)
        x_B = self.view_B_shuffle(x_B)
        x_B = self.view_B_norm(x_B)
        return x_A, x_B


def eval_transform(image_size: int = 384):
    return T.Compose([
        T.Resize((image_size + 56, image_size + 56)),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
