"""
model.py
========

ARC-Net: a contrastive Lie-algebra network for ultra-fine-grained visual
categorization, with the new ARS-Cov second-order descriptor.

The model has two branches that share the backbone:

  (a) Classification branch: backbone -> GAP -> linear classifier.
  (b) Self-supervised Lie-algebra branch:
        backbone -> ARS-Cov -> log -> classification-style head F(*; W_ss)
        that predicts the Lie algebra embedding of the *other* view.

At inference time only branch (a) is used, so the second-order descriptor
costs nothing in deployment.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm

from .ars_cov import ARSCov
from .lie_algebra import vectorise_sym


class ResNetBackbone(nn.Module):
    """ResNet-50 backbone returning the layer4 feature map (B, 2048, H, W)."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = tvm.ResNet50_Weights.DEFAULT if pretrained else None
        net = tvm.resnet50(weights=weights)
        self.stem = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool,
            net.layer1, net.layer2, net.layer3, net.layer4
        )
        self.out_channels = 2048

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


class ChannelReducer(nn.Module):
    """1x1 conv to reduce backbone channels before ARS-Cov.

    SPD descriptors are CxC; we keep C moderate (e.g. 64) so that the
    matrix logarithm is cheap and well-conditioned.
    """

    def __init__(self, in_channels: int, out_channels: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class LieAlgebraPredictor(nn.Module):
    """Predicts the upper-triangular vector of a Lie-algebra embedding from a
    pooled backbone feature."""

    def __init__(self, in_features: int, C: int, hidden: int = 512):
        super().__init__()
        self.C = C
        self.P = C * (C + 1) // 2
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.P),
        )

    def forward(self, x_pool: torch.Tensor) -> torch.Tensor:
        """Returns the predicted symmetric matrix (B, C, C) by un-vectorising."""
        B = x_pool.size(0)
        v = self.head(x_pool)
        # Re-build symmetric matrix from upper triangle vector.
        T_pred = torch.zeros(B, self.C, self.C,
                             device=v.device, dtype=v.dtype)
        rows, cols = torch.triu_indices(self.C, self.C, offset=0, device=v.device)
        diag_mask = (rows == cols).to(v.dtype)
        scale = diag_mask + (1.0 - diag_mask) * (2.0 ** -0.5)
        T_pred[:, rows, cols] = v * scale
        T_pred = T_pred + T_pred.transpose(-1, -2)
        T_pred[:, torch.arange(self.C), torch.arange(self.C)] *= 0.5  # remove diag double-count
        return T_pred


class ARCNet(nn.Module):
    """The full ARC-Net model."""

    def __init__(
        self,
        num_classes: int,
        spd_channels: int = 64,
        tyler_iters: int = 3,
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        self.backbone = ResNetBackbone(pretrained=pretrained_backbone)
        self.reducer = ChannelReducer(self.backbone.out_channels, spd_channels)
        self.ars_cov = ARSCov(in_channels=spd_channels, tyler_iters=tyler_iters)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.backbone.out_channels, num_classes)
        self.lie_predictor = LieAlgebraPredictor(
            in_features=self.backbone.out_channels, C=spd_channels
        )

    # ------------------------------------------------------------------ #
    def encode(self, x: torch.Tensor):
        """Returns the backbone feature map, GAP feature vector, and the
        SPD/Lie-algebra descriptors of the input image."""
        feat = self.backbone(x)                       # (B, 2048, h, w)
        feat_pool = self.gap(feat).flatten(1)         # (B, 2048)
        red = self.reducer(feat)                       # (B, C, h, w)
        V, T, aux = self.ars_cov(red)
        return feat, feat_pool, V, T, aux

    # ------------------------------------------------------------------ #
    def forward(self, x_A: torch.Tensor, x_B: torch.Tensor | None = None):
        """
        If x_B is None, run only the classification branch (inference path).
        Otherwise, run the full two-view forward pass.
        """
        if x_B is None:
            _, feat_pool_A, V_A, T_A, aux_A = self.encode(x_A)
            return {
                "logits": self.classifier(feat_pool_A),
                "V_A": V_A,
                "T_A": T_A,
                "aux_A": aux_A,
            }

        # Two-view path.
        _, feat_pool_A, V_A, T_A, aux_A = self.encode(x_A)
        _, feat_pool_B, V_B, T_B, aux_B = self.encode(x_B)

        # Predict Lie-algebra embeddings from each view.
        T_A_pred = self.lie_predictor(feat_pool_A)
        T_B_pred = self.lie_predictor(feat_pool_B)

        return {
            "logits_A": self.classifier(feat_pool_A),
            "logits_B": self.classifier(feat_pool_B),
            "T_A": T_A,
            "T_B": T_B,
            "T_A_pred": T_A_pred,
            "T_B_pred": T_B_pred,
            "V_A": V_A,
            "V_B": V_B,
            "aux_A": aux_A,
            "aux_B": aux_B,
        }
