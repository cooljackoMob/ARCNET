"""
contrastive_loss.py
===================

Contrastive Lie-algebra losses for ARC-Net.

We implement two flavours:
  * `LieAlgebraSSLoss`: the simple loss of CLA-Net (Eq. 7 in the paper),
        L_ss = sum_i [ ||T^A_i - T'^B_i||^2 + ||T_i - T'_i||_1 ].
  * `InfoNCELieAlgebra`: an InfoNCE-style contrastive loss in the Lie
        algebra (vectorised) for stronger negative mining.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lie_algebra import vectorise_sym, lie_algebra_distance, lie_algebra_l1


class LieAlgebraSSLoss(nn.Module):
    """Self-supervised Lie-algebra loss as in CLA-Net (Eq. 6 / Eq. 7)."""

    def forward(
        self,
        T_A: torch.Tensor,            # (B, C, C) ground-truth embedding from view A
        T_B: torch.Tensor,            # (B, C, C) ground-truth embedding from view B
        T_A_pred: torch.Tensor,       # (B, C, C) predicted embedding from view A
        T_B_pred: torch.Tensor,       # (B, C, C) predicted embedding from view B
    ) -> torch.Tensor:
        # Positive-pair Frobenius alignment.
        positive = lie_algebra_distance(T_A, T_B_pred)
        # Element-wise Lie-algebra reconstruction.
        recon = lie_algebra_l1(T_A, T_A_pred) + lie_algebra_l1(T_B, T_B_pred)
        return (positive + recon).mean()


class InfoNCELieAlgebra(nn.Module):
    """
    InfoNCE in the Lie algebra: vectorise the symmetric embeddings,
    L2-normalise, and compute a temperature-scaled cosine softmax.
    """

    def __init__(self, temperature: float = 0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, T_A: torch.Tensor, T_B: torch.Tensor) -> torch.Tensor:
        v_A = F.normalize(vectorise_sym(T_A), dim=-1)
        v_B = F.normalize(vectorise_sym(T_B), dim=-1)
        logits = (v_A @ v_B.T) / self.temperature                          # (B, B)
        labels = torch.arange(v_A.size(0), device=v_A.device)
        return 0.5 * (F.cross_entropy(logits, labels)
                      + F.cross_entropy(logits.T, labels))
