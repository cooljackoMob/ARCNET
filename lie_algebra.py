"""
lie_algebra.py
==============

Lie group / Lie algebra utilities used by ARC-Net.

We work with the abelian Lie group of d x d Symmetric Positive Definite
matrices Sym+(d) endowed with the Log-Euclidean metric:

    V1 . V2 = exp(log V1 + log V2)
    d_LE(V1, V2) = || log V1 - log V2 ||_F

Because the Lie algebra at the identity is the space of symmetric matrices,
we vectorise its upper triangle (including the diagonal) to obtain a
d(d+1)/2-dimensional embedding suitable for contrastive learning.
"""

from __future__ import annotations

import torch


def vectorise_sym(L: torch.Tensor) -> torch.Tensor:
    """
    Vectorise the upper triangle (with diagonal) of a batch of symmetric
    matrices.

    Args:
        L: (B, C, C) symmetric matrices (e.g. Lie algebra embeddings).
    Returns:
        v: (B, C*(C+1)/2) flattened vectors. Off-diagonal entries are
           multiplied by sqrt(2) so that ||v||^2 = ||L||_F^2.
    """
    B, C, _ = L.shape
    # Symmetrise.
    L = 0.5 * (L + L.transpose(-1, -2))
    # Pre-compute index masks once, on the same device/dtype.
    rows, cols = torch.triu_indices(C, C, offset=0, device=L.device)
    diag_mask = (rows == cols).to(L.dtype)
    scale = diag_mask + (1.0 - diag_mask) * (2.0 ** 0.5)                   # (P,)
    v = L[:, rows, cols] * scale                                            # (B, P)
    return v


def lie_algebra_distance(T1: torch.Tensor, T2: torch.Tensor) -> torch.Tensor:
    """
    Squared Frobenius distance in the Lie algebra (Log-Euclidean metric).

    Args:
        T1, T2: (B, C, C) Lie algebra embeddings.
    Returns:
        (B,) per-sample squared distance.
    """
    diff = T1 - T2
    return (diff * diff).sum(dim=(-2, -1))


def lie_algebra_l1(T1: torch.Tensor, T2: torch.Tensor) -> torch.Tensor:
    """L1 distance in the Lie algebra (used in L_ss as the prediction term)."""
    return (T1 - T2).abs().sum(dim=(-2, -1))
