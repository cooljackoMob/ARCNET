"""
ars_cov.py
==========

Adaptive Riemannian Shrinkage Covariance (ARS-Cov) for ARC-Net.

This module replaces the vanilla sample covariance used in CLA-Net's
Lie-algebra branch with a robust, well-conditioned, attention-weighted
SPD descriptor.

Pipeline (Algorithm 1 in the paper):
  1. Spatial-channel attention reweighting of token features.
  2. Tyler-style robust M-estimation (T unrolled iterations).
  3. Closed-form Ledoit-Wolf shrinkage to an affine-invariant target.
  4. Matrix logarithm onto the Lie algebra (Log-Euclidean).

All steps are differentiable end-to-end and produce an SPD matrix.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Helper: differentiable matrix logarithm via symmetric eigendecomposition.
# --------------------------------------------------------------------------- #
def sym_matrix_log(V: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Differentiable matrix logarithm for a batch of SPD matrices.

    Args:
        V: Tensor of shape (B, C, C). Must be symmetric and positive definite.
        eps: floor for eigenvalues to avoid log(0).

    Returns:
        log(V) of shape (B, C, C).
    """
    # Symmetrise to suppress numerical drift before eig decomposition.
    V = 0.5 * (V + V.transpose(-1, -2))
    eigvals, eigvecs = torch.linalg.eigh(V)
    eigvals = torch.clamp(eigvals, min=eps)
    log_eig = torch.diag_embed(torch.log(eigvals))
    return eigvecs @ log_eig @ eigvecs.transpose(-1, -2)


def sym_matrix_inverse(V: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Stable inverse of a batch of SPD matrices via eigendecomposition."""
    V = 0.5 * (V + V.transpose(-1, -2))
    eigvals, eigvecs = torch.linalg.eigh(V)
    eigvals = torch.clamp(eigvals, min=eps)
    inv_eig = torch.diag_embed(1.0 / eigvals)
    return eigvecs @ inv_eig @ eigvecs.transpose(-1, -2)


# --------------------------------------------------------------------------- #
# Lightweight spatial-channel attention head.
# --------------------------------------------------------------------------- #
class SpatialChannelAttention(nn.Module):
    """
    1x1 conv -> LayerNorm -> GELU -> 1x1 conv head producing per-token logits.
    Output is a softmax-normalised, spatially flattened weight vector.
    """

    def __init__(self, in_channels: int, hidden: int = 32):
        super().__init__()
        self.proj_in = nn.Conv2d(in_channels, hidden, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=hidden)
        self.act = nn.GELU()
        self.proj_out = nn.Conv2d(hidden, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            w: (B, S) non-negative weights with sum_k w_k = S, S = H*W.
        """
        B, _, H, W = x.shape
        S = H * W
        logits = self.proj_out(self.act(self.norm(self.proj_in(x))))  # (B, 1, H, W)
        logits = logits.view(B, S)
        w = F.softmax(logits, dim=-1) * S  # E[w] = 1, sum w = S
        return w


# --------------------------------------------------------------------------- #
# Adaptive Riemannian Shrinkage Covariance.
# --------------------------------------------------------------------------- #
class ARSCov(nn.Module):
    """
    Adaptive Riemannian Shrinkage Covariance descriptor.

    Args:
        in_channels: number of feature channels C of the backbone tokens.
        tyler_iters: number of unrolled Tyler M-estimation iterations (T).
        eps: numerical floor for eigenvalues.
        target: kind of shrinkage target. 'iso' = (tr V / C) * I.
        return_logV: if True, return the Lie-algebra embedding log(V*).
    """

    def __init__(
        self,
        in_channels: int,
        tyler_iters: int = 3,
        eps: float = 1e-5,
        target: str = "iso",
        return_logV: bool = True,
    ):
        super().__init__()
        if target not in ("iso",):
            raise ValueError(f"Unknown target: {target}")
        self.C = in_channels
        self.T = tyler_iters
        self.eps = eps
        self.target = target
        self.return_logV = return_logV
        self.attention = SpatialChannelAttention(in_channels)

    # ------------------------------------------------------------------ #
    # 1. Weighted mean and centred features.
    # ------------------------------------------------------------------ #
    @staticmethod
    def _weighted_mean(F_: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # F_: (B, S, C), w: (B, S)
        Wsum = w.sum(dim=-1, keepdim=True).clamp_min(1e-8)              # (B, 1)
        mean = (w.unsqueeze(-1) * F_).sum(dim=1) / Wsum                  # (B, C)
        return mean

    # ------------------------------------------------------------------ #
    # 2. Tyler-style robust M-estimation (Eq. 5 in the paper).
    # ------------------------------------------------------------------ #
    def _tyler_iteration(
        self,
        F_centered: torch.Tensor,   # (B, S, C)
        w: torch.Tensor,            # (B, S)
    ) -> torch.Tensor:
        B, S, C = F_centered.shape
        device, dtype = F_centered.device, F_centered.dtype
        Vt = torch.eye(C, device=device, dtype=dtype).expand(B, C, C).clone()

        Wsum = w.sum(dim=-1).clamp_min(1e-8)  # (B,)

        for _ in range(self.T):
            Vt_inv = sym_matrix_inverse(Vt, eps=self.eps)                # (B, C, C)
            # Mahalanobis distances d_k = x_k^T V^{-1} x_k, x_k = f_k - mu.
            # Equivalent formulation that is memory-friendly:
            tmp = F_centered @ Vt_inv                                     # (B, S, C)
            d = (tmp * F_centered).sum(dim=-1).clamp_min(self.eps)        # (B, S)
            coef = (w / d)                                                # (B, S)
            # Weighted outer-product accumulation:
            #   Vt+1 = (C / Wsum) * sum_k coef_k * f_k f_k^T
            weighted = F_centered * coef.unsqueeze(-1)                    # (B, S, C)
            Vnew = weighted.transpose(1, 2) @ F_centered                  # (B, C, C)
            Vnew = (self.C / Wsum.view(B, 1, 1)) * Vnew
            Vnew = 0.5 * (Vnew + Vnew.transpose(-1, -2))
            Vt = Vnew
        return Vt

    # ------------------------------------------------------------------ #
    # 3. Closed-form Ledoit-Wolf shrinkage on the SPD manifold (Eq. 6).
    # ------------------------------------------------------------------ #
    def _shrinkage(
        self,
        V_R: torch.Tensor,          # (B, C, C)
        F_centered: torch.Tensor,   # (B, S, C)
        w: torch.Tensor,            # (B, S)
    ):
        B, S, C = F_centered.shape
        # Affine-invariant target F* = (tr V_R / C) * I.
        trace_VR = torch.diagonal(V_R, dim1=-2, dim2=-1).sum(-1)         # (B,)
        mu_iso = (trace_VR / C).clamp_min(self.eps)                      # (B,)
        F_star = (mu_iso.view(B, 1, 1)
                  * torch.eye(C, device=V_R.device, dtype=V_R.dtype))     # (B, C, C)

        # Closed-form Ledoit-Wolf coefficients.
        # delta_hat^2 = || V_R - F* ||_F^2  (signal-target distance)
        diff = V_R - F_star
        delta2 = (diff * diff).sum(dim=(-2, -1)).clamp_min(self.eps)     # (B,)

        # beta_hat^2 estimates the variance of the sample covariance:
        # using weighted token outer products.
        Wsum = w.sum(dim=-1).clamp_min(1e-8)                              # (B,)
        # per-token outer product contributions:
        # y_k = f_k f_k^T - V_R
        # beta_hat^2 = (1 / Wsum^2) * sum_k w_k^2 * ||y_k||_F^2
        # Compute ||y_k||_F^2 efficiently:
        #   ||f_k f_k^T - V_R||_F^2
        # = ||f_k||^4 - 2 f_k^T V_R f_k + ||V_R||_F^2
        norm_f2 = (F_centered * F_centered).sum(dim=-1)                   # (B, S)
        VR_fk = F_centered @ V_R                                          # (B, S, C)
        f_VR_f = (VR_fk * F_centered).sum(dim=-1)                         # (B, S)
        VR_F2 = (V_R * V_R).sum(dim=(-2, -1))                             # (B,)
        per_token = (norm_f2 ** 2) - 2.0 * f_VR_f + VR_F2.unsqueeze(-1)
        per_token = per_token.clamp_min(0.0)
        beta2 = (w.pow(2) * per_token).sum(dim=-1) / (Wsum ** 2 + self.eps)

        # Optimal shrinkage in [0, 1].
        alpha = (beta2 / (beta2 + delta2)).clamp(0.0, 1.0)                 # (B,)

        a = (1.0 - alpha).view(B, 1, 1)
        b = alpha.view(B, 1, 1)
        V_star = a * V_R + b * F_star
        # Re-symmetrise.
        V_star = 0.5 * (V_star + V_star.transpose(-1, -2))
        return V_star, alpha

    # ------------------------------------------------------------------ #
    # 4. Forward pass: tokens -> SPD descriptor (and optionally Lie algebra).
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, C, H, W) backbone token feature map.
        Returns:
            V_star: (B, C, C) SPD descriptor.
            T:     (B, C, C) Lie-algebra embedding log(V*) if return_logV else None.
            aux:   dict with diagnostic tensors (alpha, attention).
        """
        B, C, H, W = x.shape
        assert C == self.C, f"Expected {self.C} channels, got {C}."
        S = H * W

        # Spatial-channel attention weights.
        w = self.attention(x)                                              # (B, S)

        # Reshape to tokens (B, S, C).
        F_ = x.flatten(2).transpose(1, 2)

        # Weighted mean and centred features.
        mu = self._weighted_mean(F_, w)                                    # (B, C)
        F_centered = F_ - mu.unsqueeze(1)                                   # (B, S, C)

        # Robust Tyler estimate.
        V_R = self._tyler_iteration(F_centered, w)                          # (B, C, C)

        # Riemannian shrinkage to affine-invariant target.
        V_star, alpha = self._shrinkage(V_R, F_centered, w)                 # (B, C, C)

        T = sym_matrix_log(V_star, eps=self.eps) if self.return_logV else None

        aux = {"alpha": alpha, "attention": w}
        return V_star, T, aux


# --------------------------------------------------------------------------- #
# Quick sanity check.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    torch.manual_seed(0)
    layer = ARSCov(in_channels=64, tyler_iters=3)
    x = torch.randn(2, 64, 12, 12, requires_grad=True)
    V, T, aux = layer(x)
    print("V*  :", V.shape, "min eig:", torch.linalg.eigvalsh(V).min().item())
    print("logV:", T.shape)
    print("alpha:", aux["alpha"])
    # Confirm differentiability.
    T.sum().backward()
    print("grad norm:", x.grad.norm().item())
