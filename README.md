# ARC-Net

PyTorch implementation of **ARC-Net: Adaptive Riemannian Covariance for Contrastive Lie Algebra Learning in Ultra-Fine-Grained Visual Categorization**.

ARC-Net replaces the vanilla sample covariance inside CLA-Net's Lie-algebra branch with a new SPD descriptor — **Adaptive Riemannian Shrinkage Covariance (ARS-Cov)** — that combines:

1. learnable spatial-channel attention reweighting,
2. Tyler-style robust M-estimation (`T` unrolled iterations), and
3. closed-form Ledoit–Wolf shrinkage to an affine-invariant target on the SPD manifold.

## File layout

```
arcnet/
  ars_cov.py            # ARS-Cov module (the new covariance descriptor)
  lie_algebra.py        # Log-Euclidean utilities
  contrastive_loss.py   # Lie-algebra contrastive losses (CLA + InfoNCE)
  model.py              # ARC-Net (ResNet-50 backbone + ARS-Cov + heads)
  augmentations.py      # Two-view transforms (standard + shuffle/mask)
  dataset.py            # ImageFolder-style two-view dataset
train.py                # Training script
evaluate.py             # Standalone evaluation script
requirements.txt
```

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py \
    --data-root /path/to/Cotton80 \
    --num-classes 80 \
    --batch-size 8 \
    --epochs 200 \
    --lambda-ss 0.1 \
    --image-size 384 \
    --tyler-iters 3 \
    --spd-channels 64
```

Default schedule: SGD, momentum 0.9, weight decay 5e-4, base lr 0.1, step decay at 110/160/190.

## Evaluate

```bash
python evaluate.py \
    --data-root /path/to/Cotton80 \
    --num-classes 80 \
    --checkpoint ./checkpoints/best.pt
```

## Quick sanity check (no data required)

```bash
python -m arcnet.ars_cov
```

This builds a small ARS-Cov layer, runs a synthetic forward + backward pass, prints the smallest eigenvalue of `V*` (positive), the shrinkage coefficient `α*`, and the input-gradient norm.
