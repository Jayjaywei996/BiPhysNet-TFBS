"""Loss functions used by BiPhysNet-TFBS."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SafeNTXentLoss(nn.Module):
    """Stable sequence-feature contrastive loss with device-safe labels."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = max(float(temperature), 1e-8)

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        batch_size = z_i.size(0)
        if batch_size <= 1:
            return torch.tensor(0.0, device=z_i.device, requires_grad=True)
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        logits = torch.matmul(z_i, z_j.T) / self.temperature
        labels = torch.arange(batch_size, dtype=torch.long, device=z_i.device)
        loss_a = F.cross_entropy(logits, labels)
        loss_b = F.cross_entropy(logits.T, labels)
        return (loss_a + loss_b) / 2


# Backward-compatible alias used by older scripts.
NTXentLoss = SafeNTXentLoss


class FocalLoss(nn.Module):
    """Binary focal loss for imbalanced TFBS classification."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.type_as(logits)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        at = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        loss = at * (1 - pt).pow(self.gamma) * bce_loss
        return loss.mean() if self.reduction == "mean" else loss.sum()


def bernoulli_kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """KL divergence between Bernoulli probabilities."""
    p = p.clamp(eps, 1 - eps)
    q = q.clamp(eps, 1 - eps)
    return p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))
