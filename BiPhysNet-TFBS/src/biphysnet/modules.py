"""Reusable neural-network modules for BiPhysNet-TFBS."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """Attention pooling over sequence positions."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_weights = nn.Linear(hidden_size, hidden_size)
        self.attention_query = nn.Parameter(torch.randn(hidden_size))

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None, output_attentions: bool = False):
        uit = torch.tanh(self.attention_weights(hidden_states))
        scores = torch.matmul(uit, self.attention_query)
        if attention_mask is not None:
            mask_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(attention_mask == 0, mask_value)
        weights = F.softmax(scores, dim=1)
        context_vector = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)
        if output_attentions:
            return context_vector, weights
        return context_vector


def make_se_layer(dim: int, reduction: int = 4) -> nn.Sequential:
    """Squeeze-and-excitation style channel recalibration."""
    mid = max(8, dim // reduction)
    return nn.Sequential(nn.Linear(dim, mid), nn.ReLU(), nn.Linear(mid, dim), nn.Sigmoid())
