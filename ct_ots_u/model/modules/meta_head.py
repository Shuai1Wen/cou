from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["MetaHead"]


class MetaHead(nn.Module):
    """Small network producing structured linear parameters from context."""

    def __init__(self, context_dim: int, d: int, r: int) -> None:
        super().__init__()
        hidden = max(64, context_dim * 2)
        self.backbone = nn.Sequential(
            nn.Linear(context_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.proj_U = nn.Linear(hidden, d * r)
        self.proj_V = nn.Linear(hidden, d * r)
        self.proj_W = nn.Linear(hidden, d * r)
        self.proj_log_gamma = nn.Linear(hidden, 1)
        self.d = d
        self.r = r

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if context.ndim != 2:
            raise ValueError("context must be a 2D tensor")
        emb = self.backbone(context)
        U = self.proj_U(emb).view(-1, self.d, self.r)
        V = self.proj_V(emb).view(-1, self.d, self.r)
        W = self.proj_W(emb).view(-1, self.d, self.r)
        log_gamma = self.proj_log_gamma(emb).view(-1, 1, 1)
        return U, V, W, log_gamma
