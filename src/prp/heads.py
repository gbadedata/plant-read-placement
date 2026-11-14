"""Projection head trained with contrastive loss on top of read embeddings."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Standardizer(nn.Module):
    """Fixed z-scoring with statistics from the training split (stored with the head)."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean.float())
        self.register_buffer("std", std.float().clamp_min(1e-6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 512, out_dim: int = 128, dropout: float = 0.1,
                 mean: torch.Tensor | None = None, std: torch.Tensor | None = None):
        super().__init__()
        self.in_dim, self.hidden, self.out_dim, self.dropout = in_dim, hidden, out_dim, dropout
        self.norm = Standardizer(mean if mean is not None else torch.zeros(in_dim),
                                 std if std is not None else torch.ones(in_dim))
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(self.norm(x)), dim=-1)

    def config(self) -> dict:
        return {"in_dim": self.in_dim, "hidden": self.hidden, "out_dim": self.out_dim, "dropout": self.dropout}

    @classmethod
    def from_state(cls, state: dict) -> ProjectionHead:
        cfg = state["config"]
        head = cls(**cfg)
        head.load_state_dict(state["state_dict"])
        head.eval()
        return head
