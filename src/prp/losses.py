"""Supervised contrastive loss (Khosla et al., 2020), with optional taxonomy levels.

The hierarchical version adds genus- and family-level SupCon terms so that reads from congeneric
species are pulled together more than reads from distant lineages. That geometry is what lets an
unseen species land near its relatives instead of at a random point.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def supcon_loss(z: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """z: (B, D) embeddings (normalised inside); labels: (B,) ints. Anchors without positives are skipped."""
    z = F.normalize(z, dim=1)
    B = z.shape[0]
    sim = (z @ z.T) / temperature
    self_mask = torch.eye(B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(self_mask, float("-inf"))
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos = (labels[:, None] == labels[None, :]) & ~self_mask
    n_pos = pos.sum(dim=1)
    valid = n_pos > 0
    if not valid.any():
        return z.sum() * 0.0
    summed = log_prob.masked_fill(~pos, 0.0).sum(dim=1)
    return -(summed[valid] / n_pos[valid]).mean()


def hierarchical_supcon(z: torch.Tensor, species: torch.Tensor, genus: torch.Tensor, family: torch.Tensor,
                        temperature: float, lambda_genus: float, lambda_family: float) -> tuple[torch.Tensor, dict]:
    ls = supcon_loss(z, species, temperature)
    lg = supcon_loss(z, genus, temperature) if lambda_genus > 0 else z.sum() * 0.0
    lf = supcon_loss(z, family, temperature) if lambda_family > 0 else z.sum() * 0.0
    total = ls + lambda_genus * lg + lambda_family * lf
    return total, {"species": ls.item(), "genus": lg.item(), "family": lf.item()}
