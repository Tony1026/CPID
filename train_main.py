#!/usr/bin/env python3
"""Public training entry for the CPID model.

The model and objective follow the implementation used for the paper. The
training loop accepts a caller-provided feature loader.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import torch
import torch.nn.functional as F

from module.image_model import ImageDRGDModel


@dataclass(frozen=True)
class PublicTrainConfig:
    """Method-level training settings."""

    feature_dim: int = 1024
    device: str = "cuda"
    grad_clip: float = 1.0


def build_model(config: PublicTrainConfig) -> ImageDRGDModel:
    """Construct the released model with implementation defaults."""
    return ImageDRGDModel(dim=config.feature_dim, minimal_lgnd=True).to(config.device)


def _authenticity_centers(model: ImageDRGDModel, z_a: torch.Tensor,
                          z_n: torch.Tensor, labels: torch.Tensor,
                          assignments: torch.Tensor) -> torch.Tensor:
    """Compute detached fake-group centers for one training stage."""
    groups = assignments.argmax(-1)
    centers = []
    for group in range(model.n_proto):
        selected = (labels == 1) & (groups == group)
        centers.append(z_a[selected].mean(0) if selected.any() else z_a.new_zeros(z_a.shape[-1]))
    return torch.stack(centers).detach()


def _intervene(model: ImageDRGDModel, z_a: torch.Tensor, z_n: torch.Tensor,
               labels: torch.Tensor, assignments: torch.Tensor,
               centers: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Apply the paper's cross-prototype translation to valid fake anchors."""
    fake = torch.where(labels == 1)[0]
    if fake.numel() < 2:
        return None, fake[:0]
    groups = assignments.argmax(-1)
    anchors = fake[torch.rand(fake.numel(), device=labels.device) < 0.5]
    kept, donors = [], []
    for anchor in anchors.tolist():
        candidates = fake[groups[fake] != groups[anchor]]
        if candidates.numel():
            kept.append(anchor)
            donors.append(candidates[torch.randint(candidates.numel(), (1,), device=labels.device)])
    if not kept:
        return None, fake[:0]
    anchors = torch.tensor(kept, device=labels.device)
    donors = torch.cat(donors).to(labels.device)
    delta = centers[groups[donors]] - centers[groups[anchors]]
    return z_a[anchors] + delta, anchors


def training_loss(model: ImageDRGDModel, features: torch.Tensor,
                  labels: torch.Tensor) -> Mapping[str, torch.Tensor]:
    """Compute the actual authenticity, prototype, and intervention losses."""
    z_a, _ = model.enc_auth(features, training=model.training)
    z_n = model.enc_nuis(features)
    logits = model.head_auth(z_a)
    losses: dict[str, torch.Tensor] = {"auth": F.cross_entropy(logits, labels)}

    fake = labels == 1
    if fake.any():
        proto_loss, soft_assignments, targets = model.proto.proto_loss(z_n[fake])
        losses["prototype"] = proto_loss + 0.5 * model.proto.balance_loss(soft_assignments)
        assignments, _ = model.proto.assign(z_n)
        centers = _authenticity_centers(model, z_a, z_n, labels, assignments)
        model.proto.ema_update(z_n[fake].detach(), targets.detach())
    else:
        losses["prototype"] = logits.new_zeros(())
        assignments = None
        centers = None

    if assignments is None or centers is None:
        losses["intervention"] = logits.new_zeros(())
        losses["consistency"] = logits.new_zeros(())
        return losses

    changed, anchors = _intervene(model, z_a, z_n, labels, assignments, centers)
    if changed is None:
        losses["intervention"] = logits.new_zeros(())
        losses["consistency"] = logits.new_zeros(())
        return losses

    changed_logits = model.head_auth(changed)
    losses["intervention"] = F.cross_entropy(changed_logits, labels[anchors])
    temperature = 2.0
    clean = F.log_softmax(logits[anchors] / temperature, dim=-1)
    perturbed = F.log_softmax(changed_logits / temperature, dim=-1)
    losses["consistency"] = temperature**2 * F.kl_div(
        clean, perturbed, log_target=True, reduction="batchmean")
    return losses


def train_epoch(model: ImageDRGDModel, loader: Iterable[Mapping[str, torch.Tensor]],
                optimizer: torch.optim.Optimizer, config: PublicTrainConfig) -> dict[str, float]:
    """Run one optimization epoch over a feature loader."""
    model.train()
    totals: dict[str, float] = {}
    steps = 0
    for batch in loader:
        features = batch["features"].to(config.device)
        labels = batch["label"].to(config.device)
        optimizer.zero_grad(set_to_none=True)
        losses = training_loss(model, features, labels)
        total = sum(losses.values())
        total.backward()
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        model.ema_update_prototypes()
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach())
        totals["total"] = totals.get("total", 0.0) + float(total.detach())
        steps += 1
    return {name: value / max(steps, 1) for name, value in totals.items()}


