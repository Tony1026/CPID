"""EMA prototype bank for label-free generator-nuisance discovery.

K prototypes live in the L2-normalized z_n space. Fake samples get soft
cluster assignments q; a DEC-style sharpened target p_hat serves as the
pseudo-label. Prototypes are buffers updated only by EMA, while ``proto_loss``
updates the nuisance encoder through its distances to those fixed centers.
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


class PrototypeBank(nn.Module):
    def __init__(self, n_proto=8, dim=256, tau=0.1, ema_momentum=0.9):
        super().__init__()
        self.n_proto = n_proto
        self.dim = dim
        self.tau = tau
        self.ema_momentum = ema_momentum
        proto = torch.randn(n_proto, dim)
        proto = proto / proto.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        self.register_buffer("prototypes", proto)

    # -------------------------------------------------------------- helpers
    def _norm_prototypes(self):
        return self.prototypes / self.prototypes.norm(
            dim=-1, keepdim=True).clamp_min(1e-8)

    def assign(self, z_n):
        """z_n: [B, dim] (already L2-normalized). Returns (q, p_hat).

        q     : soft assignment [B, K] (with gradient)
        p_hat : sharpened pseudo-target [B, K] (detached, batch-balanced)
        """
        p = self._norm_prototypes()
        sim = z_n @ p.t() / self.tau              # [B, K]
        q = F.softmax(sim, dim=-1)
        sharp = q.pow(2)
        col = sharp.sum(dim=0, keepdim=True).clamp_min(1e-8)
        sharp = sharp / col                       # column-normalized
        p_hat = sharp / sharp.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return q, p_hat.detach()

    # --------------------------------------------------------------- losses
    def proto_loss(self, z_n):
        """KL(p_hat || q) for a set of (fake) features."""
        q, p_hat = self.assign(z_n)
        log_p = p_hat.clamp_min(1e-8).log()
        loss = F.kl_div(q.clamp_min(1e-8).log(), log_p, log_target=True,
                        reduction="batchmean")
        return loss, q, p_hat

    def balance_loss(self, q):
        """Push average batch occupancy toward uniform to avoid collapse."""
        avg = q.mean(dim=0)                       # [K]
        uniform = torch.full_like(avg, 1.0 / self.n_proto)
        return F.kl_div(avg.clamp_min(1e-8).log(), uniform, reduction="sum")

    # -------------------------------------------------------------- updates
    @torch.no_grad()
    def ema_update(self, z_n, p_hat):
        """Move each prototype toward the p_hat-weighted mean of assigned z_n."""
        weights = p_hat.t()                       # [K, B]
        num = weights @ z_n                       # [K, dim]
        den = weights.sum(dim=-1, keepdim=True)   # [K, 1]
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(num, op=dist.ReduceOp.SUM)
            dist.all_reduce(den, op=dist.ReduceOp.SUM)
        valid = den.squeeze(-1) > 0
        if not valid.any():
            return
        targets = self._norm_prototypes().clone()
        targets[valid] = num[valid] / den[valid]
        m = self.ema_momentum
        updated = self.prototypes.clone()
        updated[valid] = (m * self.prototypes[valid]
                          + (1.0 - m) * targets[valid])
        updated = updated / updated.norm(
            dim=-1, keepdim=True).clamp_min(1e-8)
        self.prototypes.copy_(updated)

    @torch.no_grad()
    def init_from_features(self, z_n, n_iter=10, seed=0):
        """k-means++ style re-initialization from fake features."""
        g = torch.Generator(device=z_n.device).manual_seed(seed)
        n = z_n.shape[0]
        idx = torch.randint(0, n, (self.n_proto,), generator=g, device=z_n.device)
        cents = z_n[idx].clone()
        for _ in range(n_iter):
            sim = z_n @ cents.t()
            assign = sim.argmax(dim=-1)
            for k in range(self.n_proto):
                mask = assign == k
                if mask.any():
                    cents[k] = z_n[mask].mean(dim=0)
            cents = cents / cents.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        self.prototypes.copy_(cents)

    # ------------------------------------------------------------ monitoring
    @torch.no_grad()
    def usage_stats(self, q):
        avg = q.mean(dim=0)
        entropy = -(avg * avg.clamp_min(1e-8).log()).sum().item()
        return {
            "entropy": entropy,
            "max_entropy": float(torch.log(torch.tensor(float(self.n_proto)))),
            "occupancy": avg.cpu().tolist(),
        }
