"""Image-domain DRGD: label-free generator-nuisance discovery for
generalizable AI-generated image detection.

Branches (input F = [CLS; patch tokens], [B, 257, 1024], frozen CLIP-L/14):
    z_a -- authenticity branch (VIB), the only path used at inference
    z_n -- generator-nuisance branch, organized by K learnable prototypes
           WITHOUT any real generator labels
    z_s -- semantic branch, used to debias z_n / z_a via GRL probes

Training uses a GRL-warmup probe stage followed by frozen-head uniform
confusion over the branch encoders.
"""

import os

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function

from module.IB import IB_Module
from module.prototypes import PrototypeBank


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x, lambda_=1.0):
    return GradientReversalFunction.apply(x, lambda_)


def uniform_confusion_loss(logits):
    """Bounded adversarial objective used after the probe heads are frozen.

    ``KL(p || Uniform) = log(K) - H(p)`` is non-negative, reaches zero at a
    uniform prediction, and cannot diverge by driving a correct-class logit to
    negative infinity as reversed cross-entropy can.
    """
    probs = F.softmax(logits, dim=-1)
    return (probs * (probs.clamp_min(1e-8).log()
                     + torch.log(torch.tensor(
                         float(logits.shape[-1]), device=logits.device,
                         dtype=logits.dtype)))).sum(dim=-1).mean()


def assignment_js_loss(q, q_pair):
    """Jensen-Shannon divergence between two prototype assignments."""
    midpoint = 0.5 * (q + q_pair)
    left = F.kl_div(midpoint.clamp_min(1e-8).log(),
                    q.clamp_min(1e-8).log(), log_target=True,
                    reduction="batchmean")
    right = F.kl_div(midpoint.clamp_min(1e-8).log(),
                     q_pair.clamp_min(1e-8).log(), log_target=True,
                     reduction="batchmean")
    return 0.5 * (left + right)


class AttentionPool(nn.Module):
    """Single-head cross-attention pooling of token sequences."""

    def __init__(self, dim=768):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.scale = dim ** -0.5

    def forward(self, tokens):  # [B, N, D] -> [B, D]
        scores = torch.matmul(self.query, tokens.transpose(-1, -2)) * self.scale
        attn = torch.softmax(scores, dim=-1)          # [B, 1, N]
        return torch.matmul(attn, tokens).squeeze(1)


class FeatureFuser(nn.Module):
    """Shared-style fusion: concat(cls, attn_pool(patches)) -> hidden."""

    def __init__(self, dim=768, out_dim=768):
        super().__init__()
        self.pool = AttentionPool(dim)
        self.fuse = nn.Sequential(
            nn.Linear(dim * 2, out_dim),
            nn.ReLU(),
        )

    def forward(self, F_tokens):  # [B, 257, D]
        cls = F_tokens[:, 0]
        patches = F_tokens[:, 1:]
        pooled = self.pool(patches)
        return self.fuse(torch.cat([cls, pooled], dim=-1))


class BranchEncoder(nn.Module):
    """z_a authenticity encoder: fusion -> (VIB | deterministic) -> projection."""

    def __init__(self, dim=768, latent_dim=256, use_vib=True,
                 kl_beta=1e-5, ib_k=5):
        super().__init__()
        self.use_vib = use_vib
        self.fuser = FeatureFuser(dim, dim)
        if use_vib:
            self.ib = IB_Module(input_dim=dim, hidden_dim=512,
                                latent_dim=latent_dim, kl_beta=kl_beta,
                                sample_size=ib_k)
        else:
            self.det = nn.Sequential(nn.Linear(dim, 512), nn.ReLU(),
                                     nn.Linear(512, latent_dim))
        self.pro = nn.Linear(latent_dim, latent_dim)

    def forward(self, F_tokens, training=True):
        h = self.fuser(F_tokens)
        if self.use_vib:
            samples, _, kl = self.ib(h, training=training)
            z = samples.mean(dim=0) if training else samples.squeeze(0)
            return self.pro(z), kl
        return self.pro(self.det(h)), torch.zeros((), device=h.device)


class NuisanceEncoder(nn.Module):
    """z_n generator-nuisance encoder, deterministic and L2-normalized."""

    def __init__(self, dim=768, latent_dim=256):
        super().__init__()
        self.fuser = FeatureFuser(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 512), nn.ReLU(), nn.Linear(512, latent_dim))

    def forward(self, F_tokens):
        z = self.mlp(self.fuser(F_tokens))
        return F.normalize(z, dim=-1)


class SemanticEncoder(nn.Module):
    """z_s semantic encoder + semantic classifier head."""

    def __init__(self, dim=768, latent_dim=256, n_sem=64, dropout=0.5):
        super().__init__()
        self.fuser = FeatureFuser(dim, dim)
        self.proj = nn.Linear(dim, latent_dim)
        self.head = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.Dropout(dropout),
            nn.Linear(128, n_sem))

    def forward(self, F_tokens):
        z = self.proj(self.fuser(F_tokens))
        return z


class ImageDRGDModel(nn.Module):
    def __init__(self, dim=1024, latent_dim=256, n_proto=8, n_sem=64,
                 tau=0.1, alpha_perturb=0.3, prob_perturb=0.5,
                 beta_residual=0.05, lambda_grl=0.05,
                 lambda_pair=0.2, lambda_confusion=0.1,
                 kl_beta=1e-5, ib_k=5, dropout=0.5,
                 cons_temp=2.0, noise_std=0.05,
                 use_vib=True, use_prototypes=True, use_perturb=True,
                 use_grl=True, use_sem_debias=True, use_real_perturb=True,
                 use_nuisance_branch=True, use_semantic_branch=True,
                 use_cls_donor_matching=None, supervised_gen=False,
                 linear_probe=False, minimal_lgnd=False):
        super().__init__()
        if linear_probe and minimal_lgnd:
            raise ValueError("minimal LGND and linear probe are distinct variants")
        if minimal_lgnd:
            use_vib = False
            use_grl = False
            use_sem_debias = False
            use_real_perturb = False
            use_semantic_branch = False
            if use_cls_donor_matching is None:
                use_cls_donor_matching = True
            supervised_gen = False
            beta_residual = 0.0
            noise_std = 0.0
        self.dim = dim
        self.latent_dim = latent_dim
        self.n_sem = n_sem
        self.tau = tau
        self.alpha = alpha_perturb
        self.beta_residual = beta_residual
        self.prob_perturb = prob_perturb
        self.lambda_grl = lambda_grl
        self.lambda_pair = lambda_pair
        self.lambda_confusion = lambda_confusion
        self.adversarial_mode = "grl"
        self.adversarial_strength = lambda_grl
        self.cons_temp = cons_temp
        self.noise_std = noise_std
        self.use_vib = use_vib
        self.use_prototypes = use_prototypes
        self.use_perturb = use_perturb
        self.use_grl = use_grl
        self.use_sem_debias = use_sem_debias
        self.use_real_perturb = use_real_perturb
        self.use_nuisance_branch = use_nuisance_branch
        self.use_semantic_branch = use_semantic_branch
        self.use_cls_donor_matching = bool(use_cls_donor_matching)
        self.supervised_gen = supervised_gen
        self.linear_probe = linear_probe
        self.minimal_lgnd = minimal_lgnd
        self._last_cf_count = 0

        if linear_probe:
            self.n_proto = n_proto
            self.fuser = FeatureFuser(dim, dim)
            self.proj = nn.Linear(dim, latent_dim)
            self.head_auth = nn.Sequential(
                nn.Linear(latent_dim, 128), nn.Dropout(dropout),
                nn.Linear(128, 2))
            return

        if not use_nuisance_branch and use_prototypes:
            raise ValueError(
                "prototype discovery requires the nuisance branch")
        if not use_nuisance_branch and use_perturb:
            raise ValueError(
                "cross-prototype perturbation requires the nuisance branch")
        if not use_nuisance_branch and supervised_gen:
            raise ValueError(
                "supervised generator prediction requires the nuisance branch")
        if not use_semantic_branch and use_sem_debias:
            raise ValueError(
                "semantic debiasing requires the semantic branch")

        self.enc_auth = BranchEncoder(dim, latent_dim, use_vib=use_vib,
                                      kl_beta=kl_beta, ib_k=ib_k)
        self.enc_nuis = NuisanceEncoder(dim, latent_dim)
        self.enc_sem = SemanticEncoder(dim, latent_dim, n_sem=n_sem,
                                       dropout=dropout)

        self.head_auth = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.Dropout(dropout),
            nn.Linear(128, 2))
        self.head_cluster = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.Dropout(dropout),
            nn.Linear(128, n_proto))

        self.proto = PrototypeBank(n_proto=n_proto, dim=latent_dim, tau=tau)
        self.nuisance_to_auth = nn.Linear(latent_dim, latent_dim, bias=False)
        nn.init.eye_(self.nuisance_to_auth.weight)
        self.n_proto = n_proto

        if self.minimal_lgnd:
            self.enc_sem = None
            self.head_cluster = None
        if not self.use_nuisance_branch:
            for module in (self.enc_nuis, self.head_cluster,
                           self.nuisance_to_auth):
                if module is not None:
                    module.requires_grad_(False)
        if not self.use_semantic_branch and self.enc_sem is not None:
            self.enc_sem.requires_grad_(False)

    # ------------------------------------------------------------- inference
    def _z_a(self, F_tokens, training=False):
        if self.linear_probe:
            return self.proj(self.fuser(F_tokens))
        z, _ = self.enc_auth(F_tokens, training=training)
        return z

    def test_struct(self, F_tokens):
        """Inference path: image features -> z_a -> real/fake logits."""
        with torch.no_grad():
            z_a = self._z_a(F_tokens, training=False)
            return self.head_auth(z_a)

    def extract_features(self, F_tokens, semantic_tokens=None):
        with torch.no_grad():
            z_a = self._z_a(F_tokens, training=False)
            if self.linear_probe:
                z = z_a.cpu().numpy()
                return z, z, z
            z_n = (self.enc_nuis(F_tokens)
                   if self.use_nuisance_branch else None)
            z_s = (self.enc_sem(
                F_tokens if semantic_tokens is None else semantic_tokens)
                if self.use_semantic_branch else None)
            q, _ = (self.proto.assign(z_n)
                    if self.use_prototypes else (None, None))
        z_n_out = None if z_n is None else z_n.cpu().numpy()
        z_s_out = None if z_s is None else z_s.cpu().numpy()
        out = (z_a.cpu().numpy(), z_n_out, z_s_out)
        if q is not None:
            out = out + (q.argmax(dim=-1).cpu().numpy(),)
        return out

    # ------------------------------------------------------------- training
    def _counterfactual_perturb(self, z_a, z_n, labels, q, sem_ids=None,
                                semantic_embeddings=None):
        """Transfer a prototype-level nuisance direction into ``z_a``.

        Donors must belong to another latent cluster. Same-semantic donors are
        preferred; if none exist, the nearest frozen global-semantic embedding
        is used. The prototype displacement and donor-local residual are
        mapped into the authenticity space separately.
        """
        device = z_a.device
        self._last_cf_count = 0
        fake_idx = (labels == 1).nonzero(as_tuple=True)[0]
        if fake_idx.numel() < 2 or q is None:
            return None, None
        clusters = q.argmax(dim=-1)
        sel_mask = torch.rand(fake_idx.shape[0], device=device) < self.prob_perturb
        selected = fake_idx[sel_mask]
        if selected.numel() == 0:
            selected = fake_idx[:1]

        prototypes = self.proto._norm_prototypes()
        tilde_list, kept, donors = [], [], []
        for i in selected.tolist():
            ci = clusters[i].item()
            candidates = fake_idx[(fake_idx != i)
                                  & (clusters[fake_idx] != ci)]
            if candidates.numel() == 0:
                continue

            matched = candidates
            if sem_ids is not None and sem_ids[i] >= 0:
                same_sem = candidates[sem_ids[candidates] == sem_ids[i]]
                if same_sem.numel() > 0:
                    matched = same_sem

            if (matched.numel() == candidates.numel()
                    and semantic_embeddings is not None):
                semantic = F.normalize(semantic_embeddings.detach(), dim=-1)
                similarities = semantic[matched] @ semantic[i]
                donor = matched[similarities.argmax()].item()
            else:
                donor = matched[torch.randint(0, matched.numel(), (1,),
                                              device=device)].item()

            cj = clusters[donor].item()
            proto_direction = prototypes[cj] - prototypes[ci]
            mapped_direction = F.normalize(
                self.nuisance_to_auth(proto_direction), dim=-1)
            z_intervened = z_a[i] + self.alpha * mapped_direction
            if self.beta_residual != 0.0:
                donor_residual = z_n[donor] - prototypes[cj]
                mapped_residual = F.normalize(
                    self.nuisance_to_auth(donor_residual), dim=-1)
                z_intervened = (
                    z_intervened + self.beta_residual * mapped_residual)
            if self.noise_std != 0.0:
                z_intervened = (
                    z_intervened
                    + torch.randn_like(z_a[i]) * self.noise_std)
            tilde_list.append(z_intervened)
            kept.append(i)
            donors.append(donor)
        if not tilde_list:
            return None, None
        self._last_cf_donors = torch.tensor(donors, device=device)
        self._last_cf_count = len(kept)
        return torch.stack(tilde_list), torch.tensor(kept, device=device)

    def _adversarial_loss(self, head, features, targets):
        if self.adversarial_mode == "grl":
            return F.cross_entropy(
                head(grad_reverse(features, self.adversarial_strength)), targets)
        if self.adversarial_mode == "confusion":
            return self.adversarial_strength * uniform_confusion_loss(
                head(features))
        raise ValueError(f"unknown adversarial mode: {self.adversarial_mode}")

    def set_adversarial_mode(self, mode, strength=None):
        if mode not in {"grl", "confusion"}:
            raise ValueError("adversarial mode must be 'grl' or 'confusion'")
        self.adversarial_mode = mode
        if strength is None:
            strength = (self.lambda_grl if mode == "grl"
                        else self.lambda_confusion)
        if strength < 0:
            raise ValueError("adversarial strength must be non-negative")
        self.adversarial_strength = float(strength)

    def forward(self, F_tokens, labels, gen_ids=None, sem_ids=None,
                semantic_tokens=None, paired_tokens=None):
        device = F_tokens.device
        zero = lambda: torch.zeros((), device=device)
        losses = {}

        if self.linear_probe:
            h = self.fuser(F_tokens)
            logits = self.head_auth(self.proj(h))
            auth_loss = F.cross_entropy(logits, labels)
            if paired_tokens is not None:
                paired_h = self.fuser(paired_tokens)
                paired_logits = self.head_auth(self.proj(paired_h))
                auth_loss = 0.5 * (
                    auth_loss + F.cross_entropy(paired_logits, labels))
            losses["auth"] = auth_loss
            return losses

        semantic_tokens = F_tokens if semantic_tokens is None else semantic_tokens
        self._last_proto = None

        z_a, kl_a = self.enc_auth(F_tokens, training=self.training)
        z_n = (self.enc_nuis(F_tokens)
               if self.use_nuisance_branch else None)
        z_s = (self.enc_sem(semantic_tokens)
               if self.use_semantic_branch else None)

        logits = self.head_auth(z_a)
        auth_loss = F.cross_entropy(logits, labels)
        kl_loss = kl_a

        z_a_pair, z_n_pair = None, None
        if paired_tokens is not None:
            z_a_pair, kl_pair = self.enc_auth(
                paired_tokens, training=self.training)
            if self.use_nuisance_branch:
                z_n_pair = self.enc_nuis(paired_tokens)
            pair_logits = self.head_auth(z_a_pair)
            auth_loss = 0.5 * (
                auth_loss + F.cross_entropy(pair_logits, labels))
            kl_loss = 0.5 * (kl_loss + kl_pair)

        losses["auth"] = auth_loss
        losses["kl"] = 0.5 * kl_loss if self.use_vib else zero()

        fake_idx = (labels == 1).nonzero(as_tuple=True)[0]

        # ---- prototype discovery (label-free) / supervised upper bound ----
        q = None
        if self.use_prototypes and fake_idx.numel() > 0:
            if self.supervised_gen and gen_ids is not None:
                losses["proto"] = F.cross_entropy(
                    self.head_cluster(z_n[fake_idx]), gen_ids[fake_idx])
                losses["bal"] = zero()
                q, _ = self.proto.assign(z_n)
            else:
                proto_features = z_n[fake_idx]
                if z_n_pair is not None:
                    proto_features = torch.cat(
                        [proto_features, z_n_pair[fake_idx]], dim=0)
                l_proto, q_f, p_hat_f = self.proto.proto_loss(proto_features)
                losses["proto"] = l_proto
                losses["bal"] = 0.5 * self.proto.balance_loss(q_f)
                self._last_usage = self.proto.usage_stats(q_f)
                q, _ = self.proto.assign(z_n)
                self._last_proto = (proto_features.detach(), p_hat_f)
        else:
            losses["proto"] = zero()
            losses["bal"] = zero()
        losses["pair_js"] = zero()
        if (self.use_prototypes and not self.supervised_gen
                and z_n_pair is not None and fake_idx.numel() > 0):
            q_clean, _ = self.proto.assign(z_n[fake_idx])
            q_pair, _ = self.proto.assign(z_n_pair[fake_idx])
            losses["pair_js"] = self.lambda_pair * assignment_js_loss(
                q_clean, q_pair)

        # ---- counterfactual perturbation ----
        losses["pert_ce"] = zero()
        losses["cons"] = zero()
        losses["real_perturb"] = zero()
        if self.use_perturb and self.training:
            perturb_sem_ids = sem_ids if self.use_semantic_branch else None
            perturb_semantic_embeddings = (
                semantic_tokens[:, 0]
                if (self.use_semantic_branch
                    or self.use_cls_donor_matching) else None)
            z_tilde, selected = self._counterfactual_perturb(
                z_a, z_n, labels, q, sem_ids=perturb_sem_ids,
                semantic_embeddings=perturb_semantic_embeddings)
            if z_tilde is not None:
                pert_logits = self.head_auth(z_tilde)
                fake_labels = torch.ones(z_tilde.shape[0], dtype=torch.long,
                                         device=device)
                losses["pert_ce"] = F.cross_entropy(pert_logits, fake_labels)
                clean = F.log_softmax(logits[selected] / self.cons_temp, dim=-1)
                pert = F.log_softmax(pert_logits / self.cons_temp, dim=-1)
                losses["cons"] = F.kl_div(clean, pert, log_target=True,
                                          reduction="batchmean") * (self.cons_temp ** 2)
        if self.use_real_perturb and self.training:
            real_idx = (labels == 0).nonzero(as_tuple=True)[0]
            if real_idx.numel() > 0:
                z_r = z_a[real_idx]
                z_rp = z_r + torch.randn_like(z_r) * 0.02
                rp_labels = torch.zeros(z_rp.shape[0], dtype=torch.long,
                                        device=device)
                losses["real_perturb"] = 0.8 * F.cross_entropy(
                    self.head_auth(z_rp), rp_labels)

        # ---- GRL leakage removal ----
        losses["adv_na"] = zero()
        losses["adv_an"] = zero()
        if self.use_grl and self.use_nuisance_branch:
            losses["adv_na"] = self._adversarial_loss(
                self.head_auth, z_n, labels)
            if z_n_pair is not None:
                losses["adv_na"] = 0.5 * (
                    losses["adv_na"] + self._adversarial_loss(
                        self.head_auth, z_n_pair, labels))
            if (self.use_prototypes and not self.supervised_gen
                    and fake_idx.numel() > 0 and q is not None):
                _, p_hat_all = self.proto.assign(z_n)
                targets = p_hat_all[fake_idx].argmax(dim=-1)
                losses["adv_an"] = self._adversarial_loss(
                    self.head_cluster, z_a[fake_idx], targets)

        # ---- semantic branch + debiasing ----
        losses["sem"] = zero()
        losses["adv_ns"] = zero()
        losses["adv_as"] = zero()
        if (self.use_semantic_branch and sem_ids is not None
                and (sem_ids >= 0).any()):
            valid = sem_ids >= 0
            losses["sem"] = F.cross_entropy(
                self.enc_sem.head(z_s[valid]), sem_ids[valid])
            if self.use_sem_debias:
                if semantic_tokens is F_tokens:
                    z_a_sem = z_a
                    z_n_sem = z_n
                else:
                    z_a_sem, _ = self.enc_auth(
                        semantic_tokens, training=self.training)
                    z_n_sem = (self.enc_nuis(semantic_tokens)
                               if self.use_nuisance_branch else None)
                if self.use_nuisance_branch:
                    losses["adv_ns"] = self._adversarial_loss(
                        self.enc_sem.head, z_n_sem[valid], sem_ids[valid])
                losses["adv_as"] = 0.5 * self._adversarial_loss(
                    self.enc_sem.head, z_a_sem[valid], sem_ids[valid])
        return losses

    # -------------------------------------------------------------- utilities
    def ema_update_prototypes(self):
        if self.use_prototypes and not self.supervised_gen \
                and getattr(self, "_last_proto", None) is not None:
            z_f, p_hat = self._last_proto
            self.proto.ema_update(z_f, p_hat)

    def freeze_heads(self):
        """Freeze probe/classifier heads for bounded Stage-2 confusion."""
        self._heads_frozen = True
        heads = [self.head_auth]
        if self.head_cluster is not None:
            heads.append(self.head_cluster)
        if self.enc_sem is not None:
            heads.append(self.enc_sem.head)
        for m in heads:
            for p in m.parameters():
                p.requires_grad = False
            m.eval()

    def train(self, mode=True):
        super().train(mode)
        if mode and getattr(self, "_heads_frozen", False) \
                and not self.linear_probe:
            heads = [self.head_auth]
            if self.head_cluster is not None:
                heads.append(self.head_cluster)
            if self.enc_sem is not None:
                heads.append(self.enc_sem.head)
            for m in heads:
                m.eval()
        return self

    def adv_params(self):
        params = []
        modules = [self.enc_auth, self.enc_nuis, self.nuisance_to_auth]
        if self.enc_sem is not None:
            modules.extend([self.enc_sem.fuser, self.enc_sem.proj])
        for m in modules:
            params += [p for p in m.parameters() if p.requires_grad]
        return params

    def param_groups(self, backbone_lr=1e-5, head_lr=1e-4):
        adv = {id(p) for p in self.adv_params()}
        head, other = [], []
        for p in self.parameters():
            if not p.requires_grad or id(p) in adv:
                continue
            (head if p.dim() >= 1 else other).append(p)
        return [{"params": head + other, "lr": head_lr}]

    def save_model(self, save_path, epoch, step):
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.state_dict(),
                   os.path.join(save_path, f"model_weights{epoch}_{step}.pth"))
        config = {"dim": self.dim, "latent_dim": self.latent_dim,
                  "n_proto": self.n_proto, "n_sem": self.n_sem,
                  "tau": self.tau,
                  "alpha_perturb": self.alpha,
                  "beta_residual": self.beta_residual,
                  "prob_perturb": self.prob_perturb,
                  "cons_temp": self.cons_temp,
                  "noise_std": self.noise_std,
                  "lambda_grl": self.lambda_grl,
                  "lambda_pair": self.lambda_pair,
                  "lambda_confusion": self.lambda_confusion,
                  "use_vib": self.use_vib,
                  "use_prototypes": self.use_prototypes,
                  "use_perturb": self.use_perturb, "use_grl": self.use_grl,
                  "use_sem_debias": self.use_sem_debias,
                  "use_real_perturb": self.use_real_perturb,
                  "use_nuisance_branch": self.use_nuisance_branch,
                  "use_semantic_branch": self.use_semantic_branch,
                  "use_cls_donor_matching": self.use_cls_donor_matching,
                  "supervised_gen": self.supervised_gen,
                  "linear_probe": self.linear_probe,
                  "minimal_lgnd": self.minimal_lgnd}
        torch.save(config, os.path.join(
            save_path, f"model_weights{epoch}_{step}_config.pth"))
