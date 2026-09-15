import torch
import torch.nn as nn
import torch.nn.functional as F




class IB_Module(nn.Module):

    def __init__(self, input_dim=768, hidden_dim=512, latent_dim=128, kl_beta=1e-4, sample_size=5):
        super().__init__()
        intermediate_dim = (hidden_dim + input_dim) // 2
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, hidden_dim),
            nn.ReLU()
        )
        self.emb2mu = nn.Linear(hidden_dim, latent_dim)
        self.emb2std = nn.Linear(hidden_dim, latent_dim)

        # 先验分布参数（与类别无关）
        self.mu_p = nn.Parameter(torch.randn(latent_dim))
        self.std_p = nn.Parameter(torch.randn(latent_dim))
        self.kl_beta = kl_beta
        self.sample_size = sample_size
        self.latent_dim = latent_dim

    def estimate(self, emb):
        """估计分布的均值和标准差"""
        mu = self.emb2mu(emb)
        std = F.softplus(self.emb2std(emb))
        return mu, std

    def kl_div(self, mu_q, std_q):
        """计算KL散度"""
        batch_size = mu_q.size(0)
        mu_p = self.mu_p.view(1, -1).expand(batch_size, -1)
        std_p = F.softplus(self.std_p.view(1, -1).expand(batch_size, -1))
        # 防止 std 塌缩导致 1/std^2 反向梯度溢出 fp32
        std_q = torch.clamp(std_q, min=1e-4)
        std_p = torch.clamp(std_p, min=1e-4)

        k = mu_q.size(1)
        mu_diff = mu_p - mu_q
        mu_diff_sq = torch.mul(mu_diff, mu_diff)
        logdet_std_q = torch.sum(2 * torch.log(torch.clamp(std_q, min=1e-8)), dim=1)
        logdet_std_p = torch.sum(2 * torch.log(torch.clamp(std_p, min=1e-8)), dim=1)
        fs = torch.sum(torch.div(std_q ** 2, std_p ** 2), dim=1) + torch.sum(torch.div(mu_diff_sq, std_p ** 2), dim=1)
        kl_divergence = (fs - k + logdet_std_p - logdet_std_q) * 0.5
        return kl_divergence.mean()

    def reparameterize(self, mu, std):
        """重参数化采样"""
        batch_size = mu.shape[0]
        z = torch.randn(self.sample_size, batch_size, mu.shape[1]).to(mu.device)
        return mu + std * z

    def forward(self, x, training=True):
        """前向传播"""
        x = self.mlp(x) #clsout
        mu, std = self.estimate(x) # cls均值和std

        # 重参数化采样
        if training:
            z = self.reparameterize(mu, std)
        else:
            z = mu.unsqueeze(0)  # 测试时使用均值

        # 计算KL损失
        kl_loss = self.kl_div(mu, std)
        return z, mu, kl_loss * self.kl_beta
    # ========================================