"""图像观测技能条件扩散策略(CNN-DP): agentview RGB + 低维 proprio。

与 SkillDP 完全同一套 DDPM 训练/DDIM 采样/查询接口(供 ChordCompose 免训组合),
仅观测编码器换为 CNN:

    obs = (img [B,3,128,128] ∈ [-1,1], prop [B,d_prop] 已归一化)
    feat = Fuse(CNN(img), MLP(prop)) -> SkillConditionalMLP(同 FiLM 干线)

设计动机(LIBERO 在线 e2e=0 的根因): 9~121 维状态 DP 的技能进度仅 0.21,
LIBERO-Long 的空间关系(多物体、相对位姿)需要图像观测。
"""
import torch
import torch.nn as nn

from .nets import SkillConditionalMLP, ALPHA, SIGMA, b_t_epsilon


class ConvEncoder(nn.Module):
    """128x128 RGB -> feat_dim(5 个 stride-2 卷积块 -> 4x4 -> 全连接)。"""

    def __init__(self, feat_dim: int = 512):
        super().__init__()

        def blk(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=2, padding=1),
                nn.GroupNorm(8, cout), nn.SiLU())

        self.net = nn.Sequential(blk(3, 32), blk(32, 64), blk(64, 128),
                                 blk(128, 256), blk(256, 256))
        self.head = nn.Sequential(nn.Linear(256 * 4 * 4, feat_dim), nn.SiLU())

    def forward(self, x):                    # [B,3,128,128]
        return self.head(self.net(x).flatten(1))


class ImageSkillDP(nn.Module):
    """CNN-DP: 接口与 SkillDP 对齐(Q/loss/sample/B_t/save/load)。"""

    def __init__(self, act_dim: int, horizon: int, prop_dim: int,
                 n_skills: int, img_feat: int = 512, hidden: int = 512,
                 n_layers: int = 4, gamma_max: float = 12.5,
                 device: str = "cuda"):
        super().__init__()
        self.act_dim, self.horizon = act_dim, horizon
        self.prop_dim, self.obs_dim = prop_dim, prop_dim
        self.n_skills, self.gamma_max = n_skills, gamma_max
        self.device = device
        self.encoder = ConvEncoder(img_feat)
        self.prop_enc = nn.Sequential(nn.Linear(prop_dim, 128), nn.ReLU())
        self.fuse = nn.Sequential(
            nn.Linear(img_feat + 128, hidden), nn.ReLU())
        self.net = SkillConditionalMLP(act_dim, horizon, hidden, n_skills,
                                       hidden, n_layers)
        self.to(device)

    def encode(self, obs):
        img, prop = obs
        return self.fuse(torch.cat([self.encoder(img), self.prop_enc(prop)], -1))

    @torch.no_grad()
    def Q(self, a, tau, obs, s):
        return self.net(a, tau, self.encode(obs), s)

    @torch.no_grad()
    def B_t(self, tau):
        return b_t_epsilon(tau)

    def loss(self, a0, obs, s, rng=None, tau_power: float = 1.0):
        B = a0.shape[0]
        tau = torch.rand(B, 1, device=a0.device, generator=rng)
        if tau_power != 1.0:
            tau = 1.0 - tau ** tau_power
        eps = torch.randn(a0.shape, device=a0.device, generator=rng)
        alpha = ALPHA(tau).unsqueeze(-1)
        sigma = SIGMA(tau).unsqueeze(-1)
        a_noisy = alpha * a0 + sigma * eps
        eps_hat = self.net(a_noisy, tau, self.encode(obs), s)
        return nn.functional.mse_loss(eps_hat, eps)

    @torch.no_grad()
    def sample(self, obs, s, n_steps: int = 10, tau1: float = 1.0,
               seed=None):
        B = obs[0].shape[0]
        gen = torch.Generator(device=obs[0].device)
        if seed is not None:
            gen.manual_seed(seed)
        obs_feat = self.encode(obs)
        a = torch.randn(B, self.horizon, self.act_dim, device=obs[0].device,
                        generator=gen)
        ts = torch.linspace(0.0, tau1, n_steps + 1, device=obs[0].device)
        for i in range(n_steps):
            t_cur, t_next = ts[i], ts[i + 1]
            eps_hat = self.net(a, t_cur.expand(B, 1), obs_feat, s)
            x0 = (a - SIGMA(t_cur) * eps_hat) / ALPHA(t_cur).clamp(min=1e-3)
            a = ALPHA(t_next) * x0 + SIGMA(t_next) * eps_hat
        return a

    def save(self, path, prop_mean=None, prop_std=None,
             act_mean=None, act_std=None):
        torch.save({"model": self.state_dict(),
                    "cfg": dict(act_dim=self.act_dim, horizon=self.horizon,
                                prop_dim=self.prop_dim,
                                n_skills=self.n_skills,
                                gamma_max=self.gamma_max),
                    "prop_mean": prop_mean, "prop_std": prop_std,
                    "act_mean": act_mean, "act_std": act_std}, path)

    @classmethod
    def load(cls, path, device="cuda"):
        ckpt = torch.load(path, map_location=device)
        model = cls(**ckpt["cfg"], device=device)
        model.load_state_dict(ckpt["model"])
        return model, ckpt
