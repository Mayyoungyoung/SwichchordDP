"""技能条件扩散网络: MLP 骨干 + FiLM 技能调制。

约定(对齐 ChordEdit): 连续时间 tau ∈ [0,1], tau=1 对应干净动作块,
tau=0 对应纯噪声。forward 核 a_noisy = alpha(tau)*a + sigma(tau)*eps。
"""
import math

import torch
import torch.nn as nn


def get_schedule(gamma_max: float = 12.5):
    """线性对数 SNR 调度(VP)。返回 alpha/sigma 关于 tau 的可微闭式函数。

    gamma(tau) = gamma_max * (1 - tau)
    alpha(tau) = exp(-gamma(tau)/2), sigma(tau) = sqrt(1 - alpha^2)
    """
    def alpha(tau):
        return torch.exp(-0.5 * gamma_max * (1.0 - tau))

    def sigma(tau):
        a = torch.exp(-0.5 * gamma_max * (1.0 - tau))
        return torch.sqrt(torch.clamp(1.0 - a * a, min=1e-6))

    return alpha, sigma


# B_t 闭式系数(epsilon 预测, VP 调度): A_t^(eps) = -alpha_dot/(alpha*sigma) = gamma_max/(2*sigma)
# (alpha_dot = alpha * gamma_max / 2, 见 docs/survey.md 附录 C)
ALPHA, SIGMA = get_schedule()


def b_t_epsilon(tau):
    """可观测残差映射系数 A_t^(eps) = gamma_max / (2 * sigma(tau))。"""
    return 6.25 / SIGMA(tau)  # gamma_max/2 = 6.25


class FiLM(nn.Module):
    """技能 one-hot -> FiLM 调制参数(对每个隐藏层输出 gamma/beta)。"""

    def __init__(self, skill_dim: int, cond_dim: int, n_layers: int):
        super().__init__()
        self.cond = nn.Sequential(
            nn.Linear(skill_dim, cond_dim),
            nn.ReLU(),
            nn.Linear(cond_dim, 2 * cond_dim * n_layers),
        )
        self.n_layers = n_layers
        self.cond_dim = cond_dim

    def forward(self, s: torch.Tensor):
        out = self.cond(s)  # [B, 2*cond*n_layers]
        gamma, beta = out.chunk(2, dim=-1)
        return gamma.view(-1, self.n_layers, self.cond_dim), \
            beta.view(-1, self.n_layers, self.cond_dim)


class SkillConditionalMLP(nn.Module):
    """技能条件去噪网络: eps_hat = f(a_noisy, tau, obs, s)。

    - a_noisy: 动作块 [B, H, da](展平后输入)
    - obs: 观测 [B, d_obs]
    - tau: 连续时间 [B, 1]
    - s: 技能 one-hot [B, n_skills]
    - 输出: 预测噪声 [B, H, da]
    """

    def __init__(self, act_dim: int, horizon: int, obs_dim: int,
                 n_skills: int, hidden: int = 512, n_layers: int = 4):
        super().__init__()
        self.act_dim = act_dim
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.n_skills = n_skills
        a_dim = act_dim * horizon

        self.obs_enc = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU())
        self.film = FiLM(n_skills, hidden, n_layers)

        self.net = nn.ModuleList()
        d_in = a_dim + hidden + 1
        for i in range(n_layers):
            self.net.append(nn.Linear(d_in if i == 0 else hidden, hidden))
        self.head = nn.Linear(hidden, a_dim)

    def forward(self, a: torch.Tensor, tau: torch.Tensor, obs: torch.Tensor,
                s: torch.Tensor):
        B = a.shape[0]
        a_flat = a.flatten(1)
        tau = tau.to(a.dtype)
        obs_feat = self.obs_enc(obs)
        gamma, beta = self.film(s)  # [B, n_layers, hidden]
        h = torch.cat([a_flat, obs_feat, tau], dim=-1)
        for i, layer in enumerate(self.net):
            h = layer(h)
            h = h * gamma[:, i] + beta[:, i]
            h = torch.relu(h)
        return self.head(h).view(B, self.horizon, self.act_dim)


class TimeEmbed(nn.Module):
    """正弦时间嵌入(备用, 当前用标量 tau)。"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, tau: torch.Tensor):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=tau.device) / half)
        args = tau * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)
