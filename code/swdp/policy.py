"""技能条件 DDPM 扩散策略。

约定(对齐 ChordEdit): 连续时间 tau ∈ [0,1], tau=1 干净动作块, tau=0 纯噪声。
- 训练: 采样 tau ~ U[0,1], 加噪, 回归噪声 eps。
- 推理: 从 tau=0 的噪声用 Euler/DDIM 反向积分到 tau=1(支持任意步数)。
- 查询 Q(a, tau, obs, s): 返回预测噪声 eps_hat(供 ChordCompose 使用)。
- 残差映射 B_t: A_t^(eps) = gamma_max / (2*sigma(tau))(见 nets.b_t_epsilon)。
"""
import numpy as np
import torch
import torch.nn as nn

from .nets import SkillConditionalMLP, ALPHA, SIGMA, b_t_epsilon


class SkillDP(nn.Module):
    def __init__(self, act_dim: int, horizon: int, obs_dim: int, n_skills: int,
                 hidden: int = 512, n_layers: int = 4, gamma_max: float = 12.5,
                 device: str = "cuda"):
        super().__init__()
        self.act_dim = act_dim
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.n_skills = n_skills
        self.gamma_max = gamma_max
        self.device = device
        self.net = SkillConditionalMLP(act_dim, horizon, obs_dim, n_skills,
                                       hidden, n_layers)
        self.to(device)

    # ---------- 前向/查询 ----------
    @torch.no_grad()
    def Q(self, a: torch.Tensor, tau: torch.Tensor, obs: torch.Tensor,
          s: torch.Tensor):
        """可观测输出: 预测噪声 eps_hat(a, tau, obs, s)。"""
        return self.net(a, tau, obs, s)

    @torch.no_grad()
    def B_t(self, tau: torch.Tensor):
        """可观测残差映射系数(epsilon 参数化)。"""
        return b_t_epsilon(tau)

    # ---------- 训练 ----------
    def loss(self, a0: torch.Tensor, obs: torch.Tensor, s: torch.Tensor,
             rng: torch.Generator | None = None, tau_power: float = 1.0):
        B = a0.shape[0]
        tau = torch.rand(B, 1, device=a0.device, generator=rng)
        if tau_power != 1.0:
            # 低噪声偏置: tau = 1 - u^power, 密度向 tau=1(低噪声)集中
            tau = 1.0 - tau ** tau_power
        eps = torch.randn(a0.shape, device=a0.device, generator=rng)
        alpha = ALPHA(tau).unsqueeze(-1)
        sigma = SIGMA(tau).unsqueeze(-1)
        a_noisy = alpha * a0 + sigma * eps
        eps_hat = self.net(a_noisy, tau, obs, s)
        return nn.functional.mse_loss(eps_hat, eps)

    # ---------- 推理(多步 Euler/DDIM) ----------
    @torch.no_grad()
    def sample(self, obs: torch.Tensor, s: torch.Tensor, n_steps: int = 10,
               tau0: float = 0.05, tau1: float = 1.0, seed: int | None = None,
               x0_anchor: torch.Tensor | None = None,
               tau_start: float = 0.0):
        """从 tau_start 反面积分到 tau1, 返回干净动作块 a_hat(tau1)。

        - x0_anchor: 若非 None, 则用 anchor 的干净估计初始化初始噪声(部分去噪起始)。
        - 采用 DDIM 风格的确定性 Euler 反向积分。
        """
        B = obs.shape[0]
        gen = torch.Generator(device=obs.device)
        if seed is not None:
            gen.manual_seed(seed)
        if x0_anchor is None:
            x0_anchor = torch.zeros(B, self.horizon, self.act_dim, device=obs.device)
        a = ALPHA(torch.tensor(tau_start, device=obs.device)) * x0_anchor + \
            SIGMA(torch.tensor(tau_start, device=obs.device)) * torch.randn(
                x0_anchor.shape, device=obs.device, generator=gen)
        ts = torch.linspace(tau_start, tau1, n_steps + 1, device=obs.device)
        for i in range(n_steps):
            t_cur = ts[i]
            t_next = ts[i + 1]
            eps_hat = self.net(a, t_cur.expand(B, 1), obs, s)
            # x0 估计
            x0 = (a - SIGMA(t_cur) * eps_hat) / ALPHA(t_cur).clamp(min=1e-3)
            # DDIM 一步: a_next = alpha(t_next)*x0 + sigma(t_next)*eps_pred_dir
            alpha_n = ALPHA(t_next)
            sigma_n = SIGMA(t_next)
            a = alpha_n * x0 + sigma_n * eps_hat
        return a

    # ---------- 保存/加载 ----------
    def save(self, path: str):
        torch.save({"model": self.state_dict(),
                    "cfg": dict(act_dim=self.act_dim, horizon=self.horizon,
                                obs_dim=self.obs_dim, n_skills=self.n_skills,
                                gamma_max=self.gamma_max)},
                   path)

    @classmethod
    def load(cls, path: str, device: str = "cuda"):
        ckpt = torch.load(path, map_location=device)
        cfg = ckpt["cfg"]
        model = cls(**cfg, device=device)
        model.load_state_dict(ckpt["model"])
        return model
