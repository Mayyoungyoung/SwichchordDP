"""一致性蒸馏: 把技能条件 DP 蒸馏为 1/2/4 步可用策略(Consistency Policy 路线)。

学生网络: f_theta(a_t, tau, obs, s) -> a0 预测(一致性函数, 输出干净动作块)。
- 边界条件: f_theta(a, tau=1, ·) = a(干净输入恒等)。
- 训练对: 在离散 tau 序列上, 学生输出在相邻噪声水平间自一致:
    L = || f(a_{t_{n+1}}, t_{n+1}) - f_EMA(a_{t_n}, t_n) ||^2
  其中 a_{t_n} 由教师(源 DDPM)一步 DDIM 反传得到。
- 采样: 1 步 = f(z, tau_min); 多步 = 一致性多步采样。
"""
import copy

import torch
import torch.nn as nn

from .nets import SkillConditionalMLP, ALPHA, SIGMA
from .policy import SkillDP


class ConsistencyStudent(SkillDP):
    """一致性学生: 输出 x0 预测。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _f_impl(self, a, tau, obs, s):
        eps_hat = self.net(a, tau, obs, s)
        al = ALPHA(tau).unsqueeze(-1)
        sg = SIGMA(tau).unsqueeze(-1)
        x0 = (a - sg * eps_hat) / al.clamp(min=1e-3)
        if (tau >= 0.999).any():
            x0 = torch.where(tau.unsqueeze(-1) >= 0.999, a, x0)
        return x0

    def f_train(self, a, tau, obs, s):
        """训练用(保留梯度)。"""
        return self._f_impl(a, tau, obs, s)

    @torch.no_grad()
    def f(self, a: torch.Tensor, tau: torch.Tensor, obs: torch.Tensor,
          s: torch.Tensor):
        """一致性函数(推理, 无梯度)。"""
        return self._f_impl(a, tau, obs, s)

    @torch.no_grad()
    def sample(self, obs: torch.Tensor, s: torch.Tensor, n_steps: int = 1,
               tau_min: float = 0.05, seed: int | None = None):
        """一致性采样: n_steps=1 即一步策略。"""
        B = obs.shape[0]
        gen = torch.Generator(device=obs.device)
        if seed is not None:
            gen.manual_seed(seed)
        z = torch.randn(B, self.horizon, self.act_dim, device=obs.device, generator=gen)
        z = SIGMA(torch.tensor(tau_min, device=obs.device)) * z
        ts = torch.linspace(tau_min, 1.0, n_steps + 1, device=obs.device)
        for i in range(n_steps):
            t = ts[i]
            x0 = self.f(z, t.expand(B, 1), obs, s)
            t_next = ts[i + 1]
            if i == n_steps - 1:
                return x0
            z = ALPHA(t_next) * x0 + SIGMA(t_next) * torch.randn(
                x0.shape, device=x0.device, generator=gen)
        return x0


def distill(teacher: SkillDP, n_iter: int = 20000, batch: int = 256,
            lr: float = 1e-4, tau_min: float = 0.05, n_levels: int = 24,
            ema_decay: float = 0.995, device: str = "cuda",
            loader=None, save_path: str = None, log_every: int = 2000):
    """把 teacher(DDPM) 蒸馏为一致性学生。

    loader: 可迭代的数据集, 每次 yield (a0, obs, s) batch。
    """
    student = ConsistencyStudent(
        teacher.act_dim, teacher.horizon, teacher.obs_dim, teacher.n_skills,
        device=device)
    ema = copy.deepcopy(student).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(student.parameters(), lr=lr)

    # 离散 tau 水平(不含 1.0, 含 tau_min)
    levels = torch.linspace(tau_min, 1.0, n_levels + 1, device=device)
    log = []
    it = 0
    while it < n_iter:
        for a0, obs, s in loader:
            if it >= n_iter:
                break
            it += 1
            a0 = a0.to(device); obs = obs.to(device); s = s.to(device)
            # 随机选相邻水平
            idx = torch.randint(0, n_levels, (1,), device=device).item()
            t_hi = levels[idx + 1]
            t_lo = levels[idx]
            B = a0.shape[0]
            # 上水平加噪
            eps = torch.randn(a0.shape, device=a0.device)
            a_hi = ALPHA(t_hi) * a0 + SIGMA(t_hi) * eps
            # 教师一步 DDIM 反传到 t_lo
            with torch.no_grad():
                eps_hat = teacher.net(a_hi, t_hi.expand(B, 1), obs, s)
                x0_est = (a_hi - SIGMA(t_hi) * eps_hat) / ALPHA(t_hi).clamp(min=1e-3)
                a_lo = ALPHA(t_lo) * x0_est + SIGMA(t_lo) * eps_hat
                target = ema.f(a_lo, t_lo.expand(B, 1), obs, s)
            out = student.f_train(a_hi, t_hi.expand(B, 1), obs, s)
            loss = nn.functional.mse_loss(out, target.detach())
            opt.zero_grad()
            loss.backward()
            opt.step()
            # EMA 更新
            with torch.no_grad():
                for p_s, p_e in zip(student.parameters(), ema.parameters()):
                    p_e.mul_(ema_decay).add_(p_s, alpha=1 - ema_decay)
            log.append(float(loss))
            if it % log_every == 0:
                print(f"[distill] iter {it} loss {log[-1]:.5f}")
    # 用 EMA 权重做最终学生
    student.load_state_dict(ema.state_dict())
    if save_path:
        student.save(save_path)
    return student
