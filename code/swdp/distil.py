"""一致性蒸馏: 把技能条件 DP 蒸馏为 1/2/4 步可用策略(Consistency Policy 路线)。

修复说明(2026-09-03): 旧实现用 x0 = (a - sigma*eps)/alpha 直接预测干净动作,
高噪声区 alpha≈0.0026 使误差放大 ~400 倍, 训练初期 loss 发散至 1e20。
改为 Consistency Models 标准参数化:

    F_theta(z, tau) = c_skip(tau) * z + c_out(tau) * G_theta(z, tau, obs, s)
    c_skip(tau) = 1 / (1 + sigma^2)          # tau=1 (sigma=0) 时 =1 -> 边界恒等
    c_out(tau)  = sigma / sqrt(1 + sigma^2)  # tau=1 时 =0

- 训练对: 在离散 tau 水平上, 学生输出在相邻噪声水平间自一致:
    L = || F(a_{t_{n+1}}, t_{n+1}) - F_EMA(a_{t_n}, t_n) ||^2
  其中 a_{t_n} 由教师(源 DDPM)一步 DDIM 反传得到(共同噪声、时间对齐)。
- 采样: 1 步 = F(z, tau_min); 多步 = 一致性多步采样。
- 所有量均在单位尺度(无 1/alpha 放大), 训练稳定。
"""
import copy

import torch
import torch.nn as nn

from .nets import SkillConditionalMLP, ALPHA, SIGMA
from .policy import SkillDP


def c_skip_c_out(tau):
    """CM 参数化系数(连续 VP 调度)。"""
    sg = SIGMA(tau)                       # [B, 1]
    c_skip = 1.0 / (1.0 + sg ** 2)
    c_out = sg / torch.sqrt(1.0 + sg ** 2)
    return c_skip, c_out


class ConsistencyStudent(SkillDP):
    """一致性学生: F(z, tau) = c_skip*z + c_out*G(z, tau, obs, s) -> 干净动作块。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _f_impl(self, a, tau, obs, s):
        g = self.net(a, tau, obs, s)               # [B, H, da]
        c_skip, c_out = c_skip_c_out(tau)
        c_skip = c_skip.unsqueeze(-1)
        c_out = c_out.unsqueeze(-1)
        f = c_skip * a + c_out * g
        # tau≈1 边界条件: 干净输入恒等(c_skip=1, c_out=0 已保证, 显式兜底数值精度)
        if (tau >= 0.999).any():
            f = torch.where(tau.unsqueeze(-1) >= 0.999, a, f)
        return f

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
        z = torch.randn(B, self.horizon, self.act_dim, device=obs.device,
                        generator=gen)
        z = SIGMA(torch.tensor(tau_min, device=obs.device)) * z
        ts = torch.linspace(tau_min, 1.0, n_steps + 1, device=obs.device)
        for i in range(n_steps):
            t = ts[i]
            x0 = self.f(z, t.expand(B, 1), obs, s)
            if i == n_steps - 1:
                return x0
            t_next = ts[i + 1]
            z = ALPHA(t_next) * x0 + SIGMA(t_next) * torch.randn(
                x0.shape, device=x0.device, generator=gen)
        return x0


def distill(teacher: SkillDP, n_iter: int = 20000, batch: int = 256,
            lr: float = 1e-4, tau_min: float = 0.05, n_levels: int = 24,
            ema_decay: float = 0.995, device: str = "cuda",
            loader=None, save_path: str = None, log_every: int = 2000):
    """把 teacher(DDPM) 蒸馏为一致性学生(CM 参数化)。

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
            B = a0.shape[0]
            # 随机选相邻水平: t_hi 为更噪(小 tau), t_lo 为更干净(大 tau)。
            # 一致性链从数据侧(tau=1 边界 F(z,1)=z)锚定, 向噪声侧传播:
            #   F(z_hi, t_hi) = EMA F(z_lo, t_lo),  z_lo 由教师 ODE 一步从 z_hi 反传。
            idx = torch.randint(0, n_levels, (1,), device=device).item()
            t_hi = levels[idx]        # 更噪
            t_lo = levels[idx + 1]    # 更干净
            # 上水平加噪(共同噪声, 时间对齐)
            eps = torch.randn(a0.shape, device=a0.device)
            a_hi = ALPHA(t_hi) * a0 + SIGMA(t_hi) * eps
            # 教师一步 ODE 反传到 t_lo(仅在教师内部做 1/alpha, 不参与学生梯度)
            with torch.no_grad():
                eps_hat = teacher.net(a_hi, t_hi.expand(B, 1), obs, s)
                x0_est = (a_hi - SIGMA(t_hi) * eps_hat) / \
                    ALPHA(t_hi).clamp(min=1e-3)
                a_lo = ALPHA(t_lo) * x0_est + SIGMA(t_lo) * eps_hat
                target = ema.f(a_lo, t_lo.expand(B, 1), obs, s)
            out = student.f_train(a_hi, t_hi.expand(B, 1), obs, s)
            loss = nn.functional.mse_loss(out, target.detach())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 10.0)
            opt.step()
            # EMA 更新
            with torch.no_grad():
                for p_s, p_e in zip(student.parameters(), ema.parameters()):
                    p_e.mul_(ema_decay).add_(p_s, alpha=1 - ema_decay)
            log.append(float(loss))
            if it % log_every == 0:
                print(f"[distill] iter {it} loss {log[-1]:.6f} "
                      f"(avg {sum(log[-500:]) / min(500, len(log)):.6f})")
    # 用 EMA 权重做最终学生
    student.load_state_dict(ema.state_dict())
    if save_path:
        student.save(save_path)
    return student
