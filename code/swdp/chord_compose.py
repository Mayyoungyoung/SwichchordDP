"""ChordCompose: 免训技能组合算法(Chord 组合场)。

Switch/Chain/Combine 三个原子算子 + 时间掩码 + 单步可行性投影。
核心公式(已按 ChordEdit 原文核实, 见 docs/survey.md):

    R(a, tau)   = E_{z~K_tau}[ B_tau ( eps_hat(z,tau,o,s') - eps_hat(z,tau,o,s) ) ]
    û = [ t·R(t−δ) + δ·R(t) ] / (t+δ)
    a ← a + λ·û · mask

变体(消融用):
- naive(硬切): û = R(t)(δ=0)
- eff_shift(有效时间步平移): û = R(t−δ)
- energy_compose(GSC 式分数叠加): eps = Σ w_i eps_hat(·, s_i), 做一步去噪更新
"""
import torch

from .nets import ALPHA, SIGMA, b_t_epsilon
from .feasibility import prox_feasible


@torch.no_grad()
def residual_field_x0(dp, a_anchor, tau, obs, s_from, s_to, n_noise=1,
                      rng=None, weights=None):
    """x0 空间残差场(一致性模型): R = E_z[ F(z,tau,c') - F(z,tau,c) ], B_t ≡ I。

    对 CM 类模型, F(z, tau) 直接输出干净动作块, 条件差即动作空间编辑方向,
    无需 B_t 映射(等效于 ChordEdit 的速度参数化情形)。
    """
    B = a_anchor.shape[0]
    device = a_anchor.device
    gen = torch.Generator(device=device)
    if rng is not None:
        gen.manual_seed(int(rng))
    alpha = ALPHA(torch.as_tensor(tau, device=device))
    sigma = SIGMA(torch.as_tensor(tau, device=device))
    acc = torch.zeros_like(a_anchor)
    for _ in range(n_noise):
        eps = torch.randn(a_anchor.shape, device=device, generator=gen)
        z = alpha * a_anchor + sigma * eps

        def fz(s):
            if s is None:
                return torch.zeros_like(a_anchor)
            return dp.f(z, torch.full((B, 1), tau, device=device), obs, s)

        if weights is not None:
            diff = sum(w * fz(s) for w, s in weights)
        else:
            diff = fz(s_to) - fz(s_from)
        acc = acc + diff
    return acc / n_noise


@torch.no_grad()
def residual_field(dp, a_anchor, tau, obs, s_from, s_to, n_noise=1,
                   rng=None, weights=None):
    """可观测残差场 R(a_anchor, tau) = E_z[ B_tau * (Q(z,s_to) - Q(z,s_from)) ]。

    - a_anchor: 干净动作块锚点 [B, H, da]
    - s_from/s_to: 技能 one-hot [B, n_skills](可 None 表示无条件)
    - weights: 若非 None, 则为 (w_from, w_to) 的权重列表, 用于 Combine:
      Q(z, s) 替换为 Σ_i w_i Q(z, s_i)。
    """
    B = a_anchor.shape[0]
    device = a_anchor.device
    gen = torch.Generator(device=device)
    if rng is not None:
        gen.manual_seed(int(rng))
    alpha = ALPHA(torch.as_tensor(tau, device=device))
    sigma = SIGMA(torch.as_tensor(tau, device=device))
    Bt = b_t_epsilon(torch.as_tensor(tau, device=device))

    acc = torch.zeros_like(a_anchor)
    for _ in range(n_noise):
        eps = torch.randn(a_anchor.shape, device=device, generator=gen)
        z = alpha * a_anchor + sigma * eps

        def qz(s):
            if s is None:
                return torch.zeros_like(a_anchor)
            return dp.Q(z, torch.full((B, 1), tau, device=device), obs, s)

        if weights is not None:
            # Combine: 条件输出 = Σ_i w_i Q(z, s_i)
            q_to = sum(w * qz(s) for w, s in weights)
            q_from = None
        else:
            q_to = qz(s_to)
            q_from = qz(s_from)
        if q_from is None:
            diff = q_to
        else:
            diff = q_to - q_from
        acc = acc + Bt * diff
    return acc / n_noise


@torch.no_grad()
def chord_field(dp, a_anchor, tau, delta, obs, s_from, s_to, n_noise=1,
                rng=None, mode="chord", weights=None, x0_space=False):
    """Chord 控制场 û。

    mode:
    - "chord":     û = [t·R(t−δ) + δ·R(t)]/(t+δ)
    - "naive":     û = R(t)             (硬切)
    - "eff_shift": û = R(t−δ)           (有效时间步平移消融)
    - "energy":    GSC 式分数叠加, 直接返回叠加后的噪声(调用方自行去噪)
    """
    rf = residual_field_x0 if x0_space else residual_field
    if mode == "chord":
        r_prev = rf(dp, a_anchor, tau - delta, obs, s_from, s_to,
                    n_noise, rng, weights)
        r_cur = rf(dp, a_anchor, tau, obs, s_from, s_to,
                   n_noise, rng, weights)
        w_prev = tau
        w_cur = delta
        u = (w_prev * r_prev + w_cur * r_cur) / (tau + delta)
        return u, dict(r_prev=r_prev, r_cur=r_cur)
    elif mode == "naive":
        r_cur = rf(dp, a_anchor, tau, obs, s_from, s_to, n_noise, rng, weights)
        return r_cur, dict(r_cur=r_cur)
    elif mode == "eff_shift":
        r_prev = rf(dp, a_anchor, tau - delta, obs, s_from, s_to,
                    n_noise, rng, weights)
        return r_prev, dict(r_prev=r_prev)
    elif mode == "energy":
        # GSC 式: 直接对叠加分数做一步 x0 估计更新(等价于乘积专家方向)
        B = a_anchor.shape[0]
        device = a_anchor.device
        gen = torch.Generator(device=device)
        if rng is not None:
            gen.manual_seed(int(rng))
        eps = torch.randn(a_anchor.shape, device=device, generator=gen)
        z = ALPHA(torch.as_tensor(tau, device=device)) * a_anchor + \
            SIGMA(torch.as_tensor(tau, device=device)) * eps
        q_sum = sum(w * dp.Q(z, torch.full((B, 1), tau, device=device), obs, s)
                    for w, s in weights)
        # 用叠加噪声做一步 x0 估计作为"编辑方向"
        x0 = (z - SIGMA(torch.as_tensor(tau, device=device)) * q_sum) / \
             ALPHA(torch.as_tensor(tau, device=device)).clamp(min=1e-3)
        u = x0 - a_anchor
        return u, dict()
    elif mode == "chord_recon":
        # Chord 平滑 + 一步 x0 重建(GSC 强度 + Chord 稳定性):
        # 对「噪声预测差」做 Chord 时间加权平均, 加到源技能分数上做 x0 重建。
        # 与 eps 空间 chord 的区别: 平滑对象是原始 Δeps(无 B_t 放大), 且输出是重建而非扰动。
        B = a_anchor.shape[0]
        device = a_anchor.device
        gen = torch.Generator(device=device)
        if rng is not None:
            gen.manual_seed(int(rng))
        t = torch.as_tensor(tau, device=device)
        td = torch.as_tensor(tau - delta, device=device)
        acc = torch.zeros_like(a_anchor)
        for _ in range(n_noise):
            eps = torch.randn(a_anchor.shape, device=device, generator=gen)
            z = ALPHA(t) * a_anchor + SIGMA(t) * eps
            z_prev = ALPHA(td) * a_anchor + SIGMA(td) * eps
            tt = torch.full((B, 1), tau, device=device)
            ttp = torch.full((B, 1), tau - delta, device=device)
            q_a = dp.Q(z, tt, obs, s_from)
            d_cur = dp.Q(z, tt, obs, s_to) - q_a
            d_prev = dp.Q(z_prev, ttp, obs, s_to) - dp.Q(z_prev, ttp, obs, s_from)
            d_chord = (tau * d_prev + delta * d_cur) / (tau + delta)
            x0 = (z - SIGMA(t) * (q_a + d_chord)) / ALPHA(t).clamp(min=1e-3)
            acc = acc + (x0 - a_anchor)
        u = acc / n_noise
        return u, dict()


def temporal_mask(horizon: int, boundary: int, width: int, device: str = "cuda"):
    """时间掩码: 只在交接边界附近 width 个动作步施加场, 其余不动。"""
    mask = torch.zeros(horizon, device=device)
    lo = max(0, boundary - width)
    hi = min(horizon, boundary + width)
    mask[lo:hi] = 1.0
    return mask.view(1, -1, 1)


@torch.no_grad()
def switch(dp, obs, a_anchor, s_from, s_to, tau=0.9, delta=0.15, lam=1.0,
           n_noise=1, mode="chord", mask=None, use_proj=False, seed=None,
           x0_space=False):
    """Switch: 技能切换场(一次单步传输)。

    mode="energy"(GSC 式): 对两个技能的分数做乘积专家组合(等权重), 一步 x0 估计更新。
    """
    weights = [(1.0, s_from), (1.0, s_to)] if mode == "energy" else None
    u, info = chord_field(dp, a_anchor, tau, delta, obs, s_from, s_to,
                          n_noise, seed, mode, weights=weights,
                          x0_space=x0_space)
    if mask is None:
        mask = 1.0
    a_new = a_anchor + lam * u * mask
    if use_proj:
        a_new = prox_feasible(a_new)
    return a_new, dict(field=u, energy=float((u ** 2).mean()), **info)


@torch.no_grad()
def chain(dp, obs, a0, skill_seq, boundaries, tau=0.9, delta=0.15, lam=1.0,
          n_noise=1, mode="chord", mask_width=4, use_proj=False, seed=None):
    """Chain: 长程串联 = 多次 Switch(每个交接边界一次 Chord 单步传输)。

    - skill_seq: 技能 one-hot 列表 [s1, ..., sK](K 个技能)
    - boundaries: 交接边界(动作块内的步索引), len = K-1
    """
    a = a0
    total_energy = 0.0
    for k, (s_from, s_to) in enumerate(zip(skill_seq[:-1], skill_seq[1:])):
        mask = temporal_mask(a.shape[1], boundaries[k], mask_width, a.device)
        a, info = switch(dp, obs, a, s_from, s_to, tau, delta, lam, n_noise,
                         mode, mask, use_proj, seed + k if seed else None)
        total_energy += info["energy"]
    return a, total_energy


@torch.no_grad()
def combine(dp, obs, a_anchor, skill_weights, tau=0.9, delta=0.15, lam=1.0,
            n_noise=1, mode="chord", use_proj=False, seed=None):
    """Combine: 并发叠加(乘积专家)后 Chord 平滑。skill_weights = [(w_i, s_i)]。"""
    u, info = chord_field(dp, a_anchor, tau, delta, obs, None, None,
                          n_noise, seed, mode, weights=skill_weights)
    a_new = a_anchor + lam * u
    if use_proj:
        a_new = prox_feasible(a_new)
    return a_new, dict(field=u, energy=float((u ** 2).mean()), **info)
