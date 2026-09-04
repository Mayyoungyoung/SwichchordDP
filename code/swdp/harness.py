"""SWDP Harness: 冻结技能库上的统一执行层(Plan-and-Compose 的低层 + 调度钩子)。

对标 Harness VLA 的系统层思路, 差异化: 衔接算子(ChordCompose)免训、调度信号轻量。
收敛 eval_compose / eval_recovery / eval_libero_online 三份重复的 rollout 主循环。

组件:
- SkillRuntime:    单技能执行封装(观测规范化/one-hot 条件/动作块采样/NFE 计数)
- TransitionSpec:  衔接配置(mode/tau/delta/lam/mask/proj + 候选数/选择器预留)
- RiskMonitor:     场能量尖峰检测(事件式重规划的触发器, 从 eval_recovery 抽出)
- ChainExecutor:   统一 rollout 主循环, 事件钩子:
  * on_boundary(ctx): 交接前回调(默认 anchor 采样 + cc.switch; 阶段 C 的候选
    重排序在此扩展), ctx 携带 anchor/o_t/s_from/s_to 供记录 Lipschitz 等
  * on_risk(spike, energy, state): 尖峰回调, 返回重规划目标技能索引或 None
  * action_filter(t, a_raw): 每步动作过滤(扰动注入点)
  * stop_fn(state): 提前退出
切换触发策略(switch_policy):
  - "fixed":     固定步数(默认, 现有协议)
  - "criterion": 技能完成判据(skill_done_fn)提前切换 + 超时保护(消融臂 C4)
"""
from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np
import torch

from . import chord_compose as cc


@dataclass
class TransitionSpec:
    """技能衔接配置(ChordCompose 参数 + 阶段 C 候选选择预留)。"""

    mode: str = "chord"            # chord/naive/eff_shift/energy/chord_recon
    tau: float = 0.9
    delta: float = 0.15
    lam: float = 0.3
    n_noise: int = 1
    use_mask: bool = True
    mask_width: int = 4
    use_proj: bool = False
    x0_space: bool = False         # 一致性学生(x0 空间残差场)
    # 阶段 C 预留: n_candidates>1 时边界处 batch 采样多候选, selector 选优
    n_candidates: int = 1
    selector: Union[str, Callable] = "first"


class SkillRuntime:
    """单技能执行封装: 采样/规范化/去规范化(NFE 计数内建)。

    dp: SkillDP 或 ConsistencyStudent(duck typing: .sample/.Q/.n_skills)。
    """

    def __init__(self, dp, norm, device="cuda", n_ddim=24, resample=8):
        self.dp = dp
        self.obs_mean, self.obs_std, self.act_mean, self.act_std = norm
        self.device = device
        self.n_ddim = n_ddim
        self.resample = resample
        self.nfe = 0

    def norm_obs(self, obs) -> torch.Tensor:
        o = np.asarray(obs, dtype=np.float32)
        return torch.from_numpy((o - self.obs_mean) / self.obs_std).float() \
            .to(self.device).unsqueeze(0)

    def onehot(self, sid: int) -> torch.Tensor:
        z = np.zeros((1, self.dp.n_skills), dtype=np.float32)
        z[0, sid] = 1.0
        return torch.from_numpy(z).to(self.device)

    def sample_chunk(self, obs, sid: int, seed=None) -> torch.Tensor:
        a = self.dp.sample(self.norm_obs(obs), self.onehot(sid),
                           n_steps=self.n_ddim, seed=seed)
        self.nfe += self.n_ddim
        return a

    def denorm(self, chunk: torch.Tensor, step: int) -> np.ndarray:
        return np.clip(chunk[0, step].cpu().numpy() * self.act_std
                       + self.act_mean, -1.0, 1.0)


class RiskMonitor:
    """场能量尖峰检测: energy > max(k × 本回合历史中位数, floor) 判尖峰。

    第一个边界(无历史)用 first 阈值(默认 floor)。对应 eval_recovery 的
    baseline = median(energies[:-1]) if len(energies) > 1 else rho。
    """

    def __init__(self, k: float = 3.0, floor: float = 1.0, first: Optional[float] = None):
        self.k, self.floor, self.first = k, floor, first
        self.energies: list = []

    def update(self, energy: float) -> bool:
        base = float(np.median(self.energies)) if self.energies \
            else (self.first if self.first is not None else self.floor)
        self.energies.append(energy)
        return energy > max(self.k * base, self.floor)


class ChainExecutor:
    """统一 rollout 主循环(Meta-World / LIBERO 共用)。

    钩子(均可选):
    - obs_fn(raw_obs) -> policy obs 数组(LIBERO 的 dict 拼接; None 则原样)
    - on_boundary(ctx): 交接锚点采样后、switch 前回调(ctx 可记录 lips 等)
    - on_risk(spike, energy, state) -> Optional[int]: 尖峰重规划(目标技能索引)
    - action_filter(t, a_raw) -> a_raw': 扰动注入
    - stop_fn(state) -> bool: 提前退出
    - skill_done_fn(sid, env, obs, info) -> float: 技能结束时的成功判定
      (criterion 切换策略也用它做提前切换判据)
    """

    def __init__(self, runtime: SkillRuntime, spec: TransitionSpec,
                 obs_fn: Optional[Callable] = None,
                 on_boundary: Optional[Callable] = None,
                 on_risk: Optional[Callable] = None,
                 skill_done_fn: Optional[Callable] = None,
                 switch_policy: str = "fixed",
                 timeout_factor: float = 1.5,
                 min_steps_ratio: float = 0.5,
                 risk_monitor: Optional[RiskMonitor] = None):
        assert switch_policy in ("fixed", "criterion")
        self.rt = runtime
        self.spec = spec
        self.obs_fn = obs_fn or (lambda o: o)
        self.on_boundary = on_boundary
        self.on_risk = on_risk
        self.skill_done_fn = skill_done_fn
        self.switch_policy = switch_policy
        self.timeout_factor = timeout_factor
        self.min_steps_ratio = min_steps_ratio
        self.risk_monitor = risk_monitor

    # ---------------- 内部: 边界处理 ----------------
    def _do_boundary(self, obs, seq, cur, seed_t, ctx):
        """默认边界: anchor 采样(s_from) -> cc.switch -> (可选)风险重规划。

        返回 (chunk, step_in_chunk 重置, cur 可能被重规划改写, replans 增量)。
        """
        rt, spec = self.rt, self.spec
        o_t = rt.norm_obs(obs)
        anchor = rt.sample_chunk(obs, seq[cur])            # s_from 当前计划
        s_from = rt.onehot(seq[cur])
        s_to = rt.onehot(seq[cur + 1])
        ctx.update(anchor=anchor, o_t=o_t, s_from=s_from, s_to=s_to,
                   pair=(seq[cur], seq[cur + 1]), t=seed_t,
                   obs=np.asarray(obs, dtype=np.float32))
        if self.on_boundary is not None:
            self.on_boundary(ctx)
        mask = cc.temporal_mask(anchor.shape[1], 0, spec.mask_width,
                                rt.device) if spec.use_mask else None
        N = max(1, int(spec.n_candidates))
        if N > 1:
            # 阶段 C 候选: obs/one-hot 复制 N -> dp.sample 一次采 N 个独立噪声
            # 样本(dp.sample 的 generator 逐元素独立) -> 各自 switch+proj,
            # selector 选优(callable / "random" 消融臂 / "first")。
            o_b = o_t.expand(N, -1)
            anchor = rt.dp.sample(o_b, s_from.expand(N, -1),
                                  n_steps=rt.n_ddim, seed=seed_t)
            rt.nfe += rt.n_ddim
            ctx["anchor"] = anchor
            a_cands, info_t = cc.switch(rt.dp, o_b, anchor,
                                        s_from.expand(N, -1),
                                        s_to.expand(N, -1),
                                        spec.tau, spec.delta, spec.lam,
                                        spec.n_noise, spec.mode, mask,
                                        spec.use_proj, seed=seed_t,
                                        x0_space=spec.x0_space)
            u = info_t["field"]                     # [N, H, da]
            energies_i = (u ** 2).mean(dim=(1, 2))  # per-candidate 能量
            ctx.update(cands=a_cands, cand_fields=u,
                       cand_energies=energies_i.cpu().numpy())
            if callable(spec.selector):
                idx = int(spec.selector(a_cands, ctx))
            elif spec.selector == "random":         # 消融臂: 多采样不选优
                idx = int(np.random.default_rng(seed_t).integers(N))
            else:                                   # "first"
                idx = 0
            a_new = a_cands[idx:idx + 1]
            info_t["energy"] = float(energies_i[idx])  # 选中候选(语义对齐旧行为)
            ctx["cand_pick"] = idx
        else:
            a_new, info_t = cc.switch(rt.dp, o_t, anchor, s_from, s_to,
                                      spec.tau, spec.delta, spec.lam,
                                      spec.n_noise, spec.mode, mask,
                                      spec.use_proj, seed=seed_t,
                                      x0_space=spec.x0_space)
        rt.nfe += (2 if spec.mode == "chord" else 1) * spec.n_noise
        if spec.mode == "chord_recon":
            rt.nfe += 2 * spec.n_noise
        ctx["last_energy"] = info_t["energy"]
        ctx["energies"].append(info_t["energy"])
        ctx["boundary_obs"].append(np.asarray(obs, dtype=np.float32).copy())
        # 风险信号 -> 事件式重规划(对齐 eval_recovery: 重规划后跳过当步执行)
        replans = 0
        if self.risk_monitor is not None and self.on_risk is not None:
            spike = self.risk_monitor.update(info_t["energy"])
            state = dict(cur=cur, next=seq[cur + 1], seq=seq, t=seed_t,
                         replans=ctx["replans"])
            target = self.on_risk(spike, info_t["energy"], state)
            if target is not None:
                replans = 1
                ctx["replans"] += 1
                cur = target
                a_new = rt.sample_chunk(obs, seq[cur])
            else:
                cur = cur + 1  # 正常推进到后继技能
        else:
            cur = cur + 1
        return a_new, cur, replans

    def _should_switch(self, step_in_skill, skill_steps, cur, obs, env, info):
        """切换触发: fixed(步数) / criterion(完成判据提前 + 超时保护)。"""
        if cur + 1 >= len(skill_steps):
            return False
        if self.switch_policy == "fixed":
            return step_in_skill >= skill_steps[cur]
        # criterion: 达到最小步数后, 技能完成判据触发提前切换
        min_steps = int(skill_steps[cur] * self.min_steps_ratio)
        timeout = int(np.ceil(skill_steps[cur] * self.timeout_factor))
        if step_in_skill >= timeout:
            return True
        if step_in_skill >= min_steps and self.skill_done_fn is not None:
            return bool(self.skill_done_fn(cur, env, obs, info) > 0.5)
        return False

    # ---------------- 主循环 ----------------
    def run(self, env, obs, seq, skill_steps, seed=0, max_steps=None,
            action_filter: Optional[Callable] = None,
            stop_fn: Optional[Callable] = None):
        """执行技能序列。调用方负责 env 的 reset/init/setup(obs 为初始观测)。"""
        rt = self.rt
        total = sum(skill_steps) if max_steps is None \
            else min(sum(skill_steps), max_steps)
        ctx = dict(energies=[], boundary_obs=[], oos=[], lips=[], replans=0,
                   last_obs=np.asarray(obs, dtype=np.float32).copy())
        chunk = rt.sample_chunk(obs, seq[0])
        step_in_chunk = 0
        cur = 0
        step_in_skill = 0
        exec_actions = []
        per_skill = {}
        info = {}
        t = 0
        while t < total:
            raw = None
            if t > 0 and self._should_switch(step_in_skill, skill_steps, cur,
                                             obs, env, info):
                # 技能结束成功判定(fixed: 步数到界时记录)
                if self.skill_done_fn is not None:
                    per_skill[seq[cur]] = self.skill_done_fn(
                        cur, env, obs, info)
                ctx["oos"].append(float(np.abs(
                    np.asarray(obs) - ctx["last_obs"]).max()))
                chunk, cur, n_rep = self._do_boundary(obs, seq, cur,
                                                      seed + t, ctx)
                step_in_chunk = 0
                step_in_skill = 0
                if n_rep > 0:
                    t += 1
                    continue
            a_raw = rt.denorm(chunk, step_in_chunk)
            if action_filter is not None:
                a_raw = action_filter(t, a_raw)
            exec_actions.append(a_raw)
            step_out = env.step(a_raw)
            raw, info = step_out[0], step_out[-1]
            obs = self.obs_fn(raw) if raw is not None else obs
            step_in_chunk += 1
            step_in_skill += 1
            if t < total - 1:
                ctx["last_obs"] = np.asarray(obs, dtype=np.float32).copy()
            if step_in_chunk >= min(rt.resample, chunk.shape[1]):
                chunk = rt.sample_chunk(obs, seq[cur])
                step_in_chunk = 0
            if stop_fn is not None and stop_fn(
                    dict(cur=cur, step_in_skill=step_in_skill,
                         skill_steps=skill_steps, seq=seq)):
                break
            t += 1
        # 终技能判定(循环正常耗尽时)
        if self.skill_done_fn is not None and seq[cur] not in per_skill \
                and step_in_skill >= skill_steps[cur]:
            per_skill[seq[cur]] = self.skill_done_fn(cur, env, obs, info)
        return dict(seq=seq, per_skill=per_skill, cur=cur,
                    obs=obs, info=info,
                    exec_actions=np.array(exec_actions), nfe=rt.nfe,
                    energies=ctx["energies"], replans=ctx["replans"], ctx=ctx)
