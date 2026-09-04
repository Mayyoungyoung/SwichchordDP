"""ChordCompose 组合任务评测: baseline 对比 + 消融 + 理论-实验一致性。

执行协议(receding horizon):
1. 每个技能执行 n_skill 步; 每 R=8 步用当前技能重新采样动作块(DDIM n_steps 步)。
2. 交接边界处对当前动作块施加传输(mode: chord/naive/eff_shift/energy), 可选时间掩码与可行性投影。
3. 记录: 分技能成功率、端到端成功率、轨迹能量、边界 jerk、NFE、OOS 代理指标、
   以及边界锚点处 ‖∂eps_hat/∂a‖ 的有限差分 Lipschitz 估计(理论闭环)。
"""
import argparse
import json
import os
import time

import h5py
import numpy as np
import torch

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from swdp.policy import SkillDP  # noqa: E402
from swdp.distil import ConsistencyStudent  # noqa: E402
from swdp import chord_compose as cc  # noqa: E402
from swdp.feasibility import prox_feasible  # noqa: E402
from swdp.harness import (SkillRuntime, TransitionSpec,  # noqa: E402
                          ChainExecutor)
from skills import make_env, SKILLS  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
X0_SPACE = False

# ---------------- 场景任务定义 ----------------
# 每项: (技能序列, 各技能执行步数, seen/unseen)
# 每项: (技能序列, 脚本前置序列(建立首个技能的前置状态), 各技能步数, seen/unseen)
# unseen = 演示数据中从未出现的技能序对/起始技能。
TASKS = {
    "pick-place-v3": [
        (["reach", "grasp"], [], [30, 30], "seen"),
        (["reach", "grasp", "lift", "carry", "place"], [], [30, 25, 25, 30, 20], "seen"),
        (["reach", "carry"], [], [30, 30], "unseen"),
        (["grasp", "carry"], ["reach"], [30, 30], "unseen"),
        (["lift", "place"], ["reach", "grasp"], [25, 25], "unseen"),
        (["place"], ["reach", "grasp", "lift", "carry"], [25], "unseen"),
    ],
    "door-open-v3": [
        (["reach", "open"], [], [40, 75], "seen"),
        (["open"], [], [100], "unseen"),
    ],
}
SETUP_STEPS = {"reach": 30, "grasp": 30, "lift": 25, "carry": 30, "open": 40}


def skill_success(scene, name, env, obs, info):
    """技能达成判定(基于状态几何, 与 skills.py 的控制器 done 一致)。"""
    if scene == "pick-place-v3":
        hand, grip = obs[:3], obs[3]
        puck, goal = obs[4:7], obs[-3:]
        if name == "reach":
            return float(np.linalg.norm(hand[:2] - puck[:2]) < 0.03)
        if name == "grasp":
            return float(grip < 0.75)
        if name == "lift":
            return float(puck[2] > 0.08)
        if name == "carry":
            return float(np.linalg.norm(hand[:2] - goal[:2]) < 0.05 and
                         grip < 0.6)
        if name == "place":
            # 官方判定: obj_to_target(3D) <= 0.07; info 不可用时退化为几何判定
            if isinstance(info, dict) and info.get("success", 0.0) > 0.5:
                return 1.0
            return float(np.linalg.norm(puck - goal) < 0.07 and grip > 0.7)
    elif scene == "door-open-v3":
        if name == "reach":
            target = obs[4:7] + np.array([0.06, 0.02, 0.2])
            return float(np.linalg.norm(obs[:3] - target) < 0.08)
        if name == "open":
            if isinstance(info, dict) and info.get("success", 0.0) > 0.5:
                return 1.0
            return float(abs(obs[4] - getattr(env, "_target_pos", [1.0, 0, 0])[0]) <= 0.08)
    return 0.0


def est_lipschitz(dp, a, obs, s, n_pert=16, eps=1e-2):
    """有限差分估计 ‖∂模型输出/∂a‖(谱范数近似, 用最大方向差分比)。"""
    B, H, D = a.shape
    max_ratio = 0.0
    q = (lambda x: dp.f(x, torch.full((B, 1), 0.9, device=DEVICE), obs, s)) \
        if X0_SPACE else \
        (lambda x: dp.Q(x, torch.full((B, 1), 0.9, device=DEVICE), obs, s))
    for _ in range(n_pert):
        delta = torch.randn_like(a)
        delta = delta / (delta.norm(dim=(1, 2), keepdim=True) + 1e-8)
        a2 = a + eps * delta
        e1 = q(a)
        e2 = q(a2)
        ratio = ((e2 - e1).norm(dim=(1, 2)) / (eps * delta.norm(dim=(1, 2)) + 1e-12))
        max_ratio = max(max_ratio, float(ratio.max()))
    return max_ratio


def make_fb_selector(fb, rt, scene, lam_r=0.1, follow=0.5,
                     action_scale=0.01):
    """F_B 候选选择器: score_i = F_B(ŝ_i) - λ_r·Ê_i(归一化能量惩罚)。

    ŝ 外推(Meta-World 结构近似, 计划 C2 第一版):
    - 动作即 mocap 目标 delta-pos(归一化 [-1,1] × action_scale=0.01m);
      hand 二阶跟踪 -> follow 系数近似短期跟踪率(实测 8 步 ~0.5);
    - 外推步数 = resample(候选块 receding-horizon 下实际只执行前 R 步);
    - 物体: grip 闭合(<0.5)时随手平移, 否则不动(工作空间假设);
    - grip/goal 保持当前值(候选短期对 grip 影响小)。
    若诊断显示物体动力学主导, 升级为 F_θ(s, a_chunk, B) 网络外推。
    """
    from swdp.success_model import featurize

    def selector(a_cands, ctx):
        obs = ctx["obs"]
        if scene == "pick-place-v3":
            hand, grip, puck, goal = obs[:3], obs[3], obs[4:7], obs[-3:]
        else:
            return 0
        R = min(rt.resample, a_cands.shape[1])
        a_den = a_cands[:, :R, :3].cpu().numpy() \
            * rt.act_std[:3] + rt.act_mean[:3]
        disp = a_den.sum(axis=1) * action_scale * follow    # [N, 3]
        hands = hand[None, :] + disp
        if grip < 0.5:            # 携物: 物体跟手
            pucks = puck[None, :] + disp
        else:
            pucks = np.broadcast_to(puck[None, :], disp.shape)
        onehot = ctx["s_to"][0].cpu().numpy()               # 后继技能 B
        probs = np.array([
            float(fb.predict(featurize(h, grip, p, goal), onehot))
            for h, p in zip(hands, pucks)])
        E = np.asarray(ctx["cand_energies"], dtype=np.float64)
        E_n = (E - E.min()) / (np.ptp(E) + 1e-9)            # [0,1] 归一化
        scores = probs - lam_r * E_n
        ctx["fb_probs"] = probs.tolist()
        return int(np.argmax(scores))

    return selector


@torch.no_grad()
def rollout(dp, scene, seq, skill_steps, mode, setup=None, tau=0.9, delta=0.15,
            lam=1.0, n_noise=1, use_mask=True, use_proj=False, mask_width=4,
            n_ddim=24, resample=8, seed=0, n_candidates=1, selector="first",
            fb=None, lam_r=0.1, switch_policy="fixed", timeout_factor=1.5,
            min_steps_ratio=0.5):
    """执行一个技能序列, 返回指标字典(统一走 ChainExecutor 执行层)。

    setup: 脚本前置技能序列(建立首个技能的前置状态, 不计分)。
    n_candidates>1: 边界处批量采样候选(selector: first/random/fb 回调)。
    switch_policy: fixed(主协议) / criterion(C4 切换触发消融臂)。
    """
    env = make_env(scene, seed=1000 + seed)
    obs, info = env.reset()
    if setup:
        for p in setup:
            ctrl = SKILLS[scene][p](env)
            for _ in range(SETUP_STEPS[p]):
                obs, *_ = env.step(ctrl.act(obs))
    with h5py.File(os.path.join(DATA_DIR, f"{scene}.h5"), "r") as f:
        norm = (f["obs_mean"][:], f["obs_std"][:],
                f["act_mean"][:], f["act_std"][:])
    skill_ids = {n: i for i, n in enumerate(SKILL_NAMES[scene])}
    name_of = {i: n for n, i in skill_ids.items()}
    sid_seq = [skill_ids[n] for n in seq]

    rt = SkillRuntime(dp, norm, device=DEVICE, n_ddim=n_ddim,
                      resample=resample)
    sel = selector
    if selector == "fb":
        assert fb is not None, "--selector fb 需要 --fb_path"
        sel = make_fb_selector(fb, rt, scene, lam_r=lam_r)
    spec = TransitionSpec(mode=mode, tau=tau, delta=delta, lam=lam,
                          n_noise=n_noise, use_mask=use_mask,
                          mask_width=mask_width, use_proj=use_proj,
                          x0_space=X0_SPACE, n_candidates=n_candidates,
                          selector=sel)

    def done_fn(si, env_, obs_, info_):
        return skill_success(scene, seq[si], env_, obs_, info_)

    def on_boundary(ctx):
        # 理论闭环: 边界锚点 Lipschitz(与旧实现同位同参)
        a, b = ctx["pair"]
        ctx["lips"].append(dict(
            pair=f"{name_of[a]}->{name_of[b]}",
            L_from=est_lipschitz(dp, ctx["anchor"], ctx["o_t"],
                                 ctx["s_from"]),
            L_to=est_lipschitz(dp, ctx["anchor"], ctx["o_t"],
                               ctx["s_to"])))

    ex = ChainExecutor(rt, spec, skill_done_fn=done_fn,
                       on_boundary=on_boundary, switch_policy=switch_policy,
                       timeout_factor=timeout_factor,
                       min_steps_ratio=min_steps_ratio)
    out = ex.run(env, obs, sid_seq, list(skill_steps), seed=seed)
    env.close()

    exec_actions = out["exec_actions"]
    _e = np.sum(np.diff(exec_actions, axis=0) ** 2, axis=1)
    energy = float(np.median(_e)) if len(_e) else 0.0
    jerk = float(np.max(np.abs(np.diff(exec_actions, n=2, axis=0)))) \
        if len(exec_actions) > 2 else 0.0
    per_skill = {name_of[k]: v for k, v in out["per_skill"].items()}
    e2e = float(all(per_skill.get(s, 0.0) > 0.5 for s in seq))
    oos_list = out["ctx"]["oos"]
    return dict(
        seq=seq, per_skill=per_skill, e2e=e2e, energy=energy, jerk=jerk,
        nfe=out["nfe"], oos=max(oos_list) if oos_list else 0.0,
        lips=out["ctx"]["lips"], boundary_energy=out["energies"])


SKILL_NAMES = {
    "pick-place-v3": ["reach", "grasp", "lift", "carry", "place"],
    "door-open-v3": ["reach", "open"],
}
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../results/metaworld/data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../results/metaworld/models")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="pick-place-v3")
    ap.add_argument("--mode", default="chord",
                    choices=["chord", "naive", "eff_shift", "energy",
                             "chord_recon"])
    ap.add_argument("--n_episodes", type=int, default=10)
    ap.add_argument("--tau", type=float, default=0.9)
    ap.add_argument("--delta", type=float, default=0.15)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--n_noise", type=int, default=1)
    ap.add_argument("--use_mask", action="store_true", default=False)
    ap.add_argument("--use_proj", action="store_true", default=False)
    ap.add_argument("--n_ddim", type=int, default=24)
    ap.add_argument("--cd", action="store_true",
                    help="使用一致性蒸馏学生模型(x0 空间残差场, B_t≡I)")
    ap.add_argument("--n_candidates", type=int, default=1,
                    help="边界处候选数(>1 启用 Candidate Reranking)")
    ap.add_argument("--selector", default="first",
                    choices=["first", "random", "fb"])
    ap.add_argument("--fb_path", default="",
                    help="F_B 模型路径(--selector fb 时必填)")
    ap.add_argument("--lam_r", type=float, default=0.1,
                    help="候选能量惩罚系数 λ_r")
    ap.add_argument("--switch_policy", default="fixed",
                    choices=["fixed", "criterion"],
                    help="切换触发: fixed(主协议) / criterion(C4 消融)")
    ap.add_argument("--timeout_factor", type=float, default=1.5)
    ap.add_argument("--min_steps_ratio", type=float, default=0.5)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    global X0_SPACE
    if args.cd:
        ckpt = torch.load(os.path.join(MODEL_DIR, f"dp_{args.scene}_cd.pt"),
                          map_location=DEVICE)
        dp = ConsistencyStudent(**ckpt["cfg"], device=DEVICE)
        dp.load_state_dict(ckpt["model"])
        X0_SPACE = True
    else:
        dp = SkillDP.load(os.path.join(MODEL_DIR, f"dp_{args.scene}.pt"), DEVICE)
        X0_SPACE = False
    dp.eval()
    fb = None
    if args.selector == "fb":
        # 注意: fb_{scene}.pt 是 v1 诊断扰动数据模型，已知在自然终态分布上
        # 无预测力(Spearman n.s., §13.2)——仅用于复现 §12 六臂消融，
        # 勿用于新实验(新实验用 ready_poc.py 的 ensemble)。
        from swdp.success_model import load as fb_load
        fb, _ = fb_load(args.fb_path or os.path.join(
            MODEL_DIR, f"fb_{args.scene}.pt"), DEVICE)
        fb.eval()
    tasks = TASKS[args.scene]
    results = []
    for seq, setup, skill_steps, kind in tasks:
        if args.mode == "energy" and len(seq) < 2:
            continue
        succ = []
        for ep in range(args.n_episodes):
            t0 = time.time()
            r = rollout(dp, args.scene, seq, skill_steps, args.mode,
                        setup=setup, tau=args.tau, delta=args.delta, lam=args.lam,
                        n_noise=args.n_noise, use_mask=args.use_mask,
                        use_proj=args.use_proj, n_ddim=args.n_ddim,
                        seed=ep * 7 + 1, n_candidates=args.n_candidates,
                        selector=args.selector, fb=fb, lam_r=args.lam_r,
                        switch_policy=args.switch_policy,
                        timeout_factor=args.timeout_factor,
                        min_steps_ratio=args.min_steps_ratio)
            r["latency_s"] = time.time() - t0
            r["kind"] = kind
            succ.append(r)
            print(f"[eval] {args.mode} {seq} ep{ep} e2e={r['e2e']} "
                  f"energy={r['energy']:.3f} nfe={r['nfe']}")
        e2e_rate = float(np.mean([s["e2e"] for s in succ]))
        per_skill_rate = {}
        for s in seq:
            per_skill_rate[s] = float(np.mean(
                [ep["per_skill"].get(s, 0.0) for ep in succ]))
        results.append(dict(seq=seq, kind=kind, e2e=e2e_rate,
                            per_skill=per_skill_rate, episodes=succ))
    os.makedirs(os.path.join(DATA_DIR, "../eval"), exist_ok=True)
    tag = args.out or f"{args.scene}_{args.mode}_t{args.tau}_d{args.delta}_l{args.lam}_n{args.n_noise}"
    if args.use_mask:
        tag += "_mask"
    if args.use_proj:
        tag += "_proj"
    if args.n_candidates > 1:
        tag += f"_cand{args.n_candidates}{args.selector}"
    if args.switch_policy != "fixed":
        tag += f"_{args.switch_policy}"
    out_path = os.path.join(DATA_DIR, "../eval", f"{tag}.json")
    with open(out_path, "w") as f:
        json.dump(dict(args=vars(args), results=results), f, indent=2,
                  default=str)
    print(f"[eval] saved {out_path}")


if __name__ == "__main__":
    main()
