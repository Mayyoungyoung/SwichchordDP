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


@torch.no_grad()
def rollout(dp, scene, seq, skill_steps, mode, setup=None, tau=0.9, delta=0.15,
            lam=1.0, n_noise=1, use_mask=True, use_proj=False, mask_width=4,
            n_ddim=24, resample=8, seed=0):
    """执行一个技能序列, 返回指标字典。

    setup: 脚本前置技能序列(建立首个技能的前置状态, 不计分)。
    """
    env = make_env(scene, seed=1000 + seed)
    obs, info = env.reset()
    if setup:
        for p in setup:
            ctrl = SKILLS[scene][p](env)
            for _ in range(SETUP_STEPS[p]):
                obs, *_ = env.step(ctrl.act(obs))
    with h5py.File(os.path.join(DATA_DIR, f"{scene}.h5"), "r") as f:
        obs_mean = f["obs_mean"][:]; obs_std = f["obs_std"][:]
        act_mean = f["act_mean"][:]; act_std = f["act_std"][:]
    skill_ids = {n: i for i, n in enumerate(SKILL_NAMES[scene])}

    def norm_obs(o):
        return torch.from_numpy((o - obs_mean) / obs_std).float().to(DEVICE).unsqueeze(0)

    def onehot(i):
        z = np.zeros((1, dp.n_skills), dtype=np.float32)
        z[0, i] = 1.0
        return torch.from_numpy(z).to(DEVICE)

    def sample_chunk(o, s_id):
        a = dp.sample(o, onehot(s_id), n_steps=n_ddim, seed=None)  # [1,H,D] 归一化
        return a

    nfe = 0
    total_steps = sum(skill_steps)
    cur_skill_idx = 0
    chunk = sample_chunk(norm_obs(obs), skill_ids[seq[0]])
    nfe += n_ddim
    step_in_chunk = 0
    step_in_skill = 0
    energies = []
    jerks = []
    exec_actions = []
    per_skill_succ = {}
    oos_shifts = []
    lips_vals = []
    last_obs = obs

    for t in range(total_steps):
        # 技能边界检查
        if t > 0 and step_in_skill >= skill_steps[cur_skill_idx]:
            cur_skill_idx += 1
            step_in_skill = 0
            s_from = onehot(skill_ids[seq[cur_skill_idx - 1]])
            s_to = onehot(skill_ids[seq[cur_skill_idx]])
            o_t = norm_obs(obs)
            # OOS 代理: 边界处观测变化幅度
            oos_shifts.append(float(np.abs(obs - last_obs).max()))
            # 交接锚点 = 以当前观测重新采样的 s_from 动作块(当前计划),
            # 传输后从块首重新执行(对应 ChordEdit「编辑当前计划」语义)。
            anchor = sample_chunk(o_t, skill_ids[seq[cur_skill_idx - 1]])
            nfe += n_ddim
            # 理论闭环: 边界锚点 Lipschitz
            lips_vals.append(dict(
                pair=f"{seq[cur_skill_idx-1]}->{seq[cur_skill_idx]}",
                L_from=est_lipschitz(dp, anchor, o_t, s_from),
                L_to=est_lipschitz(dp, anchor, o_t, s_to)))
            # 传输
            mask = cc.temporal_mask(anchor.shape[1], 0, mask_width, DEVICE) if use_mask else None
            a_new, info_t = cc.switch(dp, o_t, anchor, s_from, s_to, tau, delta,
                                      lam, n_noise, mode, mask, use_proj,
                                      seed=seed + t, x0_space=X0_SPACE)
            nfe += (2 if mode in ("chord",) else 1) * n_noise
            if mode == "chord_recon":
                nfe += 2 * n_noise  # 两个时刻各一次条件差查询
            # 记录传输场能量
            energies.append(info_t["energy"])
            chunk = a_new
            step_in_chunk = 0

        # 执行当前块的首个动作
        a_raw = (chunk[0, step_in_chunk].cpu().numpy() * act_std) + act_mean
        a_raw = np.clip(a_raw, -1.0, 1.0)
        exec_actions.append(a_raw)
        obs, rew, term, trunc, info = env.step(a_raw)
        nfe += 0
        step_in_chunk += 1
        step_in_skill += 1
        if t < total_steps - 1:
            last_obs = obs

        # 块耗尽 -> 重采样
        if step_in_chunk >= chunk.shape[1]:
            chunk = sample_chunk(norm_obs(obs), skill_ids[seq[cur_skill_idx]])
            nfe += n_ddim
            step_in_chunk = 0
        # 步长推进: 每 resample 步重新采样(窗口前移)
        if step_in_chunk >= resample:
            chunk = sample_chunk(norm_obs(obs), skill_ids[seq[cur_skill_idx]])
            nfe += n_ddim
            step_in_chunk = 0

        # 技能结束时判定成功
        if step_in_skill >= skill_steps[cur_skill_idx]:
            per_skill_succ[seq[cur_skill_idx]] = skill_success(
                scene, seq[cur_skill_idx], env, obs, info)

    exec_actions = np.array(exec_actions)
    _e = np.sum(np.diff(exec_actions, axis=0) ** 2, axis=1)
    energy = float(np.median(_e)) if len(_e) else 0.0
    jerk = float(np.max(np.abs(np.diff(exec_actions, n=2, axis=0)))) if len(exec_actions) > 2 else 0.0
    e2e = float(all(per_skill_succ.get(s, 0.0) > 0.5 for s in seq))
    env.close()
    return dict(
        seq=seq, per_skill=per_skill_succ, e2e=e2e, energy=energy, jerk=jerk,
        nfe=nfe, oos=max(oos_shifts) if oos_shifts else 0.0,
        lips=lips_vals, boundary_energy=energies)


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
                        seed=ep * 7 + 1)
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
    out_path = os.path.join(DATA_DIR, "../eval", f"{tag}.json")
    with open(out_path, "w") as f:
        json.dump(dict(args=vars(args), results=results), f, indent=2,
                  default=str)
    print(f"[eval] saved {out_path}")


if __name__ == "__main__":
    main()
