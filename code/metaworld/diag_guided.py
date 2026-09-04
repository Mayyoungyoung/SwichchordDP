"""[DEPRECATED 2026-09-04] F_B-guided A-tail 实验。

⚠️ 本实验建立在第七轮 termdiv 初版 bug 数据之上，结果已作废
（experiment_report.md §13.3）。且其 F_B v2 模型基于修复前伪象数据训练。
保留仅供历史记录；新框架的修复实验见 ready_poc.py（READY Phase 1）。

背景: Reachability 显示被动尾部采样 ΔF_B=+0.022(显著但效应小) —— 冻结 DP
尾部自由度不足, 需主动引导。此处实现建议 2 的路线 B(Foresight Guidance 简化版):

A 的最后 H=10 步, 每步执行前对该步动作做 F_B 梯度上升:
    a' = a + η·∇_a F_B(ŝ(a))
ŝ 外推(可微, Meta-World 结构近似): hand' = hand + follow·0.01·a[:3]
(动作即 mocap delta-pos ×0.01m, hand 二阶跟踪用 follow≈0.5; 携物 grip<0.5
时 puck 跟手; grip/goal 不变), 特征 = featurize(hand', grip, puck', goal)。

配对协议(同 ep 同 mid 状态):
- baseline: 原始 DP 尾部 10 步 -> settle -> 终态 F_B + place×10(官方判定)
- guided:   同 mid 起, 10 步每步梯度引导 -> settle -> F_B + place×10
指标: 终态 F_B 增量、P_emp(place 成功率)配对差、Wilcoxon 检验。

输出: results/metaworld/eval/guided_carry_place.json
"""
import argparse
import json
import os
import time

import mujoco
import numpy as np
import torch
from scipy import stats

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from swdp.policy import SkillDP  # noqa: E402
from swdp.success_model import (featurize,  # noqa: E402
                                load as fb_load)
from skills import make_env, SKILLS, parse_pp  # noqa: E402
from diag_handoff import load_norm, diag_success  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCENE = "pick-place-v3"
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "../../results/metaworld/models")
EVAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "../../results/metaworld/eval")
SKILL_NAMES = ["reach", "grasp", "lift", "carry", "place"]
SETUP = ["reach", "grasp", "lift"]
SETUP_STEPS = {"reach": 30, "grasp": 30, "lift": 25}
A_TOTAL, A_TAIL, N_SETTLE, K_PLACE = 30, 10, 20, 10
ACTION_SCALE, FOLLOW = 0.01, 0.5
SID_CARRY, SID_PLACE = 3, 4


def save_state(env):
    d = env._env.data
    return np.concatenate([d.qpos.copy(), d.qvel.copy(),
                           d.mocap_pos.ravel().copy(),
                           d.mocap_quat.ravel().copy()])


def restore_state(env, svec):
    d = env._env.data
    m = env._env.model
    nq, nv = d.qpos.shape[0], d.qvel.shape[0]
    d.qpos[:] = svec[:nq]
    d.qvel[:] = svec[nq:nq + nv]
    d.mocap_pos[:] = svec[nq + nv:nq + nv + 3]
    d.mocap_quat[:] = svec[nq + nv + 3:nq + nv + 7]
    mujoco.mj_forward(m, d)
    # 恢复状态语义 = 新回合起点, 重置 env 计数器(避免累计超 max_path_length)
    env._env.curr_path_length = 0


def settle(env, n_steps=N_SETTLE):
    for _ in range(n_steps):
        env.step(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
    return env._env._get_obs()


@torch.no_grad()
def run_carry(dp, norm, env, obs, seed, n_steps, guide=None,
              chunk=None, step_in=0):
    """DP carry n_steps; guide(obs, a_raw) -> 可选引导修改 a_raw。

    chunk/step_in: 若非 None 则从该 chunk 的 step_in 继续(连续执行等价),
    返回 (obs, chunk, step_in) 供后续继续。"""
    obs_mean, obs_std, act_mean, act_std = norm

    def norm_obs(o):
        return torch.from_numpy((o - obs_mean) / obs_std).float().to(
            DEVICE).unsqueeze(0)

    def onehot(i):
        z = np.zeros((1, dp.n_skills), dtype=np.float32)
        z[0, i] = 1.0
        return torch.from_numpy(z).to(DEVICE)

    if chunk is None:
        chunk = dp.sample(norm_obs(obs), onehot(SID_CARRY), n_steps=24,
                          seed=seed)
    for t in range(n_steps):
        a_raw = np.clip(chunk[0, step_in].cpu().numpy() * act_std + act_mean,
                        -1.0, 1.0)
        if guide is not None:
            a_raw = guide(obs, a_raw)
        obs, *_ = env.step(a_raw)
        step_in += 1
        if step_in >= 8:
            chunk = dp.sample(norm_obs(obs), onehot(SID_CARRY), n_steps=24,
                              seed=seed + t)
            step_in = 0
    return obs, chunk, step_in


def make_guide(fb, eta=0.05, n_iter=3, lam_keep=0.1):
    """F_B 梯度引导器: 每步对动作做梯度上升 F_B(ŝ(a))。

    ŝ: hand' = hand + FOLLOW*ACTION_SCALE*a[:3]; puck 跟手(grip<0.5);
    grip/goal 不变。仅位置通道有梯度, 夹爪通道向 lam_keep 正则(保持闭合)。
    """
    sid_onehot = np.zeros(len(SKILL_NAMES), dtype=np.float32)
    sid_onehot[SID_PLACE] = 1.0

    def guide(obs, a_raw):
        o = parse_pp(obs)
        hand = np.asarray(o["hand"], dtype=np.float32)
        puck = np.asarray(o["puck"], dtype=np.float32)
        goal = np.asarray(o["goal"], dtype=np.float32)
        grip = float(o["grip"])
        holding = grip < 0.5
        with torch.enable_grad():   # run_carry 在 no_grad 下, 引导段需显式开梯度
            a = torch.from_numpy(a_raw.astype(np.float32)).to(DEVICE).clone()
            a.requires_grad_(True)
            for _ in range(n_iter):
                disp = a[:3] * (ACTION_SCALE * FOLLOW)
                hand_p = torch.from_numpy(hand).to(DEVICE) + disp
                puck_p = (torch.from_numpy(puck).to(DEVICE) + disp) if holding \
                    else torch.from_numpy(puck).to(DEVICE)
                f = torch.cat([hand_p, torch.tensor([grip], device=DEVICE),
                               puck_p, torch.from_numpy(goal).to(DEVICE),
                               hand_p - puck_p,
                               hand_p - torch.from_numpy(goal).to(DEVICE),
                               puck_p - torch.from_numpy(goal).to(DEVICE)])
                s = torch.from_numpy(sid_onehot).to(DEVICE).unsqueeze(0)
                prob = fb.forward(f.unsqueeze(0), s)
                # 最大化 F_B; 夹爪通道正则向 +1(实测 +1=闭合, 保持携物)
                loss = -prob.squeeze() + lam_keep * (a[3] - 1.0) ** 2
                loss.backward()
                with torch.no_grad():
                    a = a + eta * a.grad
                    a = torch.clamp(a, -1.0, 1.0)
                a = a.detach().clone().requires_grad_(True)
        return a.detach().cpu().numpy()

    return guide


@torch.no_grad()
def run_place(dp, norm, env, obs, seed, n_steps=20):
    """冻结 DP place, 返回 (obs, success 官方判定)。"""
    obs_mean, obs_std, act_mean, act_std = norm

    def norm_obs(o):
        return torch.from_numpy((o - obs_mean) / obs_std).float().to(
            DEVICE).unsqueeze(0)

    def onehot(i):
        z = np.zeros((1, dp.n_skills), dtype=np.float32)
        z[0, i] = 1.0
        return torch.from_numpy(z).to(DEVICE)

    chunk = dp.sample(norm_obs(obs), onehot(SID_PLACE), n_steps=24, seed=seed)
    step_in = 0
    info = {}
    for t in range(n_steps):
        a_raw = np.clip(chunk[0, step_in].cpu().numpy() * act_std + act_mean,
                        -1.0, 1.0)
        obs, rew, term, trunc, info = env.step(a_raw)
        step_in += 1
        if step_in >= 8:
            chunk = dp.sample(norm_obs(obs), onehot(SID_PLACE), n_steps=24,
                              seed=seed + t)
            step_in = 0
    return float(diag_success(SCENE, "place", env, obs, info))


def fb_score(fb, o):
    onehot = np.zeros(len(SKILL_NAMES), dtype=np.float32)
    onehot[SID_PLACE] = 1.0
    f = featurize(o["hand"], float(o["grip"]), o["puck"], o["goal"])
    return float(fb.predict(f, onehot))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_episodes", type=int, default=40)
    ap.add_argument("--eta", type=float, default=0.05,
                    help="每步梯度上升步长")
    ap.add_argument("--n_iter", type=int, default=3)
    ap.add_argument("--out", default="guided_carry_place")
    args = ap.parse_args()

    dp = SkillDP.load(os.path.join(MODEL_DIR, f"dp_{SCENE}.pt"), DEVICE)
    dp.eval()
    fb, _ = fb_load(os.path.join(MODEL_DIR, f"fb_{SCENE}_v2.pt"), DEVICE)
    norm = load_norm()
    guide = make_guide(fb, eta=args.eta, n_iter=args.n_iter)

    rows = []
    t0 = time.time()
    for ep in range(args.n_episodes):
        env = make_env(SCENE, seed=7000 + ep)
        obs, _ = env.reset()
        for p in SETUP:
            ctrl = SKILLS[SCENE][p](env)
            for _ in range(SETUP_STEPS[p]):
                obs, *_ = env.step(ctrl.act(obs))
        base_seed = ep * 7 + 1
        obs, chunk, step_in = run_carry(dp, norm, env, obs, base_seed,
                                        A_TOTAL - A_TAIL)
        mid_state = save_state(env)
        # 臂 1: baseline 继续原 chunk(连续执行等价, 同 termdiv 协议) + settle
        obs_b, _, _ = run_carry(dp, norm, env, obs, base_seed, A_TAIL,
                                chunk=chunk, step_in=step_in)
        obs_b = settle(env)
        o_b = parse_pp(obs_b)
        fb_b = fb_score(fb, o_b)
        state_b = save_state(env)
        # 臂 2: guided 同 mid 起, 继续原 chunk 但每步 F_B 梯度引导 + settle
        restore_state(env, mid_state)
        obs_g, _, _ = run_carry(dp, norm, env, env._env._get_obs(), base_seed,
                                A_TAIL, guide=guide, chunk=chunk.clone(),
                                step_in=step_in)
        obs_g = settle(env)
        o_g = parse_pp(obs_g)
        fb_g = fb_score(fb, o_g)
        state_g = save_state(env)
        # 真实 place rollout × K(配对, 每次从各自终态恢复)
        acc_b, acc_g = [], []
        for kk in range(K_PLACE):
            restore_state(env, state_b)
            acc_b.append(run_place(dp, norm, env, env._env._get_obs(),
                                   9000 + kk))
        for kk in range(K_PLACE):
            restore_state(env, state_g)
            acc_g.append(run_place(dp, norm, env, env._env._get_obs(),
                                   9100 + kk))
        rows.append(dict(ep=ep, fb_base=fb_b, fb_guided=fb_g,
                         p_base=float(np.mean(acc_b)),
                         p_guided=float(np.mean(acc_g))))
        env.close()
        if (ep + 1) % 10 == 0:
            el = time.time() - t0
            print(f"[guided] ep{ep + 1}: F_B {fb_b:.3f}->{fb_g:.3f}  "
                  f"P_place {rows[-1]['p_base']:.2f}->"
                  f"{rows[-1]['p_guided']:.2f} ({el:.0f}s)")
    fb_b = np.array([r["fb_base"] for r in rows])
    fb_g = np.array([r["fb_guided"] for r in rows])
    p_b = np.array([r["p_base"] for r in rows])
    p_g = np.array([r["p_guided"] for r in rows])
    rep = dict(
        n=len(rows),
        fb=dict(base=float(fb_b.mean()), guided=float(fb_g.mean()),
                delta=float((fb_g - fb_b).mean()),
                wilcoxon_p=float(stats.wilcoxon(fb_g, fb_b).pvalue)),
        place=dict(base=float(p_b.mean()), guided=float(p_g.mean()),
                   delta=float((p_g - p_b).mean()),
                   wilcoxon_p=float(stats.wilcoxon(p_g, p_b).pvalue),
                   n_win=int(np.sum(p_g > p_b)),
                   n_lose=int(np.sum(p_g < p_b)),
                   n_tie=int(np.sum(p_g == p_b))),
    )
    print(json.dumps(rep, indent=2))
    os.makedirs(EVAL_DIR, exist_ok=True)
    path = os.path.join(EVAL_DIR, args.out + ".json")
    with open(path, "w") as f:
        json.dump(dict(args=vars(args), report=rep, rows=rows), f, indent=2)
    print(f"[guided] saved {path}")


if __name__ == "__main__":
    main()
