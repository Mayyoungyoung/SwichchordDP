"""Boundary Monitor + Best-of-K 闭环验证（多对, 免训可插拔组件评估）。

组件: 在技能边界入口处检查上游终态质量(几何特征), 不合格则 restore
到检查技能执行前状态并换采样种子重执行(最多 K 次), 取最优候选后继续
执行下游。不碰任何技能权重。

对配置(检查点 = prefix 最后一技能的终态):
  r2g (reach→grasp→lift): 检查 reach 终态 hp ≤ 0.010; 下游评估 = settle+lift
      真实 outcome(与 §15 协议一致) + grasp 语义
  l2c (lift→carry):       检查 lift 终态 hp ≤ 0.030; 下游 = carry 语义
  c2p (carry→place):      检查 carry 终态 pg ≤ 0.070; 下游 = place 语义

双臂同种子配对(off: 自然执行; on: 入口监测 + best-of-K)。输出 JSON。
"""
import argparse
import json
import math
import os
import time

import mujoco
import numpy as np
import torch

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from swdp.policy import SkillDP  # noqa: E402
from skills import make_env, parse_pp  # noqa: E402
from diag_handoff import load_norm, diag_success  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCENE = "pick-place-v3"
EVAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "../../results/metaworld/eval")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "../../results/metaworld/models")
SID = {"reach": 0, "grasp": 1, "lift": 2, "carry": 3, "place": 4}
STEPS = 30
SETTLE = 20
K_LIFT = 10


def save_state(env):
    d = env._env.data
    return np.concatenate([d.qpos.copy(), d.qvel.copy(),
                           d.mocap_pos.ravel().copy(),
                           d.mocap_quat.ravel().copy()])


def restore_state(env, s0):
    d = env._env.data
    m = env._env.model
    nq, nv = d.qpos.shape[0], d.qvel.shape[0]
    d.qpos[:] = s0[:nq]
    d.qvel[:] = s0[nq:nq + nv]
    d.mocap_pos[:] = s0[nq + nv:nq + nv + 3]
    d.mocap_quat[:] = s0[nq + nv + 3:nq + nv + 7]
    mujoco.mj_forward(m, d)
    env._env.curr_path_length = 0
    return env._env._get_obs()


def geom(obs):
    o = parse_pp(obs)
    hand = np.asarray(o["hand"])
    puck = np.asarray(o["puck"])
    goal = np.asarray(o["goal"])
    return dict(hp=float(np.linalg.norm(hand[:2] - puck[:2])),
                pg=float(np.linalg.norm(puck[:2] - goal[:2])),
                puck_z=float(puck[2]),
                grip=float(o["grip"]))


@torch.no_grad()
def dp_rollout(dp, norm, env, obs, sid, n_steps, seed):
    obs_mean, obs_std, act_mean, act_std = norm

    def norm_obs(o):
        return torch.from_numpy((o - obs_mean) / obs_std).float().to(
            DEVICE).unsqueeze(0)

    def onehot(i):
        z = np.zeros((1, dp.n_skills), dtype=np.float32)
        z[0, i] = 1.0
        return torch.from_numpy(z).to(DEVICE)

    chunk = dp.sample(norm_obs(obs), onehot(sid), n_steps=24, seed=seed)
    step_in = 0
    for t in range(n_steps):
        a_raw = np.clip(chunk[0, step_in].cpu().numpy() * act_std
                        + act_mean, -1.0, 1.0)
        obs, *_ = env.step(a_raw)
        step_in += 1
        if step_in >= 8:
            chunk = dp.sample(norm_obs(obs), onehot(sid), n_steps=24,
                              seed=seed + t)
            step_in = 0
    return obs


@torch.no_grad()
def eval_lift_after_settle(dp, norm, env, s0, k=K_LIFT, seed0=4000):
    obs_mean, obs_std, act_mean, act_std = norm

    def norm_obs(o):
        return torch.from_numpy((o - obs_mean) / obs_std).float().to(
            DEVICE).unsqueeze(0)

    def onehot(i):
        z = np.zeros((1, dp.n_skills), dtype=np.float32)
        z[0, i] = 1.0
        return torch.from_numpy(z).to(DEVICE)

    succs = []
    for kk in range(k):
        obs = restore_state(env, s0)
        chunk = dp.sample(norm_obs(obs), onehot(SID["lift"]), n_steps=24,
                          seed=seed0 + kk)
        step_in = 0
        info = {}
        for t in range(25):
            a = np.clip(chunk[0, step_in].cpu().numpy() * act_std
                        + act_mean, -1.0, 1.0)
            obs, rew, term, trunc, info = env.step(a)
            step_in += 1
            if step_in >= 8:
                chunk = dp.sample(norm_obs(obs), onehot(SID["lift"]),
                                  n_steps=24, seed=seed0 + kk + t)
                step_in = 0
        succs.append(float(diag_success(SCENE, "lift", env, obs, info)))
    return float(np.mean(succs))


# 各对配置
PAIRS = {
    "r2g": dict(prefix=["reach"], check="reach", feat="hp", thresh=0.010,
                suffix=["grasp"], target="grasp", post="lift"),
    "l2c": dict(prefix=["reach", "grasp", "lift"], check="lift", feat="hp",
                thresh=0.030, suffix=["carry"], target="carry", post=None),
    "c2p": dict(prefix=["reach", "grasp", "lift", "carry"], check="carry",
                feat="pg", thresh=0.070, suffix=["place"], target="place",
                post=None),
    "full": dict(prefix=["reach"], check="reach", feat="hp", thresh=0.010,
                 suffix=["grasp", "lift", "carry", "place"],
                 target="place", post=None),
}


def mcnemar_p(on_wins, off_wins):
    n = on_wins + off_wins
    if n == 0:
        return 1.0
    total = 0.0
    for k in range(max(on_wins, off_wins), n + 1):
        total += math.comb(n, k)
    return min(1.0, 2.0 * total * (0.5 ** n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="r2g", choices=list(PAIRS.keys()))
    ap.add_argument("--n_episodes", type=int, default=120)
    ap.add_argument("--k", type=int, default=3, help="最大重采样次数")
    ap.add_argument("--thresh", type=float, default=None,
                    help="覆盖该对的监测阈值(默认用 PAIRS 配置)")
    ap.add_argument("--seed0", type=int, default=8000,
                    help="评估种子段(与其他段不相交)")
    ap.add_argument("--out", default="dc_bm")
    args = ap.parse_args()
    cfg = dict(PAIRS[args.pair])
    if args.thresh is not None:
        cfg["thresh"] = args.thresh

    dp = SkillDP.load(os.path.join(MODEL_DIR, f"dp_{SCENE}.pt"), DEVICE)
    dp.eval()
    norm = load_norm()

    def run_episode(ep, monitor):
        env = make_env(SCENE, seed=args.seed0 + ep)
        obs, _ = env.reset()
        skill_succ = {}
        s_pre_check = None
        for si, name in enumerate(cfg["prefix"]):
            s_pre_check = save_state(env)
            obs = dp_rollout(dp, norm, env, obs, SID[name], STEPS,
                             seed=ep * 31 + si * 7 + 5)
            skill_succ[name] = float(
                diag_success(SCENE, name, env, obs, {}))
        entry_obs = obs
        g0 = geom(obs)[cfg["feat"]]
        attempts = 1
        # 边界监测 + best-of-K 重采样(重执行 check 技能)
        if monitor and g0 > cfg["thresh"]:
            best = (g0, obs, 0)
            ck_sid = SID[cfg["check"]]
            ck_seed = ep * 31 + (len(cfg["prefix"]) - 1) * 7 + 5
            for att in range(1, args.k + 1):
                o2 = dp_rollout(dp, norm, env,
                                restore_state(env, s_pre_check),
                                ck_sid, STEPS, ck_seed + att * 100)
                g = geom(o2)[cfg["feat"]]
                if g < best[0]:
                    best = (g, o2, att)
            obs = best[1]
            attempts = 1 + best[2]
            if attempts > 1:
                skill_succ[cfg["check"]] = float(
                    diag_success(SCENE, cfg["check"], env, obs, {}))
        g1 = geom(obs)[cfg["feat"]]
        # 后续链
        for si, name in enumerate(cfg["suffix"]):
            obs = dp_rollout(dp, norm, env, obs, SID[name], STEPS,
                             seed=ep * 31 + len(cfg["prefix"]) * 7 + si * 7
                             + 5)
            skill_succ[name] = float(
                diag_success(SCENE, name, env, obs, {}))
        # 下游评估
        if cfg["post"] == "lift":
            # settle 后冻结 lift ×K(§15 协议)
            for _ in range(SETTLE):
                obs, *_ = env.step(np.array([0.0, 0.0, 0.0, 1.0],
                                            dtype=np.float32))
            grasp_succ = float(diag_success(SCENE, "grasp", env, obs, {}))
            s0 = save_state(env)
            lift_p = eval_lift_after_settle(dp, norm, env, s0,
                                            seed0=4000 + ep * 100)
            skill_succ["grasp"] = grasp_succ
            skill_succ["lift_p"] = lift_p
            target_succ = float(grasp_succ > 0.5) * lift_p
        else:
            grasp_succ = float("nan")
            target_succ = float(diag_success(SCENE, cfg["target"], env,
                                             obs, {}))
        env.close()
        return dict(monitor=monitor, ep=ep, target_succ=target_succ,
                    entry0=g0, entry=g1, attempts=attempts,
                    grasp_succ=grasp_succ, skill_succ=skill_succ)

    rows_off, rows_on = [], []
    t0 = time.time()
    for ep in range(args.n_episodes):
        rows_off.append(run_episode(ep, monitor=False))
        rows_on.append(run_episode(ep, monitor=True))
        if (ep + 1) % 20 == 0:
            print(f"[bm-{args.pair}] ep{ep + 1} ({time.time() - t0:.0f}s)")

    def summarize(rows):
        s = np.array([r["target_succ"] for r in rows])
        e0 = np.array([r["entry0"] for r in rows])
        e1 = np.array([r["entry"] for r in rows])
        at = np.array([r["attempts"] for r in rows])
        out = dict(
            P_target=float(s.mean()), n=len(rows),
            entry0_mean=float(e0.mean()),
            entry_mean=float(e1.mean()),
            entry_improved=float((e1 < e0).mean()),
            attempts_mean=float(at.mean()),
            attempts_dist={str(a): int((at == a).sum())
                           for a in sorted(np.unique(at))})
        # 各技能段成功率(存在哪些技能取哪些)
        sk_names = sorted({k for r in rows for k in r["skill_succ"].keys()})
        out["per_skill"] = {
            k: float(np.nanmean(np.array(
                [r["skill_succ"].get(k, np.nan) for r in rows]))) 
            for k in sk_names}
        return out

    out = dict(args=vars(args), pair=args.pair, cfg=cfg,
               off=summarize(rows_off), on=summarize(rows_on))
    so = np.array([r["target_succ"] for r in rows_off])
    sn = np.array([r["target_succ"] for r in rows_on])
    a = int(((so < 0.5) & (sn >= 0.5)).sum())   # off 败 -> on 胜
    b = int(((so >= 0.5) & (sn < 0.5)).sum())   # off 胜 -> on 败
    out["mcnemar"] = dict(on_wins=a, off_wins=b, p=mcnemar_p(a, b))
    # Wilcoxon 符号秩(连续口径 target_succ)
    try:
        from scipy.stats import wilcoxon
        w = wilcoxon(sn, so)
        out["wilcoxon"] = dict(stat=float(w.statistic), p=float(w.pvalue))
    except Exception as e:  # noqa: BLE001
        out["wilcoxon"] = dict(error=str(e))
    out["rows_off"] = rows_off
    out["rows_on"] = rows_on
    path = os.path.join(EVAL_DIR, f"{args.out}_{args.pair}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items()
                      if not k.startswith("rows")}, indent=2)[:3000])
    print(f"[bm] saved {path}")


if __name__ == "__main__":
    main()