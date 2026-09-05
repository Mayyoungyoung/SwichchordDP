"""DP 链 setup 下收集 grasp 自然轨迹（普适性复验，免 lift 标签）。

与 dc_collect.py 的差别：
- 前置 setup 用冻结 DP reach（而非脚本 reach 控制器）：上游技能自身的
  自然散布使 grasp 的初始分布更宽（§13.5 的劣质率 ~8.3% 出现在此协议）；
- 不跑 lift outcome（本体验证「早期定型」普适性只无线标签）：
  分组标签 y 由 settle 后终态几何伪标签给出（defective/hp+oversqueeze），
  阈值在收集后按终态分布自然断点复核。

输出: results/metaworld/eval/dc_grasp_dpchain.h5
  obs (N,39) / action (N,4) / skill / traj_id （仅 grasp 段）
  y (n_traj,) 伪标签 / quality (n_traj,) / obs_t (n_traj,39)
"""
import argparse
import os
import time

import h5py
import numpy as np
import torch

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from swdp.policy import SkillDP  # noqa: E402
from skills import make_env, parse_pp  # noqa: E402
from diag_handoff import load_norm  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCENE = "pick-place-v3"
EVAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "../../results/metaworld/eval")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "../../results/metaworld/models")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "../../results/metaworld/data")
SID_REACH, SID_GRASP, SID_LIFT = 0, 1, 2
REACH_STEPS, GRASP_STEPS, SETTLE = 30, 30, 20
# 伪标签阈值(脚本 setup 经验值, 收集后按分布复核):
HP_DEFECTIVE, GRIP_DEFECTIVE = 0.012, 0.40


@torch.no_grad()
def dp_rollout(dp, norm, env, obs, sid, n_steps, seed):
    """DP 技能自然执行 n_steps, 返回 (逐步 obs 序列, action 序列, 终态 obs)。"""
    obs_mean, obs_std, act_mean, act_std = norm

    def norm_obs(o):
        return torch.from_numpy((o - obs_mean) / obs_std).float().to(
            DEVICE).unsqueeze(0)

    def onehot(i):
        z = np.zeros((1, dp.n_skills), dtype=np.float32)
        z[0, i] = 1.0
        return torch.from_numpy(z).to(DEVICE)

    obs_seq, act_seq = [], []
    chunk = dp.sample(norm_obs(obs), onehot(sid), n_steps=24, seed=seed)
    step_in = 0
    for t in range(n_steps):
        obs_seq.append(np.asarray(obs, dtype=np.float32))
        a_raw = np.clip(chunk[0, step_in].cpu().numpy() * act_std
                        + act_mean, -1.0, 1.0)
        act_seq.append(a_raw.astype(np.float32))
        obs, *_ = env.step(a_raw)
        step_in += 1
        if step_in >= 8:
            chunk = dp.sample(norm_obs(obs), onehot(sid), n_steps=24,
                              seed=seed + t)
            step_in = 0
    return np.array(obs_seq), np.array(act_seq), obs


def terminal_geometry(obs):
    """终态几何: (hp, grip)。"""
    o = parse_pp(obs)
    hp = float(np.linalg.norm(np.asarray(o["hand"][:2])
                              - np.asarray(o["puck"][:2])))
    return hp, float(o["grip"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_episodes", type=int, default=240)
    ap.add_argument("--seed0", type=int, default=6000,
                    help="收集种子段(与 3000/5000 不相交)")
    ap.add_argument("--out", default="dc_grasp_dpchain")
    args = ap.parse_args()

    dp = SkillDP.load(os.path.join(MODEL_DIR, f"dp_{SCENE}.pt"), DEVICE)
    dp.eval()
    norm = load_norm()

    obs_all, act_all, skill_all, traj_all = [], [], [], []
    ys, geodes, obs_ts = [], [], []
    t0 = time.time()
    for ep in range(args.n_episodes):
        env = make_env(SCENE, seed=args.seed0 + ep)
        obs, _ = env.reset()
        # DP 链前置: 冻结 DP reach 自然执行(宽散布 setup)
        _, _, obs = dp_rollout(dp, norm, env, obs, SID_REACH,
                               REACH_STEPS, seed=ep * 13 + 5)
        # DP grasp 自然轨迹(含劣质抓取), 全程记录逐步 obs
        o_seq, a_seq, obs_t = dp_rollout(dp, norm, env, obs, SID_GRASP,
                                         GRASP_STEPS, seed=ep * 7 + 1)
        for _ in range(SETTLE):
            obs_t, *_ = env.step(np.array([0.0, 0.0, 0.0, 1.0],
                                          dtype=np.float32))
        hp_t, grip_t = terminal_geometry(obs_t)
        y = float(hp_t <= HP_DEFECTIVE and grip_t >= GRIP_DEFECTIVE)
        geodes.append((hp_t, grip_t))
        obs_ts.append(np.asarray(obs_t, dtype=np.float32))
        obs_all.append(o_seq)
        act_all.append(a_seq)
        skill_all.append(np.full(len(o_seq), SID_GRASP, dtype=np.int64))
        traj_all.append(np.full(len(o_seq), ep, dtype=np.int64))
        ys.append(y)
        env.close()
        if (ep + 1) % 40 == 0:
            el = time.time() - t0
            print(f"[dc-dpchain] ep{ep + 1}: 伪标签 pos={np.mean(ys):.3f} "
                  f"({el:.0f}s)")

    obs_all = np.concatenate(obs_all)
    act_all = np.concatenate(act_all)
    skill_all = np.concatenate(skill_all)
    traj_all = np.concatenate(traj_all)
    hp_arr = np.array([g[0] for g in geodes])
    grip_arr = np.array([g[1] for g in geodes])
    print(f"[dc-dpchain] collected {len(ys)} trajs, {len(obs_all)} steps, "
          f"伪标签 pos={np.mean(ys):.3f}")
    print(f"[dc-dpchain] 终态 hp: mean={hp_arr.mean():.4f} "
          f"p50={np.median(hp_arr):.4f} p90={np.percentile(hp_arr, 90):.4f} "
          f"max={hp_arr.max():.4f}")
    print(f"[dc-dpchain] 终态 grip: mean={grip_arr.mean():.4f} "
          f"p10={np.percentile(grip_arr, 10):.4f} "
          f"p50={np.median(grip_arr):.4f} min={grip_arr.min():.4f}")

    data_path = os.path.join(DATA_DIR, "pick-place-v3.h5")
    with h5py.File(data_path, "r") as f:
        obs_mean, obs_std = f["obs_mean"][:], f["obs_std"][:]
        act_mean, act_std = f["act_mean"][:], f["act_std"][:]
        n_skills = f.attrs["n_skills"]
    out_path = os.path.join(EVAL_DIR, args.out + ".h5")
    with h5py.File(out_path, "w") as f:
        f.create_dataset("obs", data=obs_all)
        f.create_dataset("action", data=act_all)
        f.create_dataset("skill", data=skill_all)
        f.create_dataset("traj_id", data=traj_all)
        f.create_dataset("y", data=np.array(ys, dtype=np.float32))
        f.create_dataset("quality", data=np.array(
            [float(hp_arr[i] <= HP_DEFECTIVE
                   and grip_arr[i] >= GRIP_DEFECTIVE)
             for i in range(len(ys))], dtype=np.float32))
        f.create_dataset("obs_t", data=np.array(obs_ts))
        f.attrs["n_skills"] = n_skills
        f["obs_mean"] = obs_mean
        f["obs_std"] = obs_std
        f["act_mean"] = act_mean
        f["act_std"] = act_std
    print(f"[dc-dpchain] saved {out_path}")


if __name__ == "__main__":
    main()