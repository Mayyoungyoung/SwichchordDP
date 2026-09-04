"""Downstream-Compatible Skill Learning 数据收集（grasp→lift 第一轮验证）。

收集 DP grasp 自然轨迹（含劣质抓取）+ 每条轨迹的下游 lift outcome 标签:
- 240 episodes: 脚本 reach setup(30) → 冻结 DP grasp(30, 记录每步 obs/action)
  → settle(20) → 终态保存 + lift ×10 真实 rollout → y_i
- 额外记录终态几何质量(grasp-quality, 供 baseline3 加权)

输出: results/metaworld/eval/dc_grasp_lift.h5
  obs (N,39) / action (N,4) / skill (N,) / traj_id (N,) /
  y (n_traj,) / quality (n_traj,) / states (n_traj, dim) /
  obs_t (n_traj,39)  # settle 后终态 obs(供 fb 加权与 F_B 评估)
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
from skills import make_env, SKILLS, parse_pp  # noqa: E402
from diag_handoff import load_norm, diag_success  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCENE = "pick-place-v3"
EVAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "../../results/metaworld/eval")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "../../results/metaworld/models")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "../../results/metaworld/data")
SID_GRASP, SID_LIFT = 1, 2
SETUP_STEPS = 30
GRASP_STEPS, K_LIFT, SETTLE = 30, 10, 20


@torch.no_grad()
def run_grasp_traj(dp, norm, env, obs, seed):
    """DP grasp 30 步, 返回 (obs 序列, action 序列, 终态 obs)。"""
    obs_mean, obs_std, act_mean, act_std = norm

    def norm_obs(o):
        return torch.from_numpy((o - obs_mean) / obs_std).float().to(
            DEVICE).unsqueeze(0)

    def onehot(i):
        z = np.zeros((1, dp.n_skills), dtype=np.float32)
        z[0, i] = 1.0
        return torch.from_numpy(z).to(DEVICE)

    obs_seq, act_seq = [], []
    chunk = dp.sample(norm_obs(obs), onehot(SID_GRASP), n_steps=24,
                      seed=seed)
    step_in = 0
    for t in range(GRASP_STEPS):
        obs_seq.append(np.asarray(obs, dtype=np.float32))
        a_raw = np.clip(chunk[0, step_in].cpu().numpy() * act_std
                        + act_mean, -1.0, 1.0)
        act_seq.append(a_raw.astype(np.float32))
        obs, *_ = env.step(a_raw)
        step_in += 1
        if step_in >= 8:
            chunk = dp.sample(norm_obs(obs), onehot(SID_GRASP), n_steps=24,
                              seed=seed + t)
            step_in = 0
    return np.array(obs_seq), np.array(act_seq), obs


@torch.no_grad()
def lift_outcome(dp, norm, env, s0, k=K_LIFT, seed0=4000):
    """从终态 s0 跑 lift × K（每次 restore）→ success rate。"""
    obs_mean, obs_std, act_mean, act_std = norm

    def norm_obs(o):
        return torch.from_numpy((o - obs_mean) / obs_std).float().to(
            DEVICE).unsqueeze(0)

    def onehot(i):
        z = np.zeros((1, dp.n_skills), dtype=np.float32)
        z[0, i] = 1.0
        return torch.from_numpy(z).to(DEVICE)

    import mujoco
    succs = []
    for kk in range(k):
        d = env._env.data
        m = env._env.model
        nq, nv = d.qpos.shape[0], d.qvel.shape[0]
        d.qpos[:] = s0[:nq]
        d.qvel[:] = s0[nq:nq + nv]
        d.mocap_pos[:] = s0[nq + nv:nq + nv + 3]
        d.mocap_quat[:] = s0[nq + nv + 3:nq + nv + 7]
        mujoco.mj_forward(m, d)
        env._env.curr_path_length = 0
        obs = env._env._get_obs()
        chunk = dp.sample(norm_obs(obs), onehot(SID_LIFT), n_steps=24,
                          seed=seed0 + kk)
        step_in = 0
        info = {}
        for t in range(25):
            a = np.clip(chunk[0, step_in].cpu().numpy() * act_std
                        + act_mean, -1.0, 1.0)
            obs, rew, term, trunc, info = env.step(a)
            step_in += 1
            if step_in >= 8:
                chunk = dp.sample(norm_obs(obs), onehot(SID_LIFT),
                                  n_steps=24, seed=seed0 + kk + t)
                step_in = 0
        succs.append(float(diag_success(SCENE, "lift", env, obs, info)))
    return float(np.mean(succs))


def grasp_quality(obs):
    """grasp 自身质量(基线3 的加权信号): 对中 + 正常闭合。

    与下游可行性相关但不含 lift 语义: hand-puck xy 小 + grip 接近 0.44。"""
    o = parse_pp(obs)
    hp = np.linalg.norm(o["hand"][:2] - o["puck"][:2])
    grip_err = abs(float(o["grip"]) - 0.44)
    q = float(np.exp(-hp / 0.02) * np.exp(-grip_err / 0.05))
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_episodes", type=int, default=240)
    ap.add_argument("--k_lift", type=int, default=10,
                    help="每条轨迹下游 lift rollout 次数(outcome 标签)")
    ap.add_argument("--seed0", type=int, default=3000)
    ap.add_argument("--out", default="dc_grasp_lift")
    args = ap.parse_args()

    dp = SkillDP.load(os.path.join(MODEL_DIR, f"dp_{SCENE}.pt"), DEVICE)
    dp.eval()
    norm = load_norm()

    obs_all, act_all, skill_all, traj_all = [], [], [], []
    ys, quals, states, obs_ts = [], [], [], []
    t0 = time.time()
    for ep in range(args.n_episodes):
        env = make_env(SCENE, seed=args.seed0 + ep)
        obs, _ = env.reset()
        ctrl = SKILLS[SCENE]["reach"](env)
        for _ in range(SETUP_STEPS):
            obs, *_ = env.step(ctrl.act(obs))
        # DP grasp 轨迹(自然, 含劣质抓取)
        o_seq, a_seq, obs_t = run_grasp_traj(dp, norm, env, obs,
                                             seed=ep * 7 + 1)
        # settle 后保存终态
        for _ in range(SETTLE):
            obs_t, *_ = env.step(np.array([0.0, 0.0, 0.0, 1.0],
                                          dtype=np.float32))
        d = env._env.data
        s0 = np.concatenate([d.qpos.copy(), d.qvel.copy(),
                             d.mocap_pos.ravel().copy(),
                             d.mocap_quat.ravel().copy()])
        y = lift_outcome(dp, norm, env, s0, k=args.k_lift,
                         seed0=4000 + ep * 100)
        q = grasp_quality(obs_t)
        obs_ts.append(np.asarray(obs_t, dtype=np.float32))
        # 记录(仅记录 grasp 段, reach 是脚本 setup 不入训练数据)
        obs_all.append(o_seq)
        act_all.append(a_seq)
        skill_all.append(np.full(len(o_seq), SID_GRASP, dtype=np.int64))
        traj_all.append(np.full(len(o_seq), ep, dtype=np.int64))
        ys.append(y)
        quals.append(q)
        states.append(s0)
        env.close()
        if (ep + 1) % 40 == 0:
            el = time.time() - t0
            print(f"[dc] ep{ep + 1}: y分布 so far "
                  f"pos={np.mean(ys):.3f} ({el:.0f}s)")

    obs_all = np.concatenate(obs_all)
    act_all = np.concatenate(act_all)
    skill_all = np.concatenate(skill_all)
    traj_all = np.concatenate(traj_all)
    ys = np.array(ys, dtype=np.float32)
    quals = np.array(quals, dtype=np.float32)
    print(f"[dc] collected {len(ys)} trajs, {len(obs_all)} steps, "
          f"y pos rate={ys.mean():.3f}, grasp 质量 q mean={quals.mean():.3f}")
    print(f"[dc] y=0 轨迹数 {np.sum(ys == 0)}, y=1 轨迹数 {np.sum(ys == 1.0)}")

    # 保存 h5(复用演示数据集的规范化: 用现有 pick-place-v3.h5 的 mean/std)
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
        f.create_dataset("y", data=ys)
        f.create_dataset("quality", data=quals)
        f.create_dataset("states", data=np.array(states))
        f.create_dataset("obs_t", data=np.array(obs_ts))
        f.attrs["n_skills"] = n_skills
        f["obs_mean"] = obs_mean
        f["obs_std"] = obs_std
        f["act_mean"] = act_mean
        f["act_std"] = act_std
    print(f"[dc] saved {out_path}")


if __name__ == "__main__":
    main()
