"""全链 DP 自然执行收集（多对边界传导复验）。

一次收集覆盖 4 个技能边界：
  reach→grasp / grasp→lift / lift→carry / carry→place

- 全链冻结 DP 自然执行（每技能 30 步, 无 settle, 与 §13.5 termdiv 协议一致）
- 每步记录执行前 obs（供逐步曲线分析）+ skill 段标签
- 免 lift outcome 标签: 各边界质量由 B 段终态几何/语义谓词定（见 dc_probe_pairs.py）

输出: results/metaworld/eval/dc_chain_full.h5
  obs (N,39) / action (N,4) / skill (N,) 每步所属技能 id /
  traj_id (N,) / n_skills + 规范化参数
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
from skills import make_env  # noqa: E402
from diag_handoff import load_norm  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCENE = "pick-place-v3"
EVAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "../../results/metaworld/eval")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "../../results/metaworld/models")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "../../results/metaworld/data")
SID = {"reach": 0, "grasp": 1, "lift": 2, "carry": 3, "place": 4}
CHAIN = ["reach", "grasp", "lift", "carry", "place"]
STEPS = 30


@torch.no_grad()
def dp_rollout(dp, norm, env, obs, sid, n_steps, seed):
    """DP 技能自然执行 n_steps, 返回 (逐步 obs 列表, action 列表, 执行后 obs)。"""
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
    return obs_seq, act_seq, obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_episodes", type=int, default=240)
    ap.add_argument("--seed0", type=int, default=7000,
                    help="收集种子段(与 3000/5000/6000 不相交)")
    ap.add_argument("--out", default="dc_chain_full")
    args = ap.parse_args()

    dp = SkillDP.load(os.path.join(MODEL_DIR, f"dp_{SCENE}.pt"), DEVICE)
    dp.eval()
    norm = load_norm()

    obs_all, act_all, skill_all, traj_all = [], [], [], []
    t0 = time.time()
    for ep in range(args.n_episodes):
        env = make_env(SCENE, seed=args.seed0 + ep)
        obs, _ = env.reset()
        for si, name in enumerate(CHAIN):
            seed = ep * 31 + si * 7 + 5
            o_seq, a_seq, obs = dp_rollout(dp, norm, env, obs,
                                           SID[name], STEPS, seed)
            obs_all.append(o_seq)
            act_all.append(a_seq)
            skill_all.append(np.full(len(o_seq), SID[name],
                                     dtype=np.int64))
            traj_all.append(np.full(len(o_seq), ep, dtype=np.int64))
        env.close()
        if (ep + 1) % 40 == 0:
            el = time.time() - t0
            print(f"[fullchain] ep{ep + 1} ({el:.0f}s)")

    obs_all = np.concatenate(obs_all)
    act_all = np.concatenate(act_all)
    skill_all = np.concatenate(skill_all)
    traj_all = np.concatenate(traj_all)
    print(f"[fullchain] collected {len(traj_all)} steps "
          f"({args.n_episodes} trajs) in {time.time() - t0:.0f}s")

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
        f.attrs["n_skills"] = n_skills
        f["obs_mean"] = obs_mean
        f["obs_std"] = obs_std
        f["act_mean"] = act_mean
        f["act_std"] = act_std
    print(f"[fullchain] saved {out_path}")


if __name__ == "__main__":
    main()