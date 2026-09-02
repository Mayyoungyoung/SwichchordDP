"""端到端基线评测: 单技能条件 DP 跑完整任务, 按窗口判定分技能成功。"""
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
from eval_compose import TASKS, skill_success, SKILL_NAMES  # noqa: E402
from skills import make_env  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "results/metaworld/data"
MODEL_DIR = "results/metaworld/models"


@torch.no_grad()
def rollout_e2e(dp, scene, seq, skill_steps, n_ddim=8, resample=8, seed=0):
    env = make_env(scene, seed=1000 + seed)
    obs, info = env.reset()
    with h5py.File(os.path.join(DATA_DIR, f"{scene}_full.h5"), "r") as f:
        obs_mean = f["obs_mean"][:]; obs_std = f["obs_std"][:]
        act_mean = f["act_mean"][:]; act_std = f["act_std"][:]
    s = torch.zeros(1, 1, device=DEVICE)
    s[0, 0] = 1.0

    def norm_obs(o):
        return torch.from_numpy((o - obs_mean) / obs_std).float().to(DEVICE).unsqueeze(0)

    chunk = dp.sample(norm_obs(obs), s, n_steps=n_ddim)
    nfe = n_ddim
    step_in_chunk = 0
    per_skill = {}
    cur_skill = 0
    step_in_skill = 0
    total = sum(skill_steps)
    exec_actions = []
    for t in range(total):
        a_raw = chunk[0, step_in_chunk].cpu().numpy() * act_std + act_mean
        exec_actions.append(a_raw)
        obs, rew, term, trunc, info = env.step(a_raw)
        step_in_chunk += 1
        step_in_skill += 1
        if step_in_skill >= skill_steps[cur_skill]:
            per_skill[seq[cur_skill]] = skill_success(scene, seq[cur_skill],
                                                      env, obs, info)
            cur_skill += 1
            step_in_skill = 0
        if step_in_chunk >= resample or step_in_chunk >= chunk.shape[1]:
            chunk = dp.sample(norm_obs(obs), s, n_steps=n_ddim)
            nfe += n_ddim
            step_in_chunk = 0
    exec_actions = np.array(exec_actions)
    energy = float(np.mean(np.sum(np.diff(exec_actions, axis=0) ** 2, axis=1)))
    e2e = float(all(per_skill.get(x, 0.0) > 0.5 for x in seq))
    env.close()
    return dict(seq=seq, per_skill=per_skill, e2e=e2e, energy=energy, nfe=nfe)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="pick-place-v3")
    ap.add_argument("--n_episodes", type=int, default=10)
    args = ap.parse_args()
    dp = SkillDP.load(os.path.join(MODEL_DIR, f"dp_{args.scene}_full.pt"), DEVICE)
    dp.eval()
    results = []
    for seq, skill_steps, kind in TASKS[args.scene]:
        succ = []
        for ep in range(args.n_episodes):
            t0 = time.time()
            r = rollout_e2e(dp, args.scene, seq, skill_steps, seed=ep * 7 + 1)
            r["latency_s"] = time.time() - t0
            r["kind"] = kind
            succ.append(r)
            print(f"[e2e] {seq} ep{ep} e2e={r['e2e']} energy={r['energy']:.3f}")
        results.append(dict(seq=seq, kind=kind,
                            e2e=float(np.mean([x["e2e"] for x in succ])),
                            per_skill={s_: float(np.mean(
                                [x["per_skill"].get(s_, 0.0) for x in succ]))
                                for s_ in seq},
                            episodes=succ))
    os.makedirs("results/metaworld/eval", exist_ok=True)
    with open(f"results/metaworld/eval/{args.scene}_e2e.json", "w") as f:
        json.dump(dict(args=vars(args), results=results), f, indent=2,
                  default=str)
    print("[e2e] saved")


if __name__ == "__main__":
    main()
