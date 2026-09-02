"""采集完整任务链演示(端到端基线训练数据, 单技能标签)。"""
import argparse
import os

import h5py
import numpy as np

from skills import make_env, SKILLS

FULL_SEQ = {
    "pick-place-v3": ["reach", "grasp", "lift", "carry", "place"],
    "door-open-v3": ["reach", "open"],
}


def collect_full(scene: str, n_demos: int, out_dir: str, seed0: int = 0):
    seq = FULL_SEQ[scene]
    obs_list, act_list = [], []
    for d in range(n_demos):
        env = make_env(scene, seed=seed0 + d)
        obs, _ = env.reset()
        steps = {"reach": 30, "grasp": 25, "lift": 25, "carry": 30,
                 "place": 20, "open": 25}
        ctrls = [SKILLS[scene][n](env) for n in seq]
        for name, ctrl in zip(seq, ctrls):
            for _ in range(steps[name]):
                a = ctrl.act(obs)
                obs_list.append(obs.astype(np.float32))
                act_list.append(a.astype(np.float32))
                obs, _, _, _, _ = env.step(a)
        env.close()
        print(f"[full] {scene} demo {d+1}/{n_demos}")
    obs = np.stack(obs_list)
    act = np.stack(act_list)
    path = os.path.join(out_dir, f"{scene}_full.h5")
    os.makedirs(out_dir, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("obs", data=obs)
        f.create_dataset("action", data=act)
        f.create_dataset("skill", data=np.zeros(len(obs), dtype=np.int64))
        f.create_dataset("obs_mean", data=obs.mean(0))
        f.create_dataset("obs_std", data=obs.std(0) + 1e-6)
        f.create_dataset("act_mean", data=act.mean(0))
        f.create_dataset("act_std", data=act.std(0) + 1e-6)
        f.attrs["n_skills"] = 1
        f.attrs["skill_names"] = [b"full"]
    print(f"[full] saved {path}: {obs.shape}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="pick-place-v3")
    ap.add_argument("--n_demos", type=int, default=40)
    ap.add_argument("--out_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../results/metaworld/data"))
    args = ap.parse_args()
    collect_full(args.scene, args.n_demos, args.out_dir)
