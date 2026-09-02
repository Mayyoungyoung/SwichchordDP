"""采集技能演示数据(脚本策略子阶段控制器)。

每个技能 = 一段短演示; 采集时先执行前置技能建立状态, 再执行本技能并记录。
输出: h5 文件, 包含 obs [N, d_obs], action [N, da], skill [N], 以及归一化统计。
"""
import argparse
import os

import h5py
import numpy as np

from skills import make_env, SKILLS

# 场景 -> (技能列表, 每个技能的 [前置技能, 前置步数, 记录步数])
SCENES = {
    "pick-place-v3": {
        "reach": (["reach"], [0], 30),
        "grasp": (["reach"], [30], 30),
        "lift": (["reach", "grasp"], [30, 30], 25),
        "carry": (["reach", "grasp", "lift"], [30, 30, 25], 30),
        "place": (["reach", "grasp", "lift", "carry"], [30, 30, 25, 30], 25),
    },
    "door-open-v3": {
        "reach": (["reach"], [0], 22),
        "open": (["reach"], [14], 22),
    },
}


def collect(scene: str, n_demos: int, out_dir: str, seed0: int = 0):
    spec = SCENES[scene]
    skill_names = list(spec.keys())
    obs_list, act_list, skill_list = [], [], []

    for si, name in enumerate(skill_names):
        pre_seq, pre_steps, rec_steps = spec[name]
        cls = SKILLS[scene]
        for d in range(n_demos):
            env = make_env(scene, seed=seed0 + si * 1000 + d)
            obs, _ = env.reset()
            ctrls = [cls[n](env) for n in pre_seq]
            # 前置阶段(不记录)
            for ctrl, n in zip(ctrls, pre_steps):
                for _ in range(n):
                    a = ctrl.act(obs)
                    obs, _, _, _, _ = env.step(a)
            # 记录阶段
            rec_ctrl = cls[name](env)
            for _ in range(rec_steps):
                a = rec_ctrl.act(obs)
                obs_list.append(obs.astype(np.float32))
                act_list.append(a.astype(np.float32))
                skill_list.append(si)
                obs, _, _, _, _ = env.step(a)
            env.close()
            print(f"[collect] {scene}/{name} demo {d+1}/{n_demos}")

    obs = np.stack(obs_list)
    act = np.stack(act_list)
    skill = np.array(skill_list, dtype=np.int64)
    obs_mean, obs_std = obs.mean(0), obs.std(0) + 1e-6
    act_mean, act_std = act.mean(0), act.std(0) + 1e-6

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{scene}.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("obs", data=obs)
        f.create_dataset("action", data=act)
        f.create_dataset("skill", data=skill)
        f.create_dataset("obs_mean", data=obs_mean)
        f.create_dataset("obs_std", data=obs_std)
        f.create_dataset("act_mean", data=act_mean)
        f.create_dataset("act_std", data=act_std)
        f.attrs["skill_names"] = [n.encode() for n in skill_names]
        f.attrs["n_skills"] = len(skill_names)
    print(f"[collect] saved {path}: {obs.shape}, skills={skill_names}")
    # 打印各技能样本统计
    for si, n in enumerate(skill_names):
        print(f"  skill {n}: {(skill == si).sum()} steps")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="pick-place-v3")
    ap.add_argument("--n_demos", type=int, default=30)
    ap.add_argument("--out_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../results/metaworld/data"))
    args = ap.parse_args()
    collect(args.scene, args.n_demos, args.out_dir)
