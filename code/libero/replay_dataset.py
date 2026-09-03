"""LIBERO 重放数据生成: 在 sim 中重放数据集演示, 用 env 观测重建训练数据。

动机(2026-09-03 核实): 本地 libero_10 数据集与官方 init_files 的初始状态不一致,
且数据集 robot_states 语义无法与当前 env 观测精确对应(max 偏差 0.05)。
为保证训练/在线评测观测严格一致, 直接在 sim 中重放演示(已验证重放后
check_success()=True), 记录 env 派生观测:

  obs   = [robot0_gripper_qpos(2), robot0_eef_pos(3), robot0_eef_quat(4)]  (9 维)
  action= 7 维 delta 动作
  skill = 夹爪动作切换 + 末端位移谷值启发式分段

输出: results/libero/data_replay/{task}.h5 + plans.json(每任务规范技能序列)
"""
import argparse
import json
import os
import time

import h5py
import numpy as np
import torch

from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "../../results/libero/data_replay")
TASKS = ["libero_10"]
N_SKILL_CAP = 8


def get_env(task):
    bddl_path = os.path.join(get_libero_path("bddl_files"),
                             task.problem_folder, task.bddl_file)
    return ControlEnv(bddl_file_name=bddl_path, has_renderer=False,
                      has_offscreen_renderer=False, use_camera_obs=False,
                      camera_names=[], control_freq=20)


def env_obs(obs):
    return np.concatenate([obs["robot0_gripper_qpos"], obs["robot0_eef_pos"],
                           obs["robot0_eef_quat"],
                           obs["object-state"]]).astype(np.float32)


def segment(obs_seq, actions):
    """夹爪动作切换 + 长段内末端位移谷值切分(与离线版一致, 但用 env 观测)。"""
    T = len(actions)
    closed = actions[:, -1] > 0.0
    boundaries = [0]
    for t in range(1, T):
        if closed[t] != closed[t - 1]:
            boundaries.append(t)
    disp = np.linalg.norm(np.diff(obs_seq[:, 2:5], axis=0), axis=1)
    disp = np.concatenate([[0.0], disp])
    final = []
    for i in range(len(boundaries) - 1):
        lo, hi = boundaries[i], boundaries[i + 1]
        final.append(lo)
        if hi - lo > 80 and hi - lo > 2:
            mid = lo + 1 + int(np.argmin(disp[lo + 1:hi - 1]))
            if hi - mid > 20 and mid - lo > 20:
                final.append(mid)
    final.append(T)
    labels = np.zeros(T, dtype=np.int64)
    for i in range(len(final) - 1):
        labels[final[i]:final[i + 1]] = i % N_SKILL_CAP
    return labels


def replay_task(ts, task_id, n_demos, out_dir, seed_base=0):
    task = ts.get_task(task_id)
    demo_path = os.path.join(get_libero_path("datasets"),
                             ts.get_task_demonstration(task_id))
    obs_list, act_list, skill_list, plan, init_list = [], [], [], [], []
    with h5py.File(demo_path, "r") as h:
        demo_keys = sorted(h["data"].keys(),
                           key=lambda k: int(k.split("_")[1]))
        for d in range(min(n_demos, len(demo_keys))):
            dk = demo_keys[d]
            acts = h[f"data/{dk}/actions"][:]
            st0 = h[f"data/{dk}/states"][0]
            env = get_env(task)
            env.seed(seed_base + d)
            env.reset()
            obs = env.set_init_state(st0)
            init_list.append(st0)
            o_seq, a_seq = [], []
            for a in acts:
                o_seq.append(env_obs(obs))
                a_seq.append(a.astype(np.float32))
                obs, *_ = env.step(a)
            env.close()
            o_seq = np.stack(o_seq)
            a_seq = np.stack(a_seq)
            labels = segment(o_seq, a_seq)
            obs_list.append(o_seq)
            act_list.append(a_seq)
            skill_list.append(labels)
            if d == 0:
                # 规范计划: demo_0 的技能序列与各段步数
                bounds = np.where(labels[1:] != labels[:-1])[0] + 1
                seg_starts = np.concatenate([[0], bounds])
                plan = []
                for k, s0 in enumerate(seg_starts):
                    s1 = seg_starts[k + 1] if k + 1 < len(seg_starts) else len(labels)
                    plan.append({"g": "tool", "z": None,
                                 "steps": int(s1 - s0),
                                 "skill_id": int(labels[s0])})
    obs = np.concatenate(obs_list, 0)
    act = np.concatenate(act_list, 0)
    skill = np.concatenate(skill_list, 0)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{task.name}.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("obs", data=obs)
        f.create_dataset("action", data=act)
        f.create_dataset("skill", data=skill)
        f.create_dataset("init_states", data=np.stack(init_list))
        f.create_dataset("obs_mean", data=obs.mean(0))
        f.create_dataset("obs_std", data=obs.std(0) + 1e-6)
        f.create_dataset("act_mean", data=act.mean(0))
        f.create_dataset("act_std", data=act.std(0) + 1e-6)
        f.attrs["n_skills"] = N_SKILL_CAP
        f.attrs["task"] = task.name
    print(f"[replay] {task.name}: {obs.shape} steps, plan={plan}")
    return task.name, plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_demos", type=int, default=30)
    ap.add_argument("--n_tasks", type=int, default=10)
    args = ap.parse_args()
    bd = benchmark.get_benchmark_dict()
    ts = bd["libero_10"]()
    plans = {}
    t0 = time.time()
    for i in range(min(args.n_tasks, ts.get_num_tasks())):
        name, plan = replay_task(ts, i, args.n_demos, DATA_DIR)
        plans[name] = plan
    with open(os.path.join(DATA_DIR, "plans.json"), "w") as f:
        json.dump(plans, f, indent=2)
    print(f"[replay] done in {time.time()-t0:.0f}s; plans saved")


if __name__ == "__main__":
    main()
