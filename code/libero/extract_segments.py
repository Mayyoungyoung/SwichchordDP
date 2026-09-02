"""LIBERO 演示 -> 技能分段数据集(启发式分段: 夹爪开合事件 + 末端速度谷值)。

输出: results/libero/data/{task}.h5
- obs [N, 9]  = robot_states(eef_pos3 + eef_quat4 + gripper_qpos1 + gripper_vel1)
- action [N, 7]
- skill [N]   分段标签(每段一个技能)
"""
import argparse
import glob
import os

import h5py
import numpy as np

DATASET = "/home/jia/VLA/libbero_tmp/datasets/libero_10"


def segment_demo(robot_states, actions):
    """启发式分段: 夹爪动作切换 + 长段内末端速度谷值切分。"""
    T = len(actions)
    grip_cmd = actions[:, -1]  # -1 开 / +1 闭
    closed = grip_cmd > 0.0
    boundaries = [0]
    for t in range(1, T):
        if closed[t] != closed[t - 1]:
            boundaries.append(t)
    # 长段切分(> 60 步)在速度谷值处再切
    vel = np.linalg.norm(np.diff(robot_states[:, :3], axis=0), axis=1)
    vel = np.concatenate([[0.0], vel])
    final = []
    for i in range(len(boundaries) - 1):
        lo, hi = boundaries[i], boundaries[i + 1]
        final.append(lo)
        if hi - lo > 80:
            mid = lo + 1 + int(np.argmin(vel[lo + 1:hi - 1])) if hi - lo > 2 else lo
            if hi - mid > 20 and mid - lo > 20:
                final.append(mid)
    final.append(T)
    labels = np.zeros(T, dtype=np.int64)
    for i in range(len(final) - 1):
        labels[final[i]:final[i + 1]] = i % 8  # 技能数封顶 8(对齐 SDP 的 8 原语)
    n_skills = min(8, len(final) - 1)
    labels[labels >= n_skills] = n_skills - 1
    return labels, n_skills


def extract(task_file: str, out_dir: str):
    task = os.path.basename(task_file).replace("_demo.hdf5", "")
    obs_list, act_list, skill_list = [], [], []
    with h5py.File(task_file, "r") as f:
        for dk in f["data"].keys():
            demo = f[f"data/{dk}"]
            robot_states = demo["robot_states"][:]
            actions = demo["actions"][:]
            labels, n_skills = segment_demo(robot_states, actions)
            obs_list.append(robot_states)
            act_list.append(actions)
            skill_list.append(labels)
    obs = np.concatenate(obs_list, 0)
    act = np.concatenate(act_list, 0)
    skill = np.concatenate(skill_list, 0)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{task}.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("obs", data=obs.astype(np.float32))
        f.create_dataset("action", data=act.astype(np.float32))
        f.create_dataset("skill", data=skill)
        f.create_dataset("obs_mean", data=obs.mean(0))
        f.create_dataset("obs_std", data=obs.std(0) + 1e-6)
        f.create_dataset("act_mean", data=act.mean(0))
        f.create_dataset("act_std", data=act.std(0) + 1e-6)
        f.attrs["n_skills"] = n_skills
        f.attrs["task"] = task
    seg_sizes = [(skill == i).sum() for i in range(n_skills)]
    print(f"[extract] {task}: {obs.shape}, {n_skills} skills, sizes={seg_sizes}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all", help="task 文件名前缀或 all")
    ap.add_argument("--out_dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../results/libero/data"))
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(DATASET, "*.hdf5")))
    for tf in files:
        if args.task != "all" and args.task not in os.path.basename(tf):
            continue
        extract(tf, args.out_dir)
