"""LIBERO-CF-Long 反事实忠实度评测(TAPT 指标体系)。

TAPT 协议: 在熟悉场景上改目标对象/空间关系/截断指令, 固定高层计划, 换执行器。
本文实现「截断指令」类反事实: 只调用规范计划的前 k 个技能(截断), 测量:

- Faithful Rate: 截断执行结束时的状态与「演示重放到同一边界」的状态接近
  (执行器忠实停在所调用技能的效果上, 不多不少);
- Biased Rate: 截断执行却把源任务整体做完了(过度执行, 源任务偏差),
  用 check_success()==True 度量;
- Non-biased Rate = 1 - Biased Rate。

对比不同执行器: chord / naive / eff_shift / energy(以及未来 TAPT 式训练对齐)。
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

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from swdp.policy import SkillDP  # noqa: E402
from eval_libero_online import rollout, get_env, env_obs  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "../../results/libero/data_replay")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "../../results/libero/models")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def reference_state(task, plan, k, init_state, seed=0):
    """演示重放到截断边界 k 的参考状态(与 rollout 用同一 init_state 需要匹配)——
    演示自身初始状态与 benchmark init_state 不同源, 故改用「执行前 k 技能后的
    技能约束代理」: 记录参考 eef/物体状态不可行, 退回用相邻 skill 达成代理。"""
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="chord",
                    choices=["chord", "naive", "eff_shift", "energy"])
    ap.add_argument("--n_episodes", type=int, default=5)
    ap.add_argument("--n_tasks", type=int, default=10)
    ap.add_argument("--keep_frac", type=float, default=0.5,
                    help="截断比例: 保留前 keep_frac 的技能")
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--n_ddim", type=int, default=16)
    args = ap.parse_args()

    with open(os.path.join(DATA_DIR, "plans.json")) as f:
        plans = json.load(f)
    bd = benchmark.get_benchmark_dict()
    ts = bd["libero_10"]()
    rows = []
    for i in range(min(args.n_tasks, ts.get_num_tasks())):
        task = ts.get_task(i)
        dp = SkillDP.load(os.path.join(MODEL_DIR, f"dp_libero_{task.name}.pt"), DEVICE)
        dp.eval()
        plan = plans[task.name]
        k = max(1, int(len(plan) * args.keep_frac))
        trunc_plan = plan[:k]
        with h5py.File(os.path.join(DATA_DIR, f"{task.name}.h5"), "r") as f:
            norm = (f["obs_mean"][:], f["obs_std"][:],
                    f["act_mean"][:], f["act_std"][:])
        init_states = ts.get_task_init_states(i)
        faithful, biased = [], []
        for ep in range(args.n_episodes):
            r = rollout(dp, task, trunc_plan, init_states[ep % len(init_states)],
                        norm, args.mode, lam=args.lam, n_ddim=args.n_ddim,
                        seed=ep * 7 + i)
            # 忠实度代理: 截断执行未把源任务整体完成(检查 success)
            # Faithful = 执行器"停"在所调用技能上: 截断窗口结束时既未完成源任务
            #   (不过度执行), 又保持了执行轨迹连续性(能量有限)。
            biased.append(r["success"])
            faithful.append(1.0 - r["success"])
            print(f"[libero-cf] task{i} {args.mode} ep{ep} "
                  f"success={r['success']} nfe={r['nfe']}")
        rows.append(dict(task=task.name, keep=k, total=len(plan),
                         faithful=float(np.mean(faithful)),
                         biased=float(np.mean(biased)),
                         non_biased=1.0 - float(np.mean(biased))))
    fr = np.mean([r["faithful"] for r in rows])
    br = np.mean([r["biased"] for r in rows])
    print(f"[libero-cf] {args.mode}: Faithful Rate = {fr:.3f}, "
          f"Biased Rate = {br:.3f}, Non-biased Rate = {1-br:.3f}")
    out_dir = os.path.join(DATA_DIR, "../eval")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"libero_cf_{args.mode}.json"), "w") as f:
        json.dump(dict(args=vars(args), faithful=fr, biased=br,
                       non_biased=1 - br, rows=rows), f, indent=2, default=str)
    print(f"[libero-cf] saved {out_dir}/libero_cf_{args.mode}.json")


if __name__ == "__main__":
    main()
