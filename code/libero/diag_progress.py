"""诊断: 在线 rollout 与演示重放的分段进度对比。

对每个任务: (1) 用演示动作重放得到各技能边界处的参考 eef 位置;
(2) 用 DP rollout(chord/naive)从同一初始状态执行, 在各边界处比较 eef 距离;
(3) 输出「技能完成率」(边界处 eef 距离 < 0.15 的技能占比)与平均距离。
"""
import json
import os
import sys

import h5py
import numpy as np
import torch

from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from swdp.policy import SkillDP  # noqa: E402
from eval_libero_online import rollout, get_env, env_obs  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "../../results/libero/data_replay")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "../../results/libero/models")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def reference_boundaries(task, plan, init_state, seed=0):
    """演示重放, 返回每个技能边界处的 eef 位置。"""
    demo_path = os.path.join(get_libero_path("datasets"),
                             benchmark.get_benchmark_dict()["libero_10"]()
                             .get_task_demonstration(0))
    # 直接用重放数据: 该任务的 demo_0 前向重放
    env = get_env(task)
    env.seed(seed)
    env.reset()
    # 用 eval 同样的 init_state(重放数据 demo_0 的初始状态)
    env.set_init_state(init_state)
    bounds = []
    acc = 0
    for p in plan:
        acc += p["steps"]
        bounds.append(acc)
    refs = []
    t = 0
    with h5py.File(os.path.join(DATA_DIR, f"{task.name}.h5"), "r") as f:
        acts = f["action"][:]
    for b in bounds:
        while t < b and t < len(acts):
            obs, *_ = env.step(acts[t])
            t += 1
        refs.append(env_obs(obs)[2:5].copy())
    env.close()
    return refs


def main():
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--mode", default="chord")
    ap.add_argument("--n_episodes", type=int, default=5)
    ap.add_argument("--n_tasks", type=int, default=10)
    args = ap.parse_args()
    ts = benchmark.get_benchmark_dict()["libero_10"]()
    with open(os.path.join(DATA_DIR, "plans.json")) as f:
        plans = json.load(f)
    rows = []
    for i in range(min(args.n_tasks, ts.get_num_tasks())):
        task = ts.get_task(i)
        plan = plans[task.name]
        dp = SkillDP.load(os.path.join(MODEL_DIR, f"dp_libero_{task.name}.pt"), DEVICE)
        dp.eval()
        with h5py.File(os.path.join(DATA_DIR, f"{task.name}.h5"), "r") as f:
            norm = (f["obs_mean"][:], f["obs_std"][:],
                    f["act_mean"][:], f["act_std"][:])
            init_states = f["init_states"][:]
        task_done = []
        for ep in range(min(args.n_episodes, len(init_states))):
            init_state = init_states[ep]
            refs = reference_boundaries(task, plan, init_state, seed=ep)
            # DP rollout 收集每边界 eef 位置
            env = get_env(task)
            env.seed(1000 + ep * 7 + i)
            env.reset()
            obs = env.set_init_state(init_state)
            ids = [p["skill_id"] for p in plan]
            steps = [p["steps"] for p in plan]
            obs_mean, obs_std, act_mean, act_std = norm

            def norm_obs(o):
                return torch.from_numpy((env_obs(o) - obs_mean) / obs_std).float().to(DEVICE).unsqueeze(0)

            def onehot(k):
                z = np.zeros((1, dp.n_skills), dtype=np.float32)
                z[0, k] = 1.0
                return torch.from_numpy(z).to(DEVICE)
            chunk = dp.sample(norm_obs(obs), onehot(ids[0]), n_steps=16)
            si = 0; cur = 0; step_in_skill = 0
            eef_at_bounds = []
            t = 0
            import sys as _s
            total = min(sum(steps), 400)
            while t < total:
                if t > 0 and step_in_skill >= steps[cur] and cur + 1 < len(ids):
                    cur += 1; step_in_skill = 0
                    eef_at_bounds.append(env_obs(obs)[2:5].copy())
                a = np.clip((chunk[0, si].cpu().numpy() * act_std) + act_mean, -1, 1)
                obs, *_ = env.step(a)
                si += 1; step_in_skill += 1; t += 1
                if si >= 5:
                    chunk = dp.sample(norm_obs(obs), onehot(ids[cur]), n_steps=16)
                    si = 0
            env.close()
            if not eef_at_bounds:
                continue
            done = 0
            for k, (ref, got) in enumerate(zip(refs[:len(eef_at_bounds)],
                                               eef_at_bounds)):
                d = float(np.linalg.norm(ref - got))
                if d < 0.15:
                    done += 1
            task_done.append(done / max(1, len(eef_at_bounds)))
        rows.append(dict(task=task.name[:40],
                         skill_progress=float(np.mean(task_done)) if task_done else 0.0))
    avg = float(np.mean([r["skill_progress"] for r in rows]))
    print(f"[diag] {args.mode}: avg skill progress = {avg:.3f}")
    for r in rows:
        print(f"  {r['task']}: {r['skill_progress']:.2f}")
    with open(os.path.join(DATA_DIR, f"../eval/diag_progress_{args.mode}.json"), "w") as f:
        json.dump(dict(avg=avg, rows=rows), f, indent=2)
    print(f"[diag] saved")


if __name__ == "__main__":
    main()
