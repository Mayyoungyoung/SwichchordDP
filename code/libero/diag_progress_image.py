"""诊断(图像版): 图像 DP rollout 的技能进度 vs 演示边界参考。

与 diag_progress.py 同协议(边界 eef 距离 < 0.15 的技能占比),
但模型为 ImageSkillDP、env 带 agentview 渲染。
对照: 状态 DP(object-state) 技能进度 0.208。
"""
import json
import os
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from swdp.policy_image import ImageSkillDP          # noqa: E402
from eval_libero_image import get_env               # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "../../results/libero/data_replay")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "../../results/libero/models")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def eef(obs):
    return obs["robot0_eef_pos"].copy()


def reference_boundaries(task, plan, init_state, acts, seed=0):
    env = get_env(task)
    env.seed(seed)
    env.reset()
    env.set_init_state(init_state)
    bounds = []
    acc = 0
    for p in plan:
        acc += p["steps"]
        bounds.append(acc)
    refs = []
    t = 0
    obs = None
    for b in bounds:
        while t < b and t < len(acts):
            obs, *_ = env.step(acts[t])
            t += 1
        refs.append(eef(obs) if obs is not None else eef(env.reset()))
    env.close()
    return refs


def main():
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--tasks", default="0,4,9")
    ap.add_argument("--n_episodes", type=int, default=3)
    args = ap.parse_args()
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()["libero_10"]()
    with open(os.path.join(DATA_DIR, "plans.json")) as f:
        plans = json.load(f)
    rows = []
    for i in [int(x) for x in args.tasks.split(",")]:
        task = ts.get_task(i)
        plan = plans[task.name]
        dp, ck = ImageSkillDP.load(
            os.path.join(MODEL_DIR, f"dpi_libero_{task.name}.pt"), DEVICE)
        dp.eval()
        with h5py.File(os.path.join(DATA_DIR, f"{task.name}.h5"), "r") as f:
            init_states = f["init_states"][:]
            acts = f["action"][:]
        pm, ps, am, asd = ck["prop_mean"], ck["prop_std"], ck["act_mean"], ck["act_std"]
        task_done = []
        for ep in range(min(args.n_episodes, len(init_states))):
            init_state = init_states[ep]
            refs = reference_boundaries(task, plan, init_state, acts, seed=ep)
            env = get_env(task)
            env.seed(1000 + ep * 7 + i)
            env.reset()
            obs = env.set_init_state(init_state)
            ids = [p["skill_id"] for p in plan]
            steps = [p["steps"] for p in plan]

            def norm_obs(o):
                img = torch.from_numpy(o["agentview_image"]).to(DEVICE)
                img = img.permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
                prop = np.concatenate([o["robot0_eef_pos"],
                                       o["robot0_gripper_qpos"]])
                prop = torch.from_numpy((prop - pm) / ps).float()
                return (img, prop.unsqueeze(0).to(DEVICE))

            def onehot(k):
                z = np.zeros((1, dp.n_skills), dtype=np.float32)
                z[0, k] = 1.0
                return torch.from_numpy(z).to(DEVICE)

            chunk = dp.sample(norm_obs(obs), onehot(ids[0]), n_steps=16)
            si, cur, step_in_skill = 0, 0, 0
            eef_at_bounds = []
            t = 0
            total = min(sum(steps), 400)
            while t < total:
                if t > 0 and step_in_skill >= steps[cur] and cur + 1 < len(ids):
                    cur += 1
                    step_in_skill = 0
                    eef_at_bounds.append(eef(obs))
                a = np.clip((chunk[0, si].cpu().numpy() * asd) + am, -1, 1)
                obs, *_ = env.step(a)
                si += 1
                step_in_skill += 1
                t += 1
                if si >= 5:
                    chunk = dp.sample(norm_obs(obs), onehot(ids[cur]), n_steps=16)
                    si = 0
            env.close()
            if not eef_at_bounds:
                continue
            done = 0
            for ref, got in zip(refs[:len(eef_at_bounds)], eef_at_bounds):
                if float(np.linalg.norm(ref - got)) < 0.15:
                    done += 1
            task_done.append(done / max(1, len(eef_at_bounds)))
        rows.append(dict(task=task.name[:40],
                         skill_progress=float(np.mean(task_done)) if task_done else 0.0))
        print(f"  {rows[-1]['task']}: {rows[-1]['skill_progress']:.2f}")
    avg = float(np.mean([r["skill_progress"] for r in rows]))
    print(f"[diag-img] avg skill progress = {avg:.3f}")
    with open(os.path.join(DATA_DIR, "../eval/diag_progress_image.json"), "w") as f:
        json.dump(dict(avg=avg, rows=rows), f, indent=2)


if __name__ == "__main__":
    main()
