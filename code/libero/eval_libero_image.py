"""LIBERO 图像 DP 在线 rollout 评测(EGL 渲染 agentview, BDDL 成功判定)。

协议与 eval_libero_online.py 完全一致(plans.json 规范计划 + replay init_states
+ receding-horizon + 边界 ChordCompose), 仅观测换为 (agentview_image, prop):
用于量化图像观测对基础执行器的提升(对照: 状态 DP 技能进度 0.208 / e2e 0)。
"""
import argparse
import json
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch  # noqa: E402
from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs.env_wrapper import ControlEnv  # noqa: E402

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from swdp.policy_image import ImageSkillDP  # noqa: E402
from swdp import chord_compose as cc       # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "../../results/libero/data_replay")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "../../results/libero/models")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HORIZON = 10


def get_env(task):
    bddl_path = os.path.join(get_libero_path("bddl_files"),
                             task.problem_folder, task.bddl_file)
    return ControlEnv(bddl_file_name=bddl_path, has_renderer=False,
                      has_offscreen_renderer=True, use_camera_obs=True,
                      camera_names=["agentview"], camera_heights=[128],
                      camera_widths=[128], control_freq=20)


@torch.no_grad()
def rollout(dp, ck, task, plan, init_state, mode, tau=0.9, delta=0.15,
            lam=0.3, use_mask=True, use_proj=True, mask_width=4,
            n_ddim=16, resample=5, seed=0, max_steps=400):
    prop_mean, prop_std = ck["prop_mean"], ck["prop_std"]
    act_mean, act_std = ck["act_mean"], ck["act_std"]
    env = get_env(task)
    env.seed(1000 + seed)
    env.reset()
    obs = env.set_init_state(init_state)
    skill_ids = [p["skill_id"] for p in plan]
    skill_steps = [p["steps"] for p in plan]

    def norm_obs(o):
        img = torch.from_numpy(o["agentview_image"]).to(DEVICE)
        img = img.permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
        prop = np.concatenate([o["robot0_eef_pos"],
                               o["robot0_gripper_qpos"]])
        prop = torch.from_numpy((prop - prop_mean) / prop_std).float()
        return (img, prop.unsqueeze(0).to(DEVICE))

    def onehot(i):
        z = np.zeros((1, dp.n_skills), dtype=np.float32)
        z[0, i] = 1.0
        return torch.from_numpy(z).to(DEVICE)

    chunk = dp.sample(norm_obs(obs), onehot(skill_ids[0]), n_steps=n_ddim)
    step_in_chunk, cur, step_in_skill = 0, 0, 0
    energies, exec_actions = [], []
    total = min(sum(skill_steps), max_steps)
    for t in range(total):
        if t > 0 and step_in_skill >= skill_steps[cur] and cur + 1 < len(skill_ids):
            cur += 1
            step_in_skill = 0
            o_t = norm_obs(obs)
            anchor = dp.sample(o_t, onehot(skill_ids[cur - 1]), n_steps=n_ddim)
            s_from, s_to = onehot(skill_ids[cur - 1]), onehot(skill_ids[cur])
            mask = cc.temporal_mask(HORIZON, 0, mask_width, DEVICE) if use_mask else None
            a_new, info = cc.switch(dp, o_t, anchor, s_from, s_to, tau, delta,
                                    lam, 1, mode, mask, use_proj, seed=seed + t)
            energies.append(info["energy"])
            chunk = a_new
            step_in_chunk = 0
        a_raw = np.clip((chunk[0, step_in_chunk].cpu().numpy() * act_std) + act_mean,
                        -1, 1)
        exec_actions.append(a_raw)
        obs, _, _, _ = env.step(a_raw)
        step_in_chunk += 1
        step_in_skill += 1
        if step_in_chunk >= resample or step_in_chunk >= HORIZON:
            chunk = dp.sample(norm_obs(obs), onehot(skill_ids[cur]), n_steps=n_ddim)
            step_in_chunk = 0
    success = float(env.check_success())
    exec_actions = np.array(exec_actions)
    _e = np.sum(np.diff(exec_actions, axis=0) ** 2, axis=1)
    env.close()
    return dict(success=success, energy=float(np.median(_e)) if len(_e) else 0.0,
                boundary_energy=energies)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="chord",
                    choices=["chord", "naive", "eff_shift"])
    ap.add_argument("--tasks", default="0,4,9")
    ap.add_argument("--n_episodes", type=int, default=3)
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--n_ddim", type=int, default=16)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    with open(os.path.join(DATA_DIR, "plans.json")) as f:
        plans = json.load(f)
    ts = benchmark.get_benchmark_dict()["libero_10"]()
    results = []
    for i in [int(x) for x in args.tasks.split(",")]:
        task = ts.get_task(i)
        dp, ck = ImageSkillDP.load(
            os.path.join(MODEL_DIR, f"dpi_libero_{task.name}.pt"), DEVICE)
        dp.eval()
        plan = plans[task.name]
        with __import__("h5py").File(
                os.path.join(DATA_DIR, f"{task.name}.h5"), "r") as f:
            init_states = f["init_states"][:]
        succ = []
        for ep in range(args.n_episodes):
            t0 = time.time()
            r = rollout(dp, ck, task, plan, init_states[ep % len(init_states)],
                        args.mode, lam=args.lam, n_ddim=args.n_ddim,
                        seed=ep * 7 + i)
            r["latency_s"] = time.time() - t0
            succ.append(r)
            print(f"[img-online] task{i} {args.mode} ep{ep} "
                  f"success={r['success']} energy={r['energy']:.3f}", flush=True)
        rate = float(np.mean([s["success"] for s in succ]))
        results.append(dict(task=task.name, rate=rate, episodes=succ))
        print(f"[img-online] task{i} rate = {rate:.2f}", flush=True)
    avg = float(np.mean([r["rate"] for r in results]))
    print(f"[img-online] {args.mode}: avg = {avg:.3f}")
    out_dir = os.path.join(DATA_DIR, "../eval")
    os.makedirs(out_dir, exist_ok=True)
    tag = args.out or f"libero_image_{args.mode}"
    with open(os.path.join(out_dir, f"{tag}.json"), "w") as f:
        json.dump(dict(args=vars(args), avg=avg, results=results), f, indent=2,
                  default=str)
    print(f"[img-online] saved {tag}.json")


if __name__ == "__main__":
    main()
