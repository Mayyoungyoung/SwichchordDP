"""LIBERO-Long 在线 rollout 评测(BDDL 成功判定, 对标 TAPT/HELM 基准协议)。

流程(Plan-and-Compose 低层侧):
1. 每个任务使用规范计划(plans.json, 由 demo_0 分段得到)作为 oracle 高层序列;
2. 每回合从 benchmark init_states 取初始状态(官方有效初始状态);
3. 低层: 冻结技能条件 DP(replay 数据训练)receding-horizon 执行,
   技能边界处 ChordCompose 交接(mode: chord/naive/eff_shift/energy);
4. 成功判定: env.check_success()(BDDL 目标谓词);
5. 指标: 任务成功率、平均成功率、NFE、轨迹能量、边界场能量、Lipschitz 估计。
"""
import argparse
import glob
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
from swdp import chord_compose as cc  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "../../results/libero/data_replay")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "../../results/libero/models")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HORIZON = 10
OBS_DIM, ACT_DIM = 9, 7


def get_env(task):
    bddl_path = os.path.join(get_libero_path("bddl_files"),
                             task.problem_folder, task.bddl_file)
    return ControlEnv(bddl_file_name=bddl_path, has_renderer=False,
                      has_offscreen_renderer=False, use_camera_obs=False,
                      camera_names=[], control_freq=20)


def env_obs(obs):
    return np.concatenate([obs["robot0_gripper_qpos"], obs["robot0_eef_pos"],
                           obs["robot0_eef_quat"]]).astype(np.float32)


@torch.no_grad()
def est_lipschitz(dp, a, obs_t, s, n_pert=6, eps=1e-2):
    max_ratio = 0.0
    for _ in range(n_pert):
        delta = torch.randn_like(a)
        delta = delta / (delta.norm(dim=(1, 2), keepdim=True) + 1e-8)
        a2 = a + eps * delta
        e1 = dp.Q(a, torch.full((a.shape[0], 1), 0.9, device=DEVICE), obs_t, s)
        e2 = dp.Q(a2, torch.full((a.shape[0], 1), 0.9, device=DEVICE), obs_t, s)
        ratio = ((e2 - e1).norm(dim=(1, 2)) /
                 (eps * delta.norm(dim=(1, 2)) + 1e-12))
        max_ratio = max(max_ratio, float(ratio.max()))
    return max_ratio


@torch.no_grad()
def rollout(dp, task, plan, init_state, norm, mode, tau=0.9, delta=0.15,
            lam=0.3, n_noise=1, use_mask=True, use_proj=True, mask_width=4,
            n_ddim=8, resample=5, seed=0, max_steps=400):
    obs_mean, obs_std, act_mean, act_std = norm
    env = get_env(task)
    env.seed(1000 + seed)
    env.reset()
    obs = env.set_init_state(init_state)
    skill_ids = [p["skill_id"] for p in plan]
    skill_steps = [p["steps"] for p in plan]
    n_skills = dp.n_skills

    def norm_obs(o):
        return torch.from_numpy((env_obs(o) - obs_mean) / obs_std).float().to(DEVICE).unsqueeze(0)

    def onehot(i):
        z = np.zeros((1, n_skills), dtype=np.float32)
        z[0, i] = 1.0
        return torch.from_numpy(z).to(DEVICE)

    def sample_chunk(o, s_id):
        return dp.sample(o, onehot(s_id), n_steps=n_ddim, seed=None)

    nfe = 0
    chunk = sample_chunk(norm_obs(obs), skill_ids[0])
    nfe += n_ddim
    step_in_chunk = 0
    cur = 0
    step_in_skill = 0
    energies, lips_vals, exec_actions = [], [], []
    total = min(sum(skill_steps), max_steps)

    for t in range(total):
        if t > 0 and step_in_skill >= skill_steps[cur] and cur + 1 < len(skill_ids):
            cur += 1
            step_in_skill = 0
            o_t = norm_obs(obs)
            anchor = sample_chunk(o_t, skill_ids[cur - 1])
            nfe += n_ddim
            s_from = onehot(skill_ids[cur - 1])
            s_to = onehot(skill_ids[cur])
            lips_vals.append(dict(
                pair=f"{skill_ids[cur-1]}->{skill_ids[cur]}",
                L=max(est_lipschitz(dp, anchor, o_t, s_from),
                      est_lipschitz(dp, anchor, o_t, s_to))))
            mask = cc.temporal_mask(HORIZON, 0, mask_width, DEVICE) if use_mask else None
            a_new, info = cc.switch(dp, o_t, anchor, s_from, s_to, tau, delta,
                                    lam, n_noise, mode, mask, use_proj,
                                    seed=seed + t)
            nfe += (2 if mode == "chord" else 1) * n_noise
            energies.append(info["energy"])
            chunk = a_new
            step_in_chunk = 0

        a_raw = np.clip((chunk[0, step_in_chunk].cpu().numpy() * act_std) + act_mean, -1, 1)
        exec_actions.append(a_raw)
        obs, _, _, _ = env.step(a_raw)
        step_in_chunk += 1
        step_in_skill += 1
        if step_in_chunk >= resample or step_in_chunk >= HORIZON:
            chunk = sample_chunk(norm_obs(obs), skill_ids[cur])
            nfe += n_ddim
            step_in_chunk = 0

    success = float(env.check_success())
    exec_actions = np.array(exec_actions)
    _e = np.sum(np.diff(exec_actions, axis=0) ** 2, axis=1)
    energy = float(np.median(_e)) if len(_e) else 0.0
    env.close()
    return dict(success=success, energy=energy, nfe=nfe,
                boundary_energy=energies, lips=lips_vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="chord",
                    choices=["chord", "naive", "eff_shift", "energy"])
    ap.add_argument("--n_episodes", type=int, default=5)
    ap.add_argument("--n_tasks", type=int, default=10)
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--n_ddim", type=int, default=8)
    ap.add_argument("--use_mask", action="store_true", default=True)
    ap.add_argument("--use_proj", action="store_true", default=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    with open(os.path.join(DATA_DIR, "plans.json")) as f:
        plans = json.load(f)
    bd = benchmark.get_benchmark_dict()
    ts = bd["libero_10"]()
    results = []
    per_task = {}
    for i in range(min(args.n_tasks, ts.get_num_tasks())):
        task = ts.get_task(i)
        dp = SkillDP.load(os.path.join(MODEL_DIR, f"dp_libero_{task.name}.pt"), DEVICE)
        dp.eval()
        plan = plans[task.name]
        with h5py.File(os.path.join(DATA_DIR, f"{task.name}.h5"), "r") as f:
            norm = (f["obs_mean"][:], f["obs_std"][:],
                    f["act_mean"][:], f["act_std"][:])
        init_states = ts.get_task_init_states(i)
        succ = []
        for ep in range(args.n_episodes):
            t0 = time.time()
            r = rollout(dp, task, plan, init_states[ep % len(init_states)],
                        norm, args.mode, lam=args.lam, n_ddim=args.n_ddim,
                        use_mask=args.use_mask, use_proj=args.use_proj,
                        seed=ep * 7 + i)
            r["latency_s"] = time.time() - t0
            succ.append(r)
            print(f"[libero-online] task{i} {args.mode} ep{ep} "
                  f"success={r['success']} energy={r['energy']:.3f} "
                  f"nfe={r['nfe']}")
        rate = float(np.mean([s["success"] for s in succ]))
        per_task[task.name] = dict(rate=rate, episodes=succ)
        results.append(dict(task=task.name, rate=rate))
    avg = float(np.mean([r["rate"] for r in results]))
    print(f"[libero-online] {args.mode}: avg success = {avg:.3f} over "
          f"{len(results)} tasks x {args.n_episodes} eps")
    out_dir = os.path.join(DATA_DIR, "../eval")
    os.makedirs(out_dir, exist_ok=True)
    tag = args.out or f"libero_online_{args.mode}"
    with open(os.path.join(out_dir, f"{tag}.json"), "w") as f:
        json.dump(dict(args=vars(args), avg=avg, results=results,
                       per_task=per_task), f, indent=2, default=str)
    print(f"[libero-online] saved {out_dir}/{tag}.json")


if __name__ == "__main__":
    main()
