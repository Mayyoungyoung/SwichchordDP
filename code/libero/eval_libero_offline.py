"""LIBERO 离线拼接评测(增信实验)。

在留出演示的交接边界处:
1. 取边界前 H 步真实动作块作为锚点 a_anchor, 边界处观测 obs;
2. 用 ChordCompose(mode: chord/naive/eff_shift)把技能 s_from 传输到 s_to;
3. 指标: 传输块 vs 边界后真实动作的 MSE(拼接保真)、场能量、边界 jerk、
   以及 ‖∂eps_hat/∂a‖ Lipschitz 估计(理论闭环)。
"""
import argparse
import glob
import json
import os
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from swdp.policy import SkillDP  # noqa: E402
from swdp import chord_compose as cc  # noqa: E402
from swdp.feasibility import prox_feasible  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "../../results/libero/data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "../../results/libero/models")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HORIZON = 10


def est_lipschitz(dp, a, obs, s, n_pert=8, eps=1e-2):
    max_ratio = 0.0
    for _ in range(n_pert):
        delta = torch.randn_like(a)
        delta = delta / (delta.norm(dim=(1, 2), keepdim=True) + 1e-8)
        a2 = a + eps * delta
        e1 = dp.Q(a, torch.full((a.shape[0], 1), 0.9, device=DEVICE), obs, s)
        e2 = dp.Q(a2, torch.full((a.shape[0], 1), 0.9, device=DEVICE), obs, s)
        ratio = ((e2 - e1).norm(dim=(1, 2)) /
                 (eps * delta.norm(dim=(1, 2)) + 1e-12))
        max_ratio = max(max_ratio, float(ratio.max()))
    return max_ratio


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all")
    ap.add_argument("--mode", default="chord",
                    choices=["chord", "naive", "eff_shift"])
    ap.add_argument("--tau", type=float, default=0.9)
    ap.add_argument("--delta", type=float, default=0.15)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--use_proj", action="store_true")
    ap.add_argument("--n_boundaries", type=int, default=100)
    args = ap.parse_args()

    dp = SkillDP.load(os.path.join(
        MODEL_DIR, f"dp_libero_{args.task}.pt"), DEVICE)
    dp.eval()
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.h5")))
    if args.task != "all":
        paths = [p for p in paths if args.task in p]

    rows = []
    n_total = 0
    for p in paths:
        with h5py.File(p, "r") as f:
            obs = f["obs"][:]; act = f["action"][:]
            skill = f["skill"][:]
            n_skills = dp.n_skills  # 与训练时全局技能数一致
            obs_mean = f["obs_mean"][:]; obs_std = f["obs_std"][:]
            act_mean = f["act_mean"][:]; act_std = f["act_std"][:]
        # 边界索引
        bounds = np.where(skill[1:] != skill[:-1])[0] + 1
        rng = np.random.default_rng(0)
        sel = rng.choice(bounds, size=min(args.n_boundaries, len(bounds)),
                         replace=False)
        for b in sel:
            if b < HORIZON or b + HORIZON > len(skill):
                continue
            a_anchor = torch.from_numpy(
                (act[b - HORIZON:b] - act_mean) / act_std).float().to(DEVICE).unsqueeze(0)
            o = torch.from_numpy(
                (obs[b] - obs_mean) / obs_std).float().to(DEVICE).unsqueeze(0)
            s_from = torch.zeros(1, n_skills, device=DEVICE)
            s_from[0, int(skill[b - 1])] = 1
            s_to = torch.zeros(1, n_skills, device=DEVICE)
            s_to[0, int(skill[b])] = 1
            mask = cc.temporal_mask(HORIZON, 0, 4, DEVICE)
            a_new, info = cc.switch(dp, o, a_anchor, s_from, s_to, args.tau,
                                    args.delta, args.lam, 1, args.mode, mask,
                                    args.use_proj, seed=b)
            # 保真度: 传输块 vs 真实后续动作(归一化空间)
            gt = torch.from_numpy(
                (act[b:b + HORIZON] - act_mean) / act_std).float().to(DEVICE).unsqueeze(0)
            mse = float(((a_new - gt) ** 2).mean())
            # 边界 jerk: 真实前一步动作与传输块第一步的差
            a_prev = torch.from_numpy(
                (act[b - 1] - act_mean) / act_std).float().to(DEVICE)
            jerk = float((a_new[0, 0] - a_prev).abs().max())
            rows.append(dict(mse=mse, jerk=jerk, energy=info["energy"],
                             L=est_lipschitz(dp, a_anchor, o, s_from),
                             skill_from=int(skill[b - 1]),
                             skill_to=int(skill[b])))
            n_total += 1
    mse = np.mean([r["mse"] for r in rows])
    jerk = np.mean([r["jerk"] for r in rows])
    energy = np.mean([r["energy"] for r in rows])
    lips = np.mean([r["L"] for r in rows])
    print(f"[libero] {args.mode} (proj={args.use_proj}): "
          f"mse={mse:.4f} jerk={jerk:.4f} energy={energy:.4f} lips={lips:.3f} "
          f"n={n_total}")
    out_dir = os.path.join(DATA_DIR, "../eval")
    os.makedirs(out_dir, exist_ok=True)
    tag = f"libero_{args.task}_{args.mode}"
    if args.use_proj:
        tag += "_proj"
    with open(os.path.join(out_dir, f"{tag}.json"), "w") as f:
        json.dump(dict(args=vars(args), mse=mse, jerk=jerk, energy=energy,
                       lips=lips, n=n_total), f, indent=2)
    print(f"[libero] saved {out_dir}/{tag}.json")


if __name__ == "__main__":
    main()
