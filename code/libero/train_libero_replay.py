"""LIBERO 技能条件 DP 训练(replay 数据, env 观测对齐)。"""
import argparse
import glob
import os
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from swdp.policy import SkillDP  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "../../results/libero/data_replay")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "../../results/libero/models")


class ReplayLoader:
    def __init__(self, paths, horizon, batch, device, n_skills, seed=0):
        obs, act, skill = [], [], []
        self.obs_mean, self.obs_std = None, None
        self.act_mean, self.act_std = None, None
        for p in paths:
            with h5py.File(p, "r") as f:
                obs.append(f["obs"][:])
                act.append(f["action"][:])
                skill.append(f["skill"][:])
        self.obs = np.concatenate(obs, 0)
        self.act = np.concatenate(act, 0)
        self.skill = np.concatenate(skill, 0)
        self.obs_mean = self.obs.mean(0)
        self.obs_std = self.obs.std(0) + 1e-6
        self.act_mean = self.act.mean(0)
        self.act_std = self.act.std(0) + 1e-6
        self.horizon = horizon
        self.batch = batch
        self.device = device
        self.n_skills = n_skills
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        while True:
            idx = self.rng.integers(0, len(self.obs) - self.horizon,
                                    size=self.batch)
            good = np.array([(self.skill[i:i + self.horizon] == self.skill[i]).all()
                             for i in idx])
            idx = idx[good]
            if len(idx) < 8:
                continue
            o = (self.obs[idx] - self.obs_mean) / self.obs_std
            a = np.stack([self.act[i:i + self.horizon] for i in idx], 0)
            a = (a - self.act_mean) / self.act_std
            s = np.eye(self.n_skills)[self.skill[idx]]
            yield (torch.from_numpy(a).float().to(self.device),
                   torch.from_numpy(o).float().to(self.device),
                   torch.from_numpy(s).float().to(self.device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_iter", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--task", default="", help="空 = 全任务全局模型; 否则单任务模型")
    ap.add_argument("--tau_power", type=float, default=2.0,
                    help="低噪声偏置采样幂次(>1 向 tau=1 集中)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.h5")))
    if args.task:
        paths = [p for p in paths if args.task in p]
        assert paths, f"task {args.task} not found"
        n_skills = 8
        tag = args.task
    else:
        n_skills = 8
        tag = "replay"
    loader = ReplayLoader(paths, args.horizon, args.batch, device, n_skills, seed=7)
    model = SkillDP(7, args.horizon, 9, n_skills, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)
    it = iter(loader)
    losses = []
    for step in range(args.n_iter):
        a0, o, s = next(it)
        loss = model.loss(a0, o, s, tau_power=args.tau_power)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss))
        if (step + 1) % 5000 == 0:
            print(f"[libero-train] {tag} iter {step+1} loss {np.mean(losses[-500:]):.5f}")
    os.makedirs(MODEL_DIR, exist_ok=True)
    out = os.path.join(MODEL_DIR, f"dp_libero_{tag}.pt")
    model.save(out)
    print(f"[libero-train] saved {out}")


if __name__ == "__main__":
    main()
