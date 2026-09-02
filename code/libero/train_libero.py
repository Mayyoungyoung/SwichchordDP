"""LIBERO 技能条件 DP 训练(低维观测, 复用 swdp.SkillDP)。"""
import argparse
import os
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from swdp.policy import SkillDP  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "../../results/libero/data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "../../results/libero/models")


class LiberoLoader:
    def __init__(self, path, horizon, batch, device, n_skills, seed=0):
        with h5py.File(path, "r") as f:
            self.obs = f["obs"][:]
            self.act = f["action"][:]
            self.skill = f["skill"][:]
            self.obs_mean = f["obs_mean"][:]
            self.obs_std = f["obs_std"][:]
            self.act_mean = f["act_mean"][:]
            self.act_std = f["act_std"][:]
        self.n_skills = n_skills
        self.horizon = horizon
        self.batch = batch
        self.device = device
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
    ap.add_argument("--task", default="all")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--n_iter", type=int, default=40000)
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 合并所有任务数据训练一个统一技能 DP(观测维度一致)
    import glob
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.h5")))
    if args.task != "all":
        paths = [p for p in paths if args.task in p]
    if not paths:
        raise RuntimeError("no data files")
    n_skills = 0
    for p in paths:
        with h5py.File(p, "r") as f:
            n_skills = max(n_skills, f.attrs["n_skills"])
    obs_dim, act_dim = 9, 7
    model = SkillDP(act_dim, args.horizon, obs_dim, n_skills, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loaders = [iter(LiberoLoader(p, args.horizon, args.batch // len(paths),
                                 device, n_skills, seed=i)) for i, p in enumerate(paths)]
    losses = []
    for step in range(args.n_iter):
        opt.zero_grad()
        loss = 0.0
        for it in loaders:
            a0, o, s = next(it)
            loss = loss + model.loss(a0, o, s)
        loss = loss / len(loaders)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss))
        if (step + 1) % 5000 == 0:
            print(f"[libero] iter {step+1} loss {np.mean(losses[-500:]):.5f}")
    os.makedirs(MODEL_DIR, exist_ok=True)
    out = os.path.join(MODEL_DIR, f"dp_libero_{args.task}.pt")
    model.save(out)
    print(f"[libero] saved {out}")


if __name__ == "__main__":
    main()
