"""训练技能条件扩散策略(场景内所有技能共享一个 DP, FiLM 技能条件)。"""
import argparse
import os

import h5py
import numpy as np
import torch

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from swdp.policy import SkillDP  # noqa: E402


class DemoLoader:
    """无限迭代的数据 loader。"""

    def __init__(self, path: str, horizon: int, batch: int, device: str,
                 seed: int = 0):
        with h5py.File(path, "r") as f:
            self.obs = f["obs"][:]
            self.act = f["action"][:]
            self.skill = f["skill"][:]
            self.n_skills = f.attrs["n_skills"]
            self.obs_mean = f["obs_mean"][:]; self.obs_std = f["obs_std"][:]
            self.act_mean = f["act_mean"][:]; self.act_std = f["act_std"][:]
        self.horizon = horizon
        self.batch = batch
        self.device = device
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        while True:
            # 随机起点, 保证动作块不跨技能边界
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
    ap.add_argument("--scene", default="pick-place-v3")
    ap.add_argument("--data_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../results/metaworld/data"))
    ap.add_argument("--out_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../results/metaworld/models"))
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--n_iter", type=int, default=60000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = os.path.join(args.data_dir, f"{args.scene}.h5")
    with h5py.File(data_path, "r") as f:
        n_skills = f.attrs["n_skills"]
    obs_dim = 39
    act_dim = 4

    model = SkillDP(act_dim, args.horizon, obs_dim, n_skills,
                    hidden=args.hidden, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loader = DemoLoader(data_path, args.horizon, args.batch, device,
                        seed=args.seed)
    it = iter(loader)
    losses = []
    for step in range(args.n_iter):
        a0, o, s = next(it)
        loss = model.loss(a0, o, s)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss))
        if (step + 1) % 5000 == 0:
            print(f"[train] {scene_label(args.scene)} iter {step+1} "
                  f"loss {np.mean(losses[-500:]):.5f}")
    os.makedirs(args.out_dir, exist_ok=True)
    save_path = os.path.join(args.out_dir, f"dp_{args.scene}.pt")
    model.save(save_path)
    print(f"[train] saved {save_path}")


def scene_label(scene):
    return scene


if __name__ == "__main__":
    main()
