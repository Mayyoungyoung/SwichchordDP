"""Downstream-Compatible Skill Learning 加权微调（第一轮: binary outcome 加权）。

从预训练 dp_pick-place-v3.pt 微调。权重方案:
- reweight(y):  w = 1 + lam * y          (方法1: outcome-weighted 上界)
- reweight(q):  w = 1 + lam * quality    (基线3: grasp 自身质量加权)
- uniform:      w = 1                    (等权微调对照, 隔离微调本身效应)
- reweight(fb): w = 1 + lam * F_B(s+)    (方法2: F_B 连续加权, --fb_pt 指向
  F_B checkpoint, 权重在收集数据的终态 obs 上现算)

加权作用于轨迹内每个 step(trajectory-level 权重广播到 step-level)。
只微调低 epoch(默认 8000 iter), 保持其他技能条件数据不参与(仅 grasp 数据)。

用法: python dc_train.py --weight {uniform,outcome,quality,fb} --lam 2.0 \
      [--fb_pt results/metaworld/models/fb_pick-place-v3_v2.pt]
输出: results/metaworld/models/dc_{weight}_l{lam}.pt
"""
import argparse
import os

import h5py
import numpy as np
import torch

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from swdp.policy import SkillDP  # noqa: E402
from swdp.success_model import featurize, load as fb_load  # noqa: E402
from skills import parse_pp  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "../../results/metaworld/models")
EVAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "../../results/metaworld/eval")


class WeightedLoader:
    """无限迭代的加权 chunk loader(仅 grasp 轨迹)。"""

    def __init__(self, path, horizon, batch, device, weight_fn, seed=0):
        with h5py.File(path, "r") as f:
            self.obs = f["obs"][:]
            self.act = f["action"][:]
            self.skill = f["skill"][:]
            self.traj = f["traj_id"][:]
            self.n_skills = f.attrs["n_skills"]
            w_traj = weight_fn(f)          # per-trajectory 权重
            self.w = w_traj[self.traj]     # 广播到 per-step
            self.obs_mean = f["obs_mean"][:]
            self.obs_std = f["obs_std"][:]
            self.act_mean = f["act_mean"][:]
            self.act_std = f["act_std"][:]
        self.horizon = horizon
        self.batch = batch
        self.device = device
        self.rng = np.random.default_rng(seed)
        print(f"[train] loader: {len(self.obs)} steps, "
              f"w range=[{self.w.min():.2f},{self.w.max():.2f}]")

    def __iter__(self):
        while True:
            idx = self.rng.integers(0, len(self.obs) - self.horizon,
                                    size=self.batch)
            # chunk 不得跨越轨迹边界(skill 与 traj_id 双检查)。跨轨迹拼接
            # 会把 A 轨迹尾与 B 轨迹头混成时间不连续的伪样本。
            good = np.array([(self.skill[i:i + self.horizon]
                              == self.skill[i]).all()
                             and (self.traj[i:i + self.horizon]
                                  == self.traj[i]).all() for i in idx])
            idx = idx[good]
            if len(idx) < 8:
                continue
            o = (self.obs[idx] - self.obs_mean) / self.obs_std
            a = np.stack([self.act[i:i + self.horizon] for i in idx], 0)
            a = (a - self.act_mean) / self.act_std
            s = np.eye(self.n_skills)[self.skill[idx]]
            w = self.w[idx].astype(np.float32)
            yield (torch.from_numpy(a).float().to(self.device),
                   torch.from_numpy(o).float().to(self.device),
                   torch.from_numpy(s).float().to(self.device),
                   torch.from_numpy(w).float().to(self.device))


def _feat_from_obs(o):
    """终态 obs -> F_B 19 维几何特征(success_model.featurize 接口)。"""
    p = parse_pp(o)
    return featurize(p["hand"], float(p["grip"]), p["puck"], p["goal"])


def make_weight_fn(kind, lam, fb_pt):
    def uniform(f):
        return np.ones(len(f["y"]))

    def outcome(f):
        return 1.0 + lam * f["y"][:]

    def quality(f):
        return 1.0 + lam * f["quality"][:]

    def fb(f):
        model, skill_names = fb_load(fb_pt, DEVICE)
        sid_lift = skill_names.index("lift")
        obs_t = f["obs_t"][:]
        feats = np.stack([_feat_from_obs(o) for o in obs_t])
        onehot = np.zeros((len(obs_t), len(skill_names)), dtype=np.float32)
        onehot[:, sid_lift] = 1.0
        p = model.predict(feats, onehot)
        return 1.0 + lam * p

    return {"uniform": uniform, "outcome": outcome,
            "quality": quality, "fb": fb}[kind]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", default="outcome",
                    choices=["uniform", "outcome", "quality", "fb"])
    ap.add_argument("--lam", type=float, default=2.0)
    ap.add_argument("--data", default=os.path.join(
        EVAL_DIR, "dc_grasp_lift.h5"))
    ap.add_argument("--n_iter", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--fb_pt", default=os.path.join(
        MODEL_DIR, "fb_pick-place-v3_v2.pt"),
        help="F_B checkpoint(自然分布训练版), --weight fb 时使用")
    args = ap.parse_args()

    torch.manual_seed(0)
    # 从预训练权重微调
    model = SkillDP.load(os.path.join(MODEL_DIR, "dp_pick-place-v3.pt"),
                         DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loader = WeightedLoader(args.data, args.horizon, args.batch, DEVICE,
                            make_weight_fn(args.weight, args.lam, args.fb_pt))
    it = iter(loader)
    losses = []
    for step in range(args.n_iter):
        a0, o, s, w = next(it)
        eps = torch.randn_like(a0)
        tau = torch.rand(a0.shape[0], 1, device=DEVICE)
        from swdp.nets import ALPHA, SIGMA
        a_noisy = ALPHA(tau).unsqueeze(-1) * a0 + SIGMA(tau).unsqueeze(-1) * eps
        eps_hat = model.net(a_noisy, tau, o, s)
        # 加权 MSE(权重在 batch 维度)
        per = ((eps_hat - eps) ** 2).mean(dim=(1, 2))
        loss = (w * per).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss))
        if (step + 1) % 2000 == 0:
            print(f"[train] {args.weight} lam={args.lam} iter {step+1} "
                  f"loss {np.mean(losses[-500:]):.5f}")
    out = os.path.join(MODEL_DIR, f"dc_{args.weight}_l{args.lam}.pt")
    model.save(out)
    print(f"[train] saved {out}")


if __name__ == "__main__":
    main()
