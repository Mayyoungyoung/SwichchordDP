"""一致性蒸馏: 把技能条件 DP 蒸馏为 1/2/4 步策略(验证少步模型上的单步价值)。"""
import argparse
import os
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from swdp.policy import SkillDP  # noqa: E402
from swdp.distil import distill  # noqa: E402
from train_dp import DemoLoader  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "../../results/metaworld/data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "../../results/metaworld/models")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="pick-place-v3")
    ap.add_argument("--n_iter", type=int, default=50000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--n_levels", type=int, default=12)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher = SkillDP.load(os.path.join(MODEL_DIR, f"dp_{args.scene}.pt"), device)
    teacher.eval()
    data_path = os.path.join(DATA_DIR, f"{args.scene}.h5")
    with h5py.File(data_path, "r") as f:
        n_skills = f.attrs["n_skills"]
    loader = iter(DemoLoader(data_path, teacher.horizon, args.batch, device, seed=7))
    save_path = os.path.join(MODEL_DIR, f"dp_{args.scene}_cd.pt")
    student = distill(teacher, n_iter=args.n_iter, batch=args.batch,
                      loader=loader, n_levels=args.n_levels,
                      ema_decay=args.ema_decay, save_path=save_path)
    # ---- 蒸馏后验证: 学生 1/4 步 vs 教师 24 步的动作 MSE(同一批验证状态) ----
    with torch.no_grad():
        v_a0, v_obs, v_s = next(loader)
        v_a0 = v_a0.to(device); v_obs = v_obs.to(device); v_s = v_s.to(device)
        teacher_out = teacher.sample(v_obs, v_s, n_steps=24, seed=0)
        for n in [1, 2, 4]:
            stu_out = student.sample(v_obs, v_s, n_steps=n, seed=0)
            mse = float(((stu_out - teacher_out) ** 2).mean())
            print(f"[distil-valid] student {n}-step vs teacher 24-step MSE = {mse:.5f}")
    print("[distil] done")


if __name__ == "__main__":
    main()
