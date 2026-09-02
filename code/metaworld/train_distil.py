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
    ap.add_argument("--n_iter", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher = SkillDP.load(os.path.join(MODEL_DIR, f"dp_{args.scene}.pt"), device)
    teacher.eval()
    data_path = os.path.join(DATA_DIR, f"{args.scene}.h5")
    with h5py.File(data_path, "r") as f:
        n_skills = f.attrs["n_skills"]
    loader = iter(DemoLoader(data_path, teacher.horizon, args.batch, device, seed=7))
    student = distill(teacher, n_iter=args.n_iter, batch=args.batch,
                      loader=loader,
                      save_path=os.path.join(MODEL_DIR, f"dp_{args.scene}_cd.pt"))
    print("[distil] done")


if __name__ == "__main__":
    main()
