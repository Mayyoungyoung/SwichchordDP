"""LIBERO 图像技能条件 DP 训练(原始数据集 agentview_rgb, 无需渲染)。

- 数据: 原始 h5 的 obs/agentview_rgb(128x128 uint8) + ee_pos/gripper_states + actions
- 技能标签: 与 replay_dataset.segment 相同启发式(夹爪切换 + 位移谷值), 直接在
  数据集 ee_pos 上计算
- 归一化: prop/act 统计量存入 ckpt(评测时复用); 图像 /255 -> [-1,1]
- 模型: ImageSkillDP(与 SkillDP 同接口, 供 ChordCompose 免训组合)
"""
import argparse
import os
import sys
import time

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from swdp.policy_image import ImageSkillDP  # noqa: E402

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "../../results/libero/models")
N_SKILL_CAP = 8


def segment_ds(ee_pos, actions):
    """与 replay_dataset.segment 一致(ee_pos 单独传入, 索引 0:3)。"""
    T = len(actions)
    closed = actions[:, -1] > 0.0
    boundaries = [0]
    for t in range(1, T):
        if closed[t] != closed[t - 1]:
            boundaries.append(t)
    disp = np.linalg.norm(np.diff(ee_pos, axis=0), axis=1)
    disp = np.concatenate([[0.0], disp])
    final = []
    for i in range(len(boundaries) - 1):
        lo, hi = boundaries[i], boundaries[i + 1]
        final.append(lo)
        if hi - lo > 80 and hi - lo > 2:
            mid = lo + 1 + int(np.argmin(disp[lo + 1:hi - 1]))
            if hi - mid > 20 and mid - lo > 20:
                final.append(mid)
    final.append(T)
    labels = np.zeros(T, dtype=np.int64)
    for i in range(len(final) - 1):
        labels[final[i]:final[i + 1]] = i % N_SKILL_CAP
    return labels


def load_task(demo_path, n_demos):
    imgs, props, acts, skills = [], [], [], []
    with h5py.File(demo_path, "r") as h:
        keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[1]))
        for dk in keys[:n_demos]:
            g = h[f"data/{dk}"]
            imgs.append(g["obs/agentview_rgb"][:])           # [T,128,128,3] uint8
            props.append(np.concatenate(
                [g["obs/ee_pos"][:], g["obs/gripper_states"][:]], 1))
            acts.append(g["actions"][:].astype(np.float32))
            skills.append(segment_ds(g["obs/ee_pos"][:], g["actions"][:]))
    return (np.concatenate(imgs, 0), np.concatenate(props, 0),
            np.concatenate(acts, 0), np.concatenate(skills, 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="0", help="逗号分隔的任务索引")
    ap.add_argument("--n_demos", type=int, default=30)
    ap.add_argument("--n_iter", type=int, default=25000)
    ap.add_argument("--batch", type=int, default=80)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--tau_power", type=float, default=2.0)
    ap.add_argument("--aug", type=int, default=0,
                    help="随机平移增广幅度(px), 0=关")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from libero.libero import benchmark, get_libero_path
    ts = benchmark.get_benchmark_dict()["libero_10"]()
    for tid in [int(x) for x in args.tasks.split(",")]:
        task = ts.get_task(tid)
        t0 = time.time()
        imgs, props, acts, skills = load_task(
            os.path.join(get_libero_path("datasets"),
                         ts.get_task_demonstration(tid)), args.n_demos)
        prop_mean, prop_std = props.mean(0), props.std(0) + 1e-6
        act_mean, act_std = acts.mean(0), acts.std(0) + 1e-6
        rng = np.random.default_rng(7 + tid)
        n = len(acts)
        model = ImageSkillDP(7, args.horizon, props.shape[1], N_SKILL_CAP,
                             device=device)
        opt = torch.optim.Adam(model.parameters(), lr=2e-4)
        losses = []
        for it in range(args.n_iter):
            idx = rng.integers(0, n - args.horizon, size=args.batch * 2)
            good = np.array([(skills[i:i + args.horizon] == skills[i]).all()
                             for i in idx])
            idx = idx[good][:args.batch]
            img = torch.from_numpy(imgs[idx]).to(device)
            img = img.permute(0, 3, 1, 2).float() / 127.5 - 1.0
            if args.aug > 0:
                px = args.aug
                dx = int(torch.randint(-px, px + 1, (1,)).item())
                dy = int(torch.randint(-px, px + 1, (1,)).item())
                img = torch.nn.functional.pad(img, (px, px, px, px),
                                              mode="replicate")
                n = img.shape[-1]
                img = img[:, :, px + dy:n - px + dy, px + dx:n - px + dx]
            prop = torch.from_numpy(
                (props[idx] - prop_mean) / prop_std).float().to(device)
            a = np.stack([acts[i:i + args.horizon] for i in idx], 0)
            a = torch.from_numpy((a - act_mean) / act_std).float().to(device)
            s = torch.from_numpy(
                np.eye(N_SKILL_CAP)[skills[idx]]).float().to(device)
            loss = model.loss(a, (img, prop), s, tau_power=args.tau_power)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss))
            if (it + 1) % 2500 == 0:
                print(f"[img-train] {task.name[:40]} iter {it+1} "
                      f"loss {np.mean(losses[-500:]):.5f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        out = os.path.join(MODEL_DIR, f"dpi_libero_{task.name}.pt")
        model.save(out, prop_mean, prop_std, act_mean, act_std)
        print(f"[img-train] saved {out} ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
