"""Reachability / Controllability 实验(决策链第三问: 高 F_B 终态可达吗?)。

RQ: frozen DP carry 的原始执行产生什么 F_B 分布? A-tail(尾部最后 H=10 步
不同采样)能否显著产生更高 F_B 的终态?

协议(每 episode):
1. 脚本 setup(reach/grasp/lift);
2. DP carry 前 20 步(基线 seed) -> 保存 mid 状态(qpos/qvel/mocap);
3. 基线: 原始 seed 继续 10 步 + settle -> 终态 F_B(v2);
4. A-tail: 8 个不同 DP seed 各跑最后 10 步 + settle -> 8 个终态 F_B(v2)。
指标: P(F_B>θ) 基线 vs tails-max/tails-mean; θ = 部署 F_B v2 在 termdiv 自然
数据上的 80 分位数。若 tails 显著超过基线 -> A-tail+选择 有真实增益空间
(接回方法: 尾部用 F_B 选最优采样, 即 Future-aware A-tail)。

输出: results/metaworld/eval/reach_carry_place.json + .png
"""
import argparse
import json
import os
import time

import h5py
import mujoco
import numpy as np
import torch
from scipy import stats

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from swdp.policy import SkillDP  # noqa: E402
from swdp.success_model import (featurize,  # noqa: E402
                                load as fb_load)
from skills import make_env, SKILLS, parse_pp  # noqa: E402
from diag_handoff import load_norm  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCENE = "pick-place-v3"
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "../../results/metaworld/models")
EVAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "../../results/metaworld/eval")
TERMDIV = os.path.join(EVAL_DIR, "termdiv_carry_place.json")
SKILL_NAMES = ["reach", "grasp", "lift", "carry", "place"]
SETUP = ["reach", "grasp", "lift"]
SETUP_STEPS = {"reach": 30, "grasp": 30, "lift": 25}
A_TOTAL, A_TAIL, N_TAILS, N_SETTLE = 30, 10, 8, 20


def fb_v2_score(fb, o):
    sid = SKILL_NAMES.index("place")
    onehot = np.zeros(len(SKILL_NAMES), dtype=np.float32)
    onehot[sid] = 1.0
    f = featurize(o["hand"], float(o["grip"]), o["puck"], o["goal"])
    return float(fb.predict(f, onehot))


def settle(env, n_steps=N_SETTLE):
    """维持抓取 settle 后返回 obs(与 termdiv 训练分布一致)。"""
    for _ in range(n_steps):
        env.step(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
    return env._env._get_obs()


def save_state(env):
    d = env._env.data
    return np.concatenate([d.qpos.copy(), d.qvel.copy(),
                           d.mocap_pos.ravel().copy(),
                           d.mocap_quat.ravel().copy()])


def restore_state(env, svec):
    d = env._env.data
    m = env._env.model
    nq, nv = d.qpos.shape[0], d.qvel.shape[0]
    d.qpos[:] = svec[:nq]
    d.qvel[:] = svec[nq:nq + nv]
    d.mocap_pos[:] = svec[nq + nv:nq + nv + 3]
    d.mocap_quat[:] = svec[nq + nv + 3:nq + nv + 7]
    mujoco.mj_forward(m, d)
    return d


@torch.no_grad()
def run(dp, norm, fb, n_episodes=40, seed0=7000):
    obs_mean, obs_std, act_mean, act_std = norm
    sid_carry = SKILL_NAMES.index("carry")

    def norm_obs(o):
        return torch.from_numpy((o - obs_mean) / obs_std).float().to(
            DEVICE).unsqueeze(0)

    def onehot(i):
        z = np.zeros((1, dp.n_skills), dtype=np.float32)
        z[0, i] = 1.0
        return torch.from_numpy(z).to(DEVICE)

    def run_carry(env, obs, seed, n_steps):
        """从 obs 起跑 DP carry n_steps, 返回终态 obs。"""
        chunk = dp.sample(norm_obs(obs), onehot(sid_carry), n_steps=24,
                          seed=seed)
        step_in = 0
        for t in range(n_steps):
            a_raw = np.clip(chunk[0, step_in].cpu().numpy() * act_std
                            + act_mean, -1.0, 1.0)
            obs, *_ = env.step(a_raw)
            step_in += 1
            if step_in >= 8:
                chunk = dp.sample(norm_obs(obs), onehot(sid_carry),
                                  n_steps=24, seed=seed + t)
                step_in = 0
        return obs

    rows = []
    t0 = time.time()
    for ep in range(n_episodes):
        env = make_env(SCENE, seed=seed0 + ep)
        obs, _ = env.reset()
        for p in SETUP:
            ctrl = SKILLS[SCENE][p](env)
            for _ in range(SETUP_STEPS[p]):
                obs, *_ = env.step(ctrl.act(obs))
        base_seed = ep * 7 + 1
        obs = run_carry(env, obs, base_seed, A_TOTAL - A_TAIL)
        mid_state = save_state(env)
        # 基线: 原始 seed 尾部 + settle
        obs_t = run_carry(env, obs, base_seed, A_TAIL)
        o = parse_pp(settle(env))
        fb_base = fb_v2_score(fb, o)
        # A-tail: 8 个不同采样(同 mid 状态起)
        fb_tails = []
        for ti in range(N_TAILS):
            restore_state(env, mid_state)
            obs_ti = run_carry(env, env._env._get_obs(),
                               base_seed * 101 + ti, A_TAIL)
            oi = parse_pp(settle(env))
            fb_tails.append(fb_v2_score(fb, oi))
        rows.append(dict(ep=ep, base=fb_base, tails=fb_tails,
                         tail_max=float(np.max(fb_tails)),
                         tail_mean=float(np.mean(fb_tails))))
        env.close()
        if (ep + 1) % 10 == 0:
            print(f"[reach] ep{ep + 1}/{n_episodes} ({time.time() - t0:.0f}s)"
                  f" base={fb_base:.3f} tailmax={rows[-1]['tail_max']:.3f}")
    return rows


def report(rows, theta):
    base = np.array([r["base"] for r in rows])
    tmax = np.array([r["tail_max"] for r in rows])
    tmean = np.array([r["tail_mean"] for r in rows])
    out = dict(
        n_episodes=len(rows), theta=float(theta),
        base=dict(mean=float(base.mean()), std=float(base.std()),
                  p_above=float(np.mean(base > theta))),
        tail_max=dict(mean=float(tmax.mean()), std=float(tmax.std()),
                      p_above=float(np.mean(tmax > theta))),
        tail_mean=dict(mean=float(tmean.mean()), std=float(tmean.std()),
                       p_above=float(np.mean(tmean > theta))),
        delta=dict(
            max_vs_base=float((tmax - base).mean()),
            max_vs_base_p=float(stats.wilcoxon(tmax, base).pvalue),
            mean_vs_base=float((tmean - base).mean()),
            mean_vs_base_p=float(stats.wilcoxon(tmean, base).pvalue)),
    )
    print(f"[reach] theta={theta:.3f}")
    print(f"[reach] base   : F_B={base.mean():.3f}±{base.std():.3f} "
          f"P(>θ)={out['base']['p_above']:.3f}")
    print(f"[reach] tailmax: F_B={tmax.mean():.3f}±{tmax.std():.3f} "
          f"P(>θ)={out['tail_max']['p_above']:.3f} "
          f"(Δ={out['delta']['max_vs_base']:+.3f}, "
          f"Wilcoxon p={out['delta']['max_vs_base_p']:.4f})")
    print(f"[reach] tailmean: F_B={tmean.mean():.3f}±{tmean.std():.3f} "
          f"P(>θ)={out['tail_mean']['p_above']:.3f}")
    return out


def plot(rows, rep, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    base = [r["base"] for r in rows]
    tmax = [r["tail_max"] for r in rows]
    ax.scatter(base, tmax, s=18, alpha=0.7)
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlabel("baseline F_B (原始执行)")
    ax.set_ylabel("max F_B over 8 tails")
    ax.set_title("A-tail 可达性 (max over 8)")
    ax = axes[1]
    labs = ["base", "tail_mean", "tail_max"]
    vals = [rep["base"]["p_above"], rep["tail_mean"]["p_above"],
            rep["tail_max"]["p_above"]]
    ax.bar(labs, vals, color=["tab:gray", "tab:blue", "tab:green"])
    ax.axhline(rep["base"]["p_above"], color="tab:gray", ls=":")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center")
    ax.set_ylabel(f"P(F_B > θ={rep['theta']:.2f})")
    ax.set_title("到达高 F_B 区域的比例")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[reach] saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_episodes", type=int, default=40)
    ap.add_argument("--out", default="reach_carry_place")
    args = ap.parse_args()

    dp = SkillDP.load(os.path.join(MODEL_DIR, f"dp_{SCENE}.pt"), DEVICE)
    dp.eval()
    fb, _ = fb_load(os.path.join(MODEL_DIR, f"fb_{SCENE}_v2.pt"), DEVICE)
    norm = load_norm()
    # θ: v2 在自然数据上的 80 分位数
    with open(TERMDIV) as f:
        trows = json.load(f)["rows"]
    preds = [fb_v2_score(fb, dict(hand=r["hand"], grip=r["grip"],
                                  puck=r["puck"], goal=r["goal"]))
             for r in trows]
    theta = float(np.quantile(preds, 0.8))
    print(f"[reach] theta(80pct of natural F_B v2) = {theta:.3f}")

    rows = run(dp, norm, fb, n_episodes=args.n_episodes)
    rep = report(rows, theta)
    os.makedirs(EVAL_DIR, exist_ok=True)
    base = os.path.join(EVAL_DIR, args.out)
    with open(base + ".json", "w") as f:
        json.dump(dict(args=vars(args), report=rep, rows=rows), f, indent=2)
    try:
        plot(rows, rep, base + ".png")
    except Exception as e:  # noqa: BLE001
        print(f"[reach] plot failed: {e}")
    print(f"[reach] saved {base}.json")


if __name__ == "__main__":
    main()
