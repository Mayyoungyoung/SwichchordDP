"""Downstream-Compatible Skill Learning 评估（grasp→lift 第一轮）。

对每个候选模型（原模型 + 各加权微调模型）rollout n_episodes 次:
脚本 reach → 候选模型 grasp 30 步 → settle → 记录终态 → 冻结 base 模型
lift ×K(真实 outcome)。下游 lift 固定用冻结模型执行: 设计上「塑造上游终态」
的下游技能是冻结的(与 dc_collect 的 y 标签协议一致), 用候选模型会混淆
「终态改善」与「lift 自身漂移」。评估种子区间(--seed0, 默认 5000)与
收集区间(dc_collect --seed0 3000)不相交, 评估集为留出集。

指标: P(grasp 语义成功) / P(lift|grasp) / e2e(期望口径: grasp_succ×lift_p 均值) /
      终态几何分布(grip/hand-puck xy) + terminal F_B 分布。

用法: python dc_eval.py --models base,dc_outcome_l2.0,dc_quality_l2.0
输出: results/metaworld/eval/dc_eval.json + dc_eval.png(Pareto 图 + 分布对比)
"""
import argparse
import json
import os
import time

import mujoco
import numpy as np
import torch

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from swdp.policy import SkillDP  # noqa: E402
from swdp.success_model import featurize, load as fb_load  # noqa: E402
from skills import make_env, SKILLS, parse_pp  # noqa: E402
from diag_handoff import load_norm, diag_success  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCENE = "pick-place-v3"
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "../../results/metaworld/models")
EVAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "../../results/metaworld/eval")
SID_GRASP, SID_LIFT = 1, 2


def load_dp(name):
    if name == "base":
        return SkillDP.load(os.path.join(MODEL_DIR,
                                         "dp_pick-place-v3.pt"), DEVICE)
    return SkillDP.load(os.path.join(MODEL_DIR, f"{name}.pt"), DEVICE)


@torch.no_grad()
def rollout_grasp(dp, norm, env, obs, seed, n_steps=30):
    obs_mean, obs_std, act_mean, act_std = norm

    def norm_obs(o):
        return torch.from_numpy((o - obs_mean) / obs_std).float().to(
            DEVICE).unsqueeze(0)

    def onehot(i):
        z = np.zeros((1, dp.n_skills), dtype=np.float32)
        z[0, i] = 1.0
        return torch.from_numpy(z).to(DEVICE)

    chunk = dp.sample(norm_obs(obs), onehot(SID_GRASP), n_steps=24,
                      seed=seed)
    step_in = 0
    for t in range(n_steps):
        a_raw = np.clip(chunk[0, step_in].cpu().numpy() * act_std
                        + act_mean, -1.0, 1.0)
        obs, *_ = env.step(a_raw)
        step_in += 1
        if step_in >= 8:
            chunk = dp.sample(norm_obs(obs), onehot(SID_GRASP), n_steps=24,
                              seed=seed + t)
            step_in = 0
    return obs


@torch.no_grad()
def eval_lift(dp, norm, env, s0, k=10, seed0=4000):
    obs_mean, obs_std, act_mean, act_std = norm

    def norm_obs(o):
        return torch.from_numpy((o - obs_mean) / obs_std).float().to(
            DEVICE).unsqueeze(0)

    def onehot(i):
        z = np.zeros((1, dp.n_skills), dtype=np.float32)
        z[0, i] = 1.0
        return torch.from_numpy(z).to(DEVICE)

    succs = []
    for kk in range(k):
        d = env._env.data
        m = env._env.model
        nq, nv = d.qpos.shape[0], d.qvel.shape[0]
        d.qpos[:] = s0[:nq]
        d.qvel[:] = s0[nq:nq + nv]
        d.mocap_pos[:] = s0[nq + nv:nq + nv + 3]
        d.mocap_quat[:] = s0[nq + nv + 3:nq + nv + 7]
        mujoco.mj_forward(m, d)
        env._env.curr_path_length = 0
        obs = env._env._get_obs()
        chunk = dp.sample(norm_obs(obs), onehot(SID_LIFT), n_steps=24,
                          seed=seed0 + kk)
        step_in = 0
        info = {}
        for t in range(25):
            a = np.clip(chunk[0, step_in].cpu().numpy() * act_std
                        + act_mean, -1.0, 1.0)
            obs, rew, term, trunc, info = env.step(a)
            step_in += 1
            if step_in >= 8:
                chunk = dp.sample(norm_obs(obs), onehot(SID_LIFT),
                                  n_steps=24, seed=seed0 + kk + t)
                step_in = 0
        succs.append(float(diag_success(SCENE, "lift", env, obs, info)))
    return float(np.mean(succs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="base,dc_outcome_l2.0")
    ap.add_argument("--n_episodes", type=int, default=120)
    ap.add_argument("--seed0", type=int, default=5000,
                    help="评估回合种子起点(与 dc_collect 的 3000 不相交)")
    ap.add_argument("--k_lift", type=int, default=10,
                    help="每终态下游 lift rollout 次数")
    ap.add_argument("--fb", default=os.path.join(MODEL_DIR,
                                                 "fb_pick-place-v3_v2.pt"))
    ap.add_argument("--out", default="dc_eval")
    args = ap.parse_args()

    norm = load_norm()
    # F_B(默认 v2 自然分布版, 在 grasp→lift 自然终态上有排序能力, §13.5)
    fb, fb_skills = fb_load(args.fb, DEVICE)
    fb_onehot = np.zeros(len(fb_skills), dtype=np.float32)
    fb_onehot[fb_skills.index("lift")] = 1.0
    # 下游 lift 固定用冻结 base 模型执行(见模块 docstring)
    dp_lift = SkillDP.load(os.path.join(MODEL_DIR,
                                        "dp_pick-place-v3.pt"), DEVICE)
    dp_lift.eval()
    models = args.models.split(",")
    results = {}
    rows_by_model = {}
    for name in models:
        dp = load_dp(name)
        dp.eval()
        rows = []
        t0 = time.time()
        for ep in range(args.n_episodes):
            env = make_env(SCENE, seed=args.seed0 + ep)
            obs, _ = env.reset()
            ctrl = SKILLS[SCENE]["reach"](env)
            for _ in range(30):
                obs, *_ = env.step(ctrl.act(obs))
            obs = rollout_grasp(dp, norm, env, obs, seed=ep * 7 + 1)
            for _ in range(20):
                obs, *_ = env.step(np.array([0.0, 0.0, 0.0, 1.0],
                                            dtype=np.float32))
            d = env._env.data
            s0 = np.concatenate([d.qpos.copy(), d.qvel.copy(),
                                 d.mocap_pos.ravel().copy(),
                                 d.mocap_quat.ravel().copy()])
            o = parse_pp(obs)
            grasp_succ = float(diag_success(SCENE, "grasp", env, obs, {}))
            lift_p = eval_lift(dp_lift, norm, env, s0, k=args.k_lift,
                               seed0=4000 + ep * 100)
            fb_v = float(fb.predict(featurize(
                o["hand"], float(o["grip"]), o["puck"], o["goal"]),
                fb_onehot))
            rows.append(dict(grasp_succ=grasp_succ, lift_p=lift_p,
                             # e2e 期望口径(与 lift_cond 连续口径一致):
                             # E[grasp∧lift] = grasp_succ × P(lift|s)
                             e2e=float(grasp_succ > 0.5) * lift_p,
                             fb=fb_v, grip=float(o["grip"]),
                             hp_xy=float(np.linalg.norm(
                                 o["hand"][:2] - o["puck"][:2]))))
            env.close()
            if (ep + 1) % 40 == 0:
                print(f"[dc-eval] {name} ep{ep + 1} ({time.time() - t0:.0f}s)")
        g = np.array([r["grasp_succ"] for r in rows])
        p = np.array([r["lift_p"] for r in rows])
        e = np.array([r["e2e"] for r in rows])
        fb_v = np.array([r["fb"] for r in rows])
        results[name] = dict(
            grasp_rate=float(g.mean()),
            lift_cond=float(p[g > 0.5].mean()) if (g > 0.5).any() else 0.0,
            e2e=float(e.mean()),
            fb_mean=float(fb_v.mean()),
            fb_hi=float(np.mean(fb_v > 0.5)),
            grip=dict(mean=float(np.mean([r["grip"] for r in rows])),
                      std=float(np.std([r["grip"] for r in rows]))),
            hp_xy=dict(mean=float(np.mean([r["hp_xy"] for r in rows])),
                       std=float(np.std([r["hp_xy"] for r in rows]))),
            n=len(rows))
        rows_by_model[name] = rows
        print(f"[dc-eval] {name}: grasp={results[name]['grasp_rate']:.3f} "
              f"lift|grasp={results[name]['lift_cond']:.3f} "
              f"e2e={results[name]['e2e']:.3f} fb_mean={results[name]['fb_mean']:.3f} "
              f"grip={results[name]['grip']['mean']:.3f}±{results[name]['grip']['std']:.3f}")
    path = os.path.join(EVAL_DIR, args.out + ".json")
    with open(path, "w") as f:
        json.dump(dict(args=vars(args), results=results), f, indent=2)
    print(f"[dc-eval] saved {path}")

    # Pareto 图 + 分布对比图(grip / hand-puck xy / terminal F_B)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = list(rows_by_model.keys())
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        ax = axes[0, 0]
        for name in names:
            r = results[name]
            ax.scatter(r["grasp_rate"], r["lift_cond"], s=110, label=name)
        ax.set_xlabel("P(grasp)")
        ax.set_ylabel("P(lift | grasp)")
        ax.set_title("Pareto: grasp success vs downstream take-over")
        ax.legend(fontsize=8)
        for label, key, ax in (("grip", "grip", axes[0, 1]),
                               ("hand-puck xy", "hp_xy", axes[1, 0]),
                               ("terminal F_B", "fb", axes[1, 1])):
            for name in names:
                vals = [r[key] for r in rows_by_model[name]]
                ax.hist(vals, bins=20, alpha=0.4, density=True, label=name)
            ax.set_xlabel(label)
            ax.set_ylabel("density")
            ax.legend(fontsize=8)
        fig.suptitle("Downstream-Compatible Skill Learning: terminal-state distributions")
        fig.tight_layout()
        png = os.path.join(EVAL_DIR, args.out + ".png")
        fig.savefig(png, dpi=130)
        plt.close(fig)
        print(f"[dc-eval] saved {png}")
    except Exception as e:  # noqa: BLE001
        print(f"[dc-eval] plot failed: {e}")


if __name__ == "__main__":
    main()
