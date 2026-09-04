"""技能衔接分水岭诊断: terminal state 扰动 -> 后继技能 B 真实 rollout -> P(B|s) 曲线。

科学问题(「分水岭」实验): A 的终态差异是否强烈决定 B 的成功率?
- 连续类扰动(puck_dx/dy, eef_offset)敏感 -> Future Success Predictor + Candidate
  Reranking 路线成立(阶段 C);
- 结构类扰动(puck_drop)敏感 -> recovery 路线; 连续类平坦 -> 回头强化 action-level。

协议(冻结技能, 不训任何新模型):
1. calibrate: n 回合 chord 5 链(主表配置 λ=0.3, mask+proj, 协议同 eval_compose),
   记录各边界 obs 散布 -> 扰动档位参考;
2. diagnose: 每边界用脚本控制器 setup 到 A 终态 -> 施加参数化扰动
   (物体 xy 瞬移 / eef 携物移开 / 物体掉落) -> 冻结 DP receding-horizon
   rollout B(n_ddim=24, resample=8, 协议同 eval_compose) -> skill_success。
输出: results/metaworld/eval/diag_handoff.json + diag_handoff.png + 判读结论。
"""
import argparse
import json
import os
import time

import h5py
import mujoco
import numpy as np
import torch

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from swdp.policy import SkillDP  # noqa: E402
from swdp import chord_compose as cc  # noqa: E402
from skills import make_env, SKILLS, parse_pp  # noqa: E402
from eval_compose import skill_success, SKILL_NAMES  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCENE = "pick-place-v3"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "../../results/metaworld/data")
EVAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "../../results/metaworld/eval")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "../../results/metaworld/models")

# ---------------- 边界与扰动定义 ----------------
# 每项: (A, B, A 的脚本前置序列)
BOUNDARIES = [
    ("carry", "place", ["reach", "grasp", "lift", "carry"]),
    ("grasp", "carry", ["reach", "grasp"]),
    ("lift", "place", ["reach", "grasp", "lift"]),
    ("reach", "grasp", ["reach"]),
]
SETUP_STEPS = {"reach": 30, "grasp": 30, "lift": 25, "carry": 30}
# B 的执行步数(与 eval_compose TASKS 5 链 [30,25,25,30,20] 一致)
B_STEPS = {"reach": 30, "grasp": 25, "lift": 25, "carry": 30, "place": 20}

# 每边界适用的扰动维度(drop 对 reach->grasp 无意义: 物体本就在桌面)
PERTURB_DIMS = {
    "carry->place": ["puck_dx", "puck_dy", "eef_offset", "puck_drop"],
    "grasp->carry": ["puck_dx", "puck_dy", "eef_offset", "puck_drop"],
    "lift->place": ["puck_dx", "puck_dy", "eef_offset", "puck_drop"],
    "reach->grasp": ["puck_dx", "puck_dy", "eef_offset"],
}
CONTINUOUS_KINDS = {"puck_dx", "puck_dy", "eef_offset"}
STRUCTURAL_KINDS = {"puck_drop"}

# 默认扰动档位(米); calibrate 的实测散布可覆盖
MAGS = {
    "puck_dx": [0.02, 0.04, 0.06, 0.09, 0.12, 0.15],
    "puck_dy": [0.02, 0.04, 0.06, 0.09, 0.12, 0.15],
    "eef_offset": [0.03, 0.05, 0.08, 0.11, 0.14, 0.18],
    "puck_drop": [0.0, 0.03, 0.06, 0.09, 0.12, 0.15],
}
OBJ_QPOS_ADR = 9          # 物体 free joint 的 pos 地址(qpos[9:12]=xyz)


def diag_success(scene, name, env, obs, info):
    """诊断用增强判定(比 eval_compose 更严: 技能语义目标必须含物体)。-
    grasp/carry 的原判定不看物体(空手闭合/空手到位也算成功), 会掩盖扰动效应。"""
    if scene == "pick-place-v3":
        hand, grip = obs[:3], obs[3]
        puck, goal = obs[4:7], obs[-3:]
        if name == "reach":
            return float(np.linalg.norm(hand[:2] - puck[:2]) < 0.03)
        if name == "grasp":
            return float(grip < 0.75 and
                         np.linalg.norm(hand[:2] - puck[:2]) < 0.04)
        if name == "lift":
            return float(puck[2] > 0.08)
        if name == "carry":
            return float(np.linalg.norm(hand[:2] - goal[:2]) < 0.05 and
                         np.linalg.norm(puck[:2] - goal[:2]) < 0.07 and
                         grip < 0.6)
        if name == "place":
            if isinstance(info, dict) and info.get("success", 0.0) > 0.5:
                return 1.0
            return float(np.linalg.norm(puck - goal) < 0.07 and grip > 0.7)
    return skill_success(scene, name, env, obs, info)


def load_norm():
    with h5py.File(os.path.join(DATA_DIR, f"{SCENE}.h5"), "r") as f:
        return (f["obs_mean"][:], f["obs_std"][:],
                f["act_mean"][:], f["act_std"][:])


def setup_boundary(env, obs, setup_seq):
    """脚本控制器执行 A 序列, 返回 A 终态 obs(与 eval_compose.setup 同协议)。"""
    for name in setup_seq:
        ctrl = SKILLS[SCENE][name](env)
        for _ in range(SETUP_STEPS[name]):
            obs, *_ = env.step(ctrl.act(obs))
    return obs


def perturb_obs(env, obs, kind, mag, rng, target, holding=False):
    """在边界状态上施加扰动, 返回新 obs。

    - puck_dx/puck_dy: 物体沿轴瞬移 ±mag(构造假想 A 终态, 隔离 B 对物体位置的敏感度)
    - eef_offset:      eef 朝远离 target 的水平方向移开 mag(holding=True 携物:
                       手-物-目标联合偏移, 测 B 闭环纠正; 物理一致: 闭环脚本动作)
    - puck_drop:       物体掉落桌面(z=0.025) + xy 随机偏移 mag(结构失配/恢复场景)
    """
    inner = env._env
    m, d = inner.model, inner.data
    if kind in ("puck_dx", "puck_dy"):
        sgn = 1.0 if rng.random() < 0.5 else -1.0
        d.qpos[OBJ_QPOS_ADR + (0 if kind == "puck_dx" else 1)] += sgn * mag
        d.qvel[OBJ_QPOS_ADR:OBJ_QPOS_ADR + 6] = 0.0
        mujoco.mj_forward(m, d)
        return inner._get_obs()
    if kind == "eef_offset":
        # eef 朝远离 target 的水平方向移开 mag: holding=True 携物(手-物-目标
        # 联合偏移, 测 B 闭环纠正), holding=False 空手(如 reach 边界)。
        o = parse_pp(obs)
        v = o["hand"][:2] - np.asarray(target)[:2]
        n = float(np.linalg.norm(v))
        v = v / (n + 1e-8) if n > 1e-6 else np.array([1.0, 0.0])
        goal_pt = o["hand"][:2] + v * mag
        g = 1.0 if holding else 0.0
        for _ in range(60):
            o = parse_pp(obs)
            delta = np.clip(5.0 * (goal_pt - o["hand"][:2]), -1.0, 1.0)
            obs, *_ = env.step(
                np.array([delta[0], delta[1], 0.0, g], dtype=np.float32))
            if float(np.linalg.norm(
                    parse_pp(obs)["hand"][:2] - goal_pt)) < 0.01:
                break
        return obs
    if kind == "puck_drop":
        ang = rng.uniform(0, 2 * np.pi)
        d.qpos[OBJ_QPOS_ADR] += np.cos(ang) * mag
        d.qpos[OBJ_QPOS_ADR + 1] += np.sin(ang) * mag
        d.qpos[OBJ_QPOS_ADR + 2] = 0.025  # 桌面高度(掉落终态)
        d.qvel[OBJ_QPOS_ADR:OBJ_QPOS_ADR + 6] = 0.0
        mujoco.mj_forward(m, d)
        return inner._get_obs()
    raise ValueError(kind)


@torch.no_grad()
def rollout_B(dp, env, obs, skill_b, norm, seed=0, n_ddim=24, resample=8):
    """冻结 DP receding-horizon 执行 B(协议同 eval_compose)。返回 success。"""
    obs_mean, obs_std, act_mean, act_std = norm
    sid = SKILL_NAMES[SCENE].index(skill_b)

    def norm_obs(o):
        return torch.from_numpy((o - obs_mean) / obs_std).float().to(
            DEVICE).unsqueeze(0)

    def onehot(i):
        z = np.zeros((1, dp.n_skills), dtype=np.float32)
        z[0, i] = 1.0
        return torch.from_numpy(z).to(DEVICE)

    def sample_chunk(o, ctr):
        return dp.sample(o, onehot(sid), n_steps=n_ddim, seed=seed * 100003 + ctr)

    steps = B_STEPS[skill_b]
    chunk = sample_chunk(norm_obs(obs), 0)
    step_in_chunk = 0
    for t in range(steps):
        a_raw = np.clip(chunk[0, step_in_chunk].cpu().numpy() * act_std
                        + act_mean, -1.0, 1.0)
        obs, rew, term, trunc, info = env.step(a_raw)
        step_in_chunk += 1
        if step_in_chunk >= resample:
            chunk = sample_chunk(norm_obs(obs), t + 1)
            step_in_chunk = 0
    return float(diag_success(SCENE, skill_b, env, obs, info))


@torch.no_grad()
def calibrate(dp, norm, n_episodes=20, lam=0.3):
    """跑 chord 5 链(主表配置), 记录各边界 obs 的散布 -> 扰动档位参考。"""
    obs_mean, obs_std, act_mean, act_std = norm
    seq = ["reach", "grasp", "lift", "carry", "place"]
    steps = [B_STEPS[s] for s in seq]
    ids = {n: i for i, n in enumerate(SKILL_NAMES[SCENE])}
    snaps = {f"{seq[i]}->{seq[i + 1]}": [] for i in range(len(seq) - 1)}

    def norm_obs(o):
        return torch.from_numpy((o - obs_mean) / obs_std).float().to(
            DEVICE).unsqueeze(0)

    def onehot(i):
        z = np.zeros((1, dp.n_skills), dtype=np.float32)
        z[0, i] = 1.0
        return torch.from_numpy(z).to(DEVICE)

    for ep in range(n_episodes):
        env = make_env(SCENE, seed=1000 + ep)
        obs, _ = env.reset()
        chunk = dp.sample(norm_obs(obs), onehot(ids[seq[0]]), n_steps=24,
                          seed=ep * 7 + 1)
        step_in_chunk = step_in_skill = 0
        cur = 0
        for t in range(sum(steps)):
            if t > 0 and step_in_skill >= steps[cur]:
                cur += 1
                step_in_skill = 0
                o = parse_pp(obs)
                snaps[f"{seq[cur-1]}->{seq[cur]}"].append(
                    dict(hand=o["hand"].tolist(), grip=float(o["grip"]),
                         puck=o["puck"].tolist(), goal=o["goal"].tolist()))
                o_t = norm_obs(obs)
                anchor = dp.sample(o_t, onehot(ids[seq[cur - 1]]), n_steps=24,
                                   seed=ep * 11 + t)
                mask = cc.temporal_mask(anchor.shape[1], 0, 4, DEVICE)
                a_new, _ = cc.switch(dp, o_t, anchor,
                                     onehot(ids[seq[cur - 1]]),
                                     onehot(ids[seq[cur]]), 0.9, 0.15, lam,
                                     1, "chord", mask, True, seed=ep + t)
                chunk = a_new
                step_in_chunk = 0
            a_raw = np.clip(chunk[0, step_in_chunk].cpu().numpy() * act_std
                            + act_mean, -1.0, 1.0)
            obs, *_ = env.step(a_raw)
            step_in_chunk += 1
            step_in_skill += 1
            if step_in_chunk >= 8:
                chunk = dp.sample(norm_obs(obs), onehot(ids[seq[cur]]),
                                  n_steps=24, seed=ep * 13 + t)
                step_in_chunk = 0
        env.close()

    stats = {}
    for key, rows in snaps.items():
        if not rows:
            continue
        hand = np.array([r["hand"] for r in rows])
        puck = np.array([r["puck"] for r in rows])
        hand_goal = hand[:, :2] - np.array([r["goal"][:2] for r in rows])
        stats[key] = dict(
            n=len(rows),
            hand_goal_xy_std=float(np.linalg.norm(hand_goal.std(0))),
            hand_z_std=float(hand[:, 2].std()),
            puck_xy_std=float(np.linalg.norm(puck[:, :2].std(0))),
            puck_z_std=float(puck[:, 2].std()),
            hand_goal_xy_mean=float(np.linalg.norm(hand_goal.mean(0))),
        )
    return stats


def diagnose(dp, norm, n_seeds=8, smoke=False):
    """主诊断: 每边界 x (baseline + 扰动维度 x 档位) x seeds -> P(B|s)。"""
    rows = []
    t0 = time.time()
    for a_name, b_name, setup in BOUNDARIES:
        bkey = f"{a_name}->{b_name}"
        for seed in range(n_seeds):
            env = make_env(SCENE, seed=1000 + seed)
            # --- baseline(无扰动) ---
            obs, _ = env.reset()
            obs0 = setup_boundary(env, obs, setup)
            o = parse_pp(obs0)
            succ = rollout_B(dp, env, obs0, b_name, norm, seed=seed)
            rows.append(dict(boundary=bkey, kind="baseline", mag=0.0,
                             seed=seed, success=succ, skill=b_name,
                             hand=o["hand"].tolist(), puck=o["puck"].tolist(),
                             grip=float(o["grip"]), goal=o["goal"].tolist()))
            # --- 扰动 ---
            kinds = PERTURB_DIMS[bkey]
            mags = {k: (MAGS[k][:2] if smoke else MAGS[k]) for k in kinds}
            for ki, kind in enumerate(kinds):
                for mi, mag in enumerate(mags[kind]):
                    obs, _ = env.reset()
                    obs0 = setup_boundary(env, obs, setup)
                    rng = np.random.default_rng(seed * 1000 + ki * 100 + mi)
                    # eef 远离方向的目标点: place 类=goal, 其余=puck;
                    # holding: A 拿物时携物移开(grasp/lift/carry 边界)
                    target = parse_pp(obs0)["goal"] if b_name == "place" \
                        else parse_pp(obs0)["puck"]
                    holding = a_name != "reach"
                    obs_p = perturb_obs(env, obs0, kind, mag, rng, target,
                                        holding=holding)
                    op = parse_pp(obs_p)
                    succ = rollout_B(dp, env, obs_p, b_name, norm,
                                     seed=seed * 31 + ki * 7 + mi)
                    rows.append(dict(boundary=bkey, kind=kind, mag=float(mag),
                                     seed=seed, success=succ, skill=b_name,
                                     hand=op["hand"].tolist(),
                                     puck=op["puck"].tolist(),
                                     grip=float(op["grip"]),
                                     goal=op["goal"].tolist()))
                    if (len(rows)) % 50 == 0:
                        el = time.time() - t0
                        print(f"[diag] {len(rows)} rows, {el:.0f}s "
                              f"(last {bkey}/{kind}/{mag}: {succ})")
            env.close()
    return rows


def summarize(rows):
    """按 边界x维度 聚合成功率曲线 + 判读。"""
    out = {}
    for bkey in dict.fromkeys(r["boundary"] for r in rows):
        base = [r["success"] for r in rows
                if r["boundary"] == bkey and r["kind"] == "baseline"]
        entry = dict(baseline=float(np.mean(base)), n_baseline=len(base),
                     kinds={})
        for kind in dict.fromkeys(r["kind"] for r in rows
                                  if r["boundary"] == bkey
                                  and r["kind"] != "baseline"):
            mags = sorted({r["mag"] for r in rows
                           if r["boundary"] == bkey and r["kind"] == kind})
            rates, ses = [], []
            for mag in mags:
                ys = [r["success"] for r in rows if r["boundary"] == bkey
                      and r["kind"] == kind and r["mag"] == mag]
                rates.append(float(np.mean(ys)))
                ses.append(float(np.sqrt(max(np.mean(ys) * (1 - np.mean(ys)),
                                             1e-6) / len(ys))))
            entry["kinds"][kind] = dict(
                mags=mags, rates=rates, se=ses, n_per_mag=len(ys) // len(mags),
                max_drop=entry["baseline"] - min(rates),
                min_rate=min(rates))
        out[bkey] = entry
    # 全局判读: 连续类 vs 结构类
    cont_drops, struct_drops = [], []
    for bkey, e in out.items():
        for kind, k in e["kinds"].items():
            if kind in CONTINUOUS_KINDS:
                cont_drops.append(k["max_drop"])
            elif kind in STRUCTURAL_KINDS:
                struct_drops.append(k["max_drop"])
    cont_drop = float(np.mean(cont_drops)) if cont_drops else 0.0
    struct_drop = float(np.mean(struct_drops)) if struct_drops else 0.0
    verdict = dict(
        continuous_max_drop=cont_drop, structural_max_drop=struct_drop,
        sensitive=bool(cont_drop >= 0.25),
        verdict=("SENSITIVE: 连续类扰动显著降低 B 成功率 -> 启动阶段 C "
                 "(F_B + Candidate Reranking)" if cont_drop >= 0.25 else
                 "FLAT: 连续类扰动内 P(B) 平坦 -> 不上 F_B, 回头强化 "
                 "action-level 衔接"))
    return out, verdict


def plot(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bkeys = list(summary.keys())
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, bkey in zip(axes.flat, bkeys):
        e = summary[bkey]
        ax.axhline(e["baseline"], color="gray", ls=":", lw=1.5,
                   label=f"baseline={e['baseline']:.2f}")
        for kind, k in e["kinds"].items():
            struct = kind in STRUCTURAL_KINDS
            mags = k["mags"]
            lo = [max(0, r - 1.96 * s) for r, s in zip(k["rates"], k["se"])]
            hi = [min(1, r + 1.96 * s) for r, s in zip(k["rates"], k["se"])]
            ax.plot(mags, k["rates"], marker="o", ls="--" if struct else "-",
                    label=f"{kind} (drop {k['max_drop']:.2f})")
            ax.fill_between(mags, lo, hi, alpha=0.15)
        ax.set_title(bkey)
        ax.set_xlabel("扰动幅度 (m)")
        ax.set_ylabel("P(B success)")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7)
    fig.suptitle("Terminal-state 扰动 -> 后继技能成功率 (分水岭诊断)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["calibrate", "diagnose"])
    ap.add_argument("--n_episodes", type=int, default=20,
                    help="calibrate 回合数")
    ap.add_argument("--n_seeds", type=int, default=8, help="diagnose 种子数")
    ap.add_argument("--smoke", action="store_true", help="小规模冒烟")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    dp = SkillDP.load(os.path.join(MODEL_DIR, f"dp_{SCENE}.pt"), DEVICE)
    dp.eval()
    norm = load_norm()
    os.makedirs(EVAL_DIR, exist_ok=True)

    if args.cmd == "calibrate":
        stats = calibrate(dp, norm, n_episodes=args.n_episodes)
        path = os.path.join(EVAL_DIR, "diag_handoff_calib.json")
        with open(path, "w") as f:
            json.dump(stats, f, indent=2)
        print(json.dumps(stats, indent=2))
        print(f"[calib] saved {path}")
        return

    rows = diagnose(dp, norm, n_seeds=args.n_seeds, smoke=args.smoke)
    summary, verdict = summarize(rows)
    out = dict(args=vars(args), rows=rows, summary=summary, verdict=verdict)
    tag = args.out or ("diag_handoff_smoke.json" if args.smoke
                       else "diag_handoff.json")
    path = os.path.join(EVAL_DIR, tag)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    try:
        plot(summary, path.replace(".json", ".png"))
    except Exception as e:  # noqa: BLE001
        print(f"[diag] plot failed: {e}")
    print(json.dumps(verdict, indent=2, ensure_ascii=False))
    for bkey, e in summary.items():
        kinds = {k: round(v["rates"][0], 2) for k, v in e["kinds"].items()}
        print(f"[diag] {bkey}: base={e['baseline']:.2f} "
              f"first-mag rates={kinds}")
    print(f"[diag] saved {path}")


if __name__ == "__main__":
    main()
