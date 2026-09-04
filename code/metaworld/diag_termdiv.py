"""Terminal-State Diversity 决策性实验（RQ: 合法 A 终态是否显著决定 B 成功率）。

按外部建议的决策链实施（不做 candidate reranking / Foresight Guidance）:
Phase 1 collect:  前置技能用冻结 DP 链执行(误差累积 -> 代表真实链的终态分布,
                  第七轮审查发现脚本 setup 的散布仅真实链的 43%, 已改 DP),
                  A 自然执行 N 回合, 保存完整 sim state(qpos/qvel/mocap);
                  合法性过滤 = A-success(语义判定) ∧ physics-valid(静止/无掉落/限位);
Phase 2 rollout:  每个合法终态每次 rollout 前 restore 完整 sim state
                  (qpos/qvel/mocap + curr_path_length 重置) -> B × K -> P_emp;
Phase 3 metrics:  R_B dynamic range / top20-bottom20 gap(oracle) / FB ranking gap
                  (bootstrap 对 (fb, p_emp) 成对重采样) / Spearman / Brier / ECE;
Phase 4 plot:     几何特征 vs P_emp / F_B vs P_emp / PCA 2D / FB 分桶。

多技能对(普适性检验):
  carry->place  (第七轮原始对)
  reach->grasp  (§11 唯一显示散布>纠正域的对: 0.070m > 0.05m)
  grasp->lift   (抓取质量变异 -> 掉物, 真实失败模式)

Go/No-Go: R_B>0.5 且 Gap_FB>0.2 且 Spearman>0.5 -> GO;
          R_B<0.2 且 Gap_FB<0.1 -> NO-GO。中间情况如实报告。

子命令: collect / rollout / all
输出: results/metaworld/eval/termdiv_<pair>.json + _states.npz + .png
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
from swdp.success_model import featurize, ece, load as fb_load  # noqa: E402
from skills import make_env, SKILLS, parse_pp  # noqa: E402
from diag_handoff import diag_success, load_norm  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCENE = "pick-place-v3"
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "../../results/metaworld/models")
EVAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "../../results/metaworld/eval")
SKILL_NAMES = ["reach", "grasp", "lift", "carry", "place"]
OBJ_QPOS_ADR = 9
K_ROLLOUT = 10

# 每对配置: setup=前置技能(DP 链执行), settle_grip=settle 时的夹爪动作
# (持物 +1 闭合; 空手 0 保持张开)
PAIRS = {
    "carry->place": dict(a="carry", b="place", setup=["reach", "grasp", "lift"],
                         settle_grip=1.0),
    "reach->grasp": dict(a="reach", b="grasp", setup=[],
                         settle_grip=0.0),
    "grasp->lift": dict(a="grasp", b="lift", setup=["reach"],
                        settle_grip=1.0),
}
# 各技能步数(与 eval_compose 5 链 [30,25,25,30,20] 一致)
SKILL_STEPS = {"reach": 30, "grasp": 30, "lift": 25, "carry": 30, "place": 20}


def get_sid(name):
    return SKILL_NAMES.index(name)


@torch.no_grad()
def run_skill(dp, norm, env, obs, name, n_steps, seed):
    """冻结 DP 执行单技能 n_steps(receding-horizon, 协议同 eval_compose)。"""
    obs_mean, obs_std, act_mean, act_std = norm

    def norm_obs(o):
        return torch.from_numpy((o - obs_mean) / obs_std).float().to(
            DEVICE).unsqueeze(0)

    def onehot(i):
        z = np.zeros((1, dp.n_skills), dtype=np.float32)
        z[0, i] = 1.0
        return torch.from_numpy(z).to(DEVICE)

    chunk = dp.sample(norm_obs(obs), onehot(get_sid(name)), n_steps=24,
                      seed=seed)
    step_in = 0
    for t in range(n_steps):
        a_raw = np.clip(chunk[0, step_in].cpu().numpy() * act_std
                        + act_mean, -1.0, 1.0)
        obs, *_ = env.step(a_raw)
        step_in += 1
        if step_in >= 8:
            chunk = dp.sample(norm_obs(obs), onehot(get_sid(name)),
                              n_steps=24, seed=seed + t)
            step_in = 0
    return obs


def valid_A(cfg, env, obs):
    """合法性过滤: A-success(语义) ∧ physics-valid(通用)。"""
    d = env._env.data
    m = env._env.model
    o = parse_pp(obs)
    a = cfg["a"]
    # --- A-success(语义判定, 与 diag_success 增强版一致) ---
    if a == "carry":
        succ = (np.linalg.norm(o["hand"][:2] - o["goal"][:2]) < 0.05 and
                np.linalg.norm(o["puck"][:2] - o["goal"][:2]) < 0.07 and
                float(o["grip"]) < 0.6 and
                np.linalg.norm(o["hand"] - o["puck"]) < 0.10)
    elif a == "grasp":
        succ = (float(o["grip"]) < 0.75 and
                np.linalg.norm(o["hand"][:2] - o["puck"][:2]) < 0.04 and
                np.linalg.norm(o["hand"] - o["puck"]) < 0.10)
    elif a == "reach":
        succ = (np.linalg.norm(o["hand"][:2] - o["puck"][:2]) < 0.03 and
                float(o["grip"]) > 0.8)
    else:
        raise ValueError(a)
    if not succ:
        return False
    # --- physics-valid ---
    puck_z = d.qpos[OBJ_QPOS_ADR + 2]
    v_lin_obj = np.linalg.norm(d.qvel[OBJ_QPOS_ADR:OBJ_QPOS_ADR + 3])
    in_limit = True
    for i in range(len(d.qpos)):
        if i < len(m.jnt_range) and m.jnt_range[i][0] < m.jnt_range[i][1]:
            if d.qpos[i] < m.jnt_range[i][0] - 0.05 or \
               d.qpos[i] > m.jnt_range[i][1] + 0.05:
                in_limit = False
    if a == "carry":
        return bool(puck_z > 0.03 and v_lin_obj < 0.2 and in_limit)
    if a in ("grasp", "lift"):
        return bool(v_lin_obj < 0.3 and in_limit)
    # reach: 物体仍在桌面原位且静止(未被扰动)
    return bool(puck_z < 0.06 and v_lin_obj < 0.05 and in_limit)


def settle(env, n_steps=20, grip=1.0):
    """技能结束后 settle(grip 动作按技能配置: 持物 +1 / 空手 0)。
    返回 (settled_obs, 最后 5 步 hand/puck 位置漂移)。"""
    poses = []
    for _ in range(n_steps):
        obs, *_ = env.step(np.array([0.0, 0.0, 0.0, grip], dtype=np.float32))
        o = parse_pp(obs)
        poses.append(np.concatenate([o["hand"], o["puck"]]))
    last = np.array(poses[-5:])
    drift_v = float(np.abs(last - last[-1]).max())
    return obs, drift_v


@torch.no_grad()
def collect(dp, norm, cfg, n_episodes=240, seed0=1000):
    """Phase 1: 自然终态收集(DP 链 setup + 冻结 DP A + settle + 合法性过滤)。"""
    states, rows = [], []
    n_ok = n_fail_a = n_fail_phys = 0
    t0 = time.time()
    for ep in range(n_episodes):
        env = make_env(SCENE, seed=seed0 + ep)
        obs, _ = env.reset()
        # 前置技能: DP 链执行(误差累积, 代表真实链终态分布)
        for si, p in enumerate(cfg["setup"]):
            obs = run_skill(dp, norm, env, obs, p, SKILL_STEPS[p],
                            seed=ep * 11 + si)
        # A 自然执行
        obs = run_skill(dp, norm, env, obs, cfg["a"], SKILL_STEPS[cfg["a"]],
                        seed=ep * 7 + 1)
        obs, drift_v = settle(env, grip=cfg["settle_grip"])
        ok_a = valid_A(cfg, env, obs)
        if drift_v > 0.005:
            n_fail_phys += 1
            env.close()
            continue
        if not ok_a:
            n_fail_a += 1
            env.close()
            continue
        n_ok += 1
        d = env._env.data
        o = parse_pp(obs)
        states.append(np.concatenate([
            d.qpos.copy(), d.qvel.copy(),
            d.mocap_pos.ravel().copy(), d.mocap_quat.ravel().copy()]))
        rows.append(dict(idx=n_ok - 1, env_seed=seed0 + ep,
                         hand=o["hand"].tolist(), grip=float(o["grip"]),
                         puck=o["puck"].tolist(), goal=o["goal"].tolist()))
        env.close()
        if n_ok % 25 == 0:
            el = time.time() - t0
            print(f"[termdiv] collect ep{ep}: kept={n_ok} failA={n_fail_a} "
                  f"failPhys={n_fail_phys} ({el:.0f}s)")
    stats_c = dict(n_episodes=n_episodes, kept=n_ok, fail_a=n_fail_a,
                   fail_phys=n_fail_phys)
    print(f"[termdiv] collect done: {stats_c}")
    return np.array(states), rows, stats_c


def restore_state(env, svec):
    """恢复完整 sim state + 重置步数计数(每次 rollout 前必须调用)。"""
    d = env._env.data
    m = env._env.model
    nq, nv = d.qpos.shape[0], d.qvel.shape[0]
    d.qpos[:] = svec[:nq]
    d.qvel[:] = svec[nq:nq + nv]
    d.mocap_pos[:] = svec[nq + nv:nq + nv + 3]
    d.mocap_quat[:] = svec[nq + nv + 3:nq + nv + 7]
    mujoco.mj_forward(m, d)
    env._env.curr_path_length = 0


@torch.no_grad()
def rollout_B(dp, norm, cfg, states, rows, k=K_ROLLOUT, seed0=5000):
    """Phase 2: 每终态 -> B × K(每次 rollout 前 restore) -> P_emp。"""
    obs_mean, obs_std, act_mean, act_std = norm
    sid_b = get_sid(cfg["b"])

    def norm_obs(o):
        return torch.from_numpy((o - obs_mean) / obs_std).float().to(
            DEVICE).unsqueeze(0)

    def onehot(i):
        z = np.zeros((1, dp.n_skills), dtype=np.float32)
        z[0, i] = 1.0
        return torch.from_numpy(z).to(DEVICE)

    t0 = time.time()
    b_steps = SKILL_STEPS[cfg["b"]]
    for ri, (row, svec) in enumerate(zip(rows, states)):
        env = make_env(SCENE, seed=row["env_seed"])
        env.reset()
        succs = []
        for kk in range(k):
            restore_state(env, svec)      # 每次 rollout 前恢复终态(协议关键)
            obs = env._env._get_obs()
            chunk = dp.sample(norm_obs(obs), onehot(sid_b), n_steps=24,
                              seed=seed0 + ri * k + kk)
            step_in = 0
            info = {}
            for t in range(b_steps):
                a_raw = np.clip(chunk[0, step_in].cpu().numpy() * act_std
                                + act_mean, -1.0, 1.0)
                obs, rew, term, trunc, info = env.step(a_raw)
                step_in += 1
                if step_in >= 8:
                    chunk = dp.sample(norm_obs(obs), onehot(sid_b),
                                      n_steps=24,
                                      seed=seed0 + ri * k + kk + t)
                    step_in = 0
            succs.append(float(diag_success(SCENE, cfg["b"], env, obs, info)))
        row["p_emp"] = float(np.mean(succs))
        row["n_succ"] = int(np.sum(succs))
        env.close()
        if (ri + 1) % 50 == 0:
            el = time.time() - t0
            print(f"[termdiv] rollout {ri + 1}/{len(rows)} ({el:.0f}s) "
                  f"last p_emp={row['p_emp']:.1f}")
    return rows


def fb_score(fb, cfg, row):
    """F_B(s) 对 B=cfg['b']。"""
    onehot = np.zeros(len(SKILL_NAMES), dtype=np.float32)
    onehot[get_sid(cfg["b"])] = 1.0
    f = featurize(row["hand"], row["grip"], row["puck"], row["goal"])
    return float(fb.predict(f, onehot))


def bootstrap_paired(fbs, p_emp, fn, n_boot=1000, seed=0):
    """对 state 索引成对重采样 (fbs, p_emp) -> fn(fbs*, p_emp*) 的 95% CI。
    (修复: fb_gap 的 CI 必须保持 fb->p_emp 配对关系, 不能只重采样 p_emp)"""
    rng = np.random.default_rng(seed)
    n = len(p_emp)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(fn(np.asarray(fbs)[idx], np.asarray(p_emp)[idx]))
    vals = np.sort(vals)
    return float(vals[int(0.025 * n_boot)]), float(vals[int(0.975 * n_boot)])


def report(cfg, rows, stats_c):
    """Phase 3: 指标 + Go/No-Go。"""
    p_emp = np.array([r["p_emp"] for r in rows])
    fbs = np.array([r["fb"] for r in rows])
    n = len(rows)
    kk = max(1, int(n * 0.2))

    def gap(f, p):
        order = np.argsort(f)
        return float(p[order[-kk:]].mean() - p[order[:kk]].mean())

    r_b = float(p_emp.max() - p_emp.min())
    r_b_ci = bootstrap_paired(fbs, p_emp,
                              lambda f, p: p.max() - p.min())
    oracle_gap = gap(p_emp, p_emp)
    fb_gap = gap(fbs, p_emp)
    fb_gap_ci = bootstrap_paired(fbs, p_emp, gap)
    spearman = stats.spearmanr(fbs, p_emp)
    pearson = stats.pearsonr(fbs, p_emp)
    brier = float(np.mean((fbs - p_emp) ** 2))
    ece_v = ece(fbs, p_emp)
    qs = np.quantile(fbs, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    bins = []
    for i in range(5):
        lo, hi = qs[i], qs[i + 1]
        msk = (fbs >= lo) & (fbs <= hi)
        if msk.sum() == 0:
            continue
        bins.append(dict(lo=float(lo), hi=float(hi), n=int(msk.sum()),
                         mean_fb=float(fbs[msk].mean()),
                         mean_p=float(p_emp[msk].mean()),
                         ci=bootstrap_paired(fbs[msk], p_emp[msk],
                                             lambda f, p: p.mean())))
    verdict = "GO" if (r_b > 0.5 and fb_gap > 0.2 and
                       spearman.statistic > 0.5) else \
              "NO-GO" if (r_b < 0.2 and fb_gap < 0.1) else "INTERMEDIATE"
    out = dict(
        pair=f"{cfg['a']}->{cfg['b']}", n_states=n, stats_collect=stats_c,
        dynamic_range=dict(value=r_b, ci95=r_b_ci),
        p_emp=dict(mean=float(p_emp.mean()), std=float(p_emp.std()),
                   min=float(p_emp.min()), max=float(p_emp.max()),
                   hist=dict(zip(map(str, np.arange(0, 1.05, 0.1)),
                                 np.histogram(p_emp,
                                              bins=np.arange(-0.05, 1.06,
                                                             0.1))[0].tolist()))),
        oracle_gap=oracle_gap, fb_gap=dict(value=fb_gap, ci95=fb_gap_ci),
        spearman=dict(r=float(spearman.statistic), p=float(spearman.pvalue)),
        pearson=dict(r=float(pearson.statistic), p=float(pearson.pvalue)),
        brier=brier, ece=ece_v, fb_bins=bins,
        verdict=verdict,
        go_criteria="R_B>0.5 且 Gap_FB>0.2 且 Spearman>0.5",
    )
    print(f"[termdiv] {cfg['a']}->{cfg['b']} n={n}  R_B={r_b:.3f} {r_b_ci}  "
          f"oracle_gap={oracle_gap:.3f}  FB_gap={fb_gap:.3f} {fb_gap_ci}")
    print(f"[termdiv] Spearman r={spearman.statistic:.3f} "
          f"(p={spearman.pvalue:.4f})  Pearson r={pearson.statistic:.3f}")
    print(f"[termdiv] Brier={brier:.3f} ECE={ece_v:.3f}  "
          f"P_emp mean/std={p_emp.mean():.3f}/{p_emp.std():.3f}")
    print(f"[termdiv] VERDICT: {verdict}")
    return out


def plot(cfg, rows, report_d, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p_emp = np.array([r["p_emp"] for r in rows])
    fbs = np.array([r["fb"] for r in rows])
    hand = np.array([r["hand"] for r in rows])
    puck = np.array([r["puck"] for r in rows])
    goal = np.array([r["goal"] for r in rows])
    feats = {
        "hand-puck xy": np.linalg.norm(hand[:, :2] - puck[:, :2], axis=1),
        "hand z": hand[:, 2],
        "grip": np.array([r["grip"] for r in rows]),
    }
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (name, x) in zip(axes.flat[:3], feats.items()):
        ax.scatter(x, p_emp, s=18, alpha=0.6)
        r = stats.spearmanr(x, p_emp).statistic
        ax.set_title(f"{name} (rho={r:.2f})", fontsize=9)
        ax.set_ylabel("P_emp(B|s)")
        ax.set_ylim(-0.05, 1.05)
    ax = axes[1, 0]
    ax.scatter(fbs, p_emp, s=18, alpha=0.6)
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlabel("F_B(s)"); ax.set_ylabel("P_emp(B|s)")
    ax.set_title(f"F_B vs P_emp (rho={report_d['spearman']['r']:.2f})")
    ax.set_ylim(-0.05, 1.05)
    # PCA 2D (方差贡献 = 奇异值平方)
    X = np.stack([featurize(r["hand"], r["grip"], r["puck"], r["goal"])
                  for r in rows])
    Xc = X - X.mean(0)
    u, s, vt = np.linalg.svd(Xc, full_matrices=False)
    pc = Xc @ vt.T[:, :2]
    ax = axes[1, 1]
    sc = ax.scatter(pc[:, 0], pc[:, 1], c=p_emp, cmap="RdYlGn", vmin=0, vmax=1,
                    s=22)
    fig.colorbar(sc, ax=ax, label="P_emp(B|s)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    evr = float((s[:2] ** 2).sum() / (s ** 2).sum()) if len(s) > 2 else 1.0
    ax.set_title(f"terminal states PCA (evr={evr:.2f})")
    ax = axes[1, 2]
    bins = report_d["fb_bins"]
    xs = [(b["lo"] + b["hi"]) / 2 for b in bins]
    ys = [b["mean_p"] for b in bins]
    cis = [b["ci"] for b in bins]
    ax.errorbar(xs, ys, yerr=[[y - c[0] for y, c in zip(ys, cis)],
                              [c[1] - y for y, c in zip(ys, cis)]],
                fmt="o-", capsize=3)
    ax.set_xlabel("F_B score bin"); ax.set_ylabel("mean P_emp")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"FB-ranked bins (gap={report_d['fb_gap']['value']:.2f})")
    fig.suptitle(f"Terminal-State Diversity: {cfg['a']}->{cfg['b']} "
                 f"(n={len(rows)}, R_B={report_d['dynamic_range']['value']:.2f}, "
                 f"verdict={report_d['verdict']})")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[termdiv] saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["collect", "rollout", "all"])
    ap.add_argument("--pair", default="carry->place", choices=list(PAIRS))
    ap.add_argument("--n_episodes", type=int, default=240)
    ap.add_argument("--k", type=int, default=K_ROLLOUT,
                    help="每终态 B rollout 次数")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cfg = PAIRS[args.pair]
    tag = args.out or f"termdiv_{cfg['a']}_{cfg['b']}"
    dp = SkillDP.load(os.path.join(MODEL_DIR, f"dp_{SCENE}.pt"), DEVICE)
    dp.eval()
    fb, _ = fb_load(os.path.join(MODEL_DIR, f"fb_{SCENE}.pt"), DEVICE)
    norm = load_norm()
    os.makedirs(EVAL_DIR, exist_ok=True)
    base = os.path.join(EVAL_DIR, tag)

    states, rows, stats_c = None, None, None
    if args.cmd in ("collect", "all"):
        states, rows, stats_c = collect(dp, norm, cfg,
                                        n_episodes=args.n_episodes)
        np.savez_compressed(base + "_states.npz", states=states)
        with open(base + "_collect.json", "w") as f:
            json.dump(dict(stats=stats_c, rows=rows), f, indent=2)
        print(f"[termdiv] collect saved {base}_states.npz "
              f"({len(rows)} states)")
    if args.cmd in ("rollout", "all"):
        if states is None:
            z = np.load(base + "_states.npz")
            states = z["states"]
            with open(base + "_collect.json") as f:
                cd = json.load(f)
            rows, stats_c = cd["rows"], cd["stats"]
        rows = rollout_B(dp, norm, cfg, states, rows, k=args.k)
        for r in rows:
            r["fb"] = fb_score(fb, cfg, r)
        rep = report(cfg, rows, stats_c)
        out = dict(args=vars(args), report=rep, rows=rows)
        with open(base + ".json", "w") as f:
            json.dump(out, f, indent=2)
        try:
            plot(cfg, rows, rep, base + ".png")
        except Exception as e:  # noqa: BLE001
            print(f"[termdiv] plot failed: {e}")
        print(f"[termdiv] saved {base}.json")


if __name__ == "__main__":
    main()
