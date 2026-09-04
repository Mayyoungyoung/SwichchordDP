"""READY Phase 1 PoC: 五臂对照验证「显式优化后继就绪」的修复增量。

论文方案的 decisive experiment（docs/paper_plan_READY.md §5 Phase 1）:
对 grasp→lift 的失败态（语义合法但物理劣质抓取），比较五种处理:
  1. direct:     直接执行 lift（基线锚点, 预期 ≈0）
  2. random:     随机动作序列 H=10, 不变量过滤后执行（排除「任何扰动都有用」）
  3. re-execute: 重执行前序技能 DP grasp（关键基线: 修复收益是否只是重试?）
  4. READY:      CEM 搜索 H=10 动作序列, 目标 = ensemble mean V_lift(ŝ_H)
                 + 不变量约束（方法主体）
  5. oracle:     松爪放下 + 脚本 regrasp（上界）
每臂修复后 lift ×10（每次 restore 修复终态）→ P_lift。

指标: P_lift、不变量保持率（物体在手中）、动作代价 ‖a‖、墙钟。

用法: python ready_poc.py [--states tag ...] [--n_cem_pop 64] [--n_cem_gen 5]
数据: termdiv_grasp_lift(_ext) 的失败态（p_emp<=0.2）+ 全量状态训练 ensemble。
输出: results/metaworld/eval/ready_poc.json
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
from swdp.success_model import SuccessModel, featurize  # noqa: E402
from skills import make_env, SKILLS, parse_pp  # noqa: E402
from diag_handoff import load_norm, diag_success  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCENE = "pick-place-v3"
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "../../results/metaworld/models")
EVAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "../../results/metaworld/eval")
SKILL_NAMES = ["reach", "grasp", "lift", "carry", "place"]
SID_GRASP, SID_LIFT = 1, 2
H_REPAIR, K_EVAL = 10, 10


# ---------------- sim state 工具 ----------------

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
    env._env.curr_path_length = 0


def invariant_ok(obs):
    """任务不变量: 物体仍在手中(修复不撤销 grasp 成果)。"""
    o = parse_pp(obs)
    return (np.linalg.norm(o["hand"] - o["puck"]) < 0.10 and
            float(o["grip"]) < 0.6)


def fb_input(obs):
    o = parse_pp(obs)
    onehot = np.zeros(len(SKILL_NAMES), dtype=np.float32)
    onehot[SID_LIFT] = 1.0
    return featurize(o["hand"], float(o["grip"]), o["puck"],
                     o["goal"]), onehot


# ---------------- ensemble V_lift ----------------

def train_ensemble(rows, n_members=5, epochs=300, seed=0):
    """5-member ensemble（不同数据划分+初始化）。返回模型列表 + OOF 报告。"""
    X = np.stack([featurize(r["hand"], r["grip"], r["puck"], r["goal"])
                  for r in rows])
    S = np.tile(np.eye(len(SKILL_NAMES))[SID_LIFT].astype(np.float32),
                (len(rows), 1))
    y, p_state = [], []
    for r in rows:
        n = int(r["n_succ"])
        y.append(n / 10.0)                     # 状态级频率标签
        p_state.append(r["p_emp"])
    y = np.array(y, dtype=np.float32)
    models = []
    rng = np.random.default_rng(seed)
    for m in range(n_members):
        idx = rng.permutation(len(y))
        va = idx[:len(y) // 5]
        tr = idx[len(y) // 5:]
        model = SuccessModel(n_skills=len(SKILL_NAMES)).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3,
                               weight_decay=1e-4)
        Xt = torch.from_numpy(X[tr]).to(DEVICE)
        St = torch.from_numpy(S[tr]).to(DEVICE)
        yt = torch.from_numpy(y[tr]).to(DEVICE)
        for _ in range(epochs):
            opt.zero_grad()
            logit = model.net(torch.cat([Xt, St], dim=-1)).squeeze(-1)
            torch.nn.functional.binary_cross_entropy_with_logits(
                logit, yt).backward()
            opt.step()
        models.append(model)
    return models


@torch.no_grad()
def ensemble_V(models, obs):
    """返回 (mean, std)。"""
    f, onehot = fb_input(obs)
    ps = [float(m.predict(f, onehot)) for m in models]
    return float(np.mean(ps)), float(np.std(ps))


# ---------------- 技能执行 ----------------

@torch.no_grad()
def run_skill(dp, norm, env, obs, name, n_steps, seed):
    obs_mean, obs_std, act_mean, act_std = norm

    def norm_obs(o):
        return torch.from_numpy((o - obs_mean) / obs_std).float().to(
            DEVICE).unsqueeze(0)

    def onehot(i):
        z = np.zeros((1, dp.n_skills), dtype=np.float32)
        z[0, i] = 1.0
        return torch.from_numpy(z).to(DEVICE)

    sid = SKILL_NAMES.index(name)
    chunk = dp.sample(norm_obs(obs), onehot(sid), n_steps=24, seed=seed)
    step_in = 0
    for t in range(n_steps):
        a_raw = np.clip(chunk[0, step_in].cpu().numpy() * act_std
                        + act_mean, -1.0, 1.0)
        obs, *_ = env.step(a_raw)
        step_in += 1
        if step_in >= 8:
            chunk = dp.sample(norm_obs(obs), onehot(sid), n_steps=24,
                              seed=seed + t)
            step_in = 0
    return obs


@torch.no_grad()
def eval_lift(dp, norm, env, state, k=K_EVAL, seed0=9000):
    """给定终态 state → lift × K（每次 restore）→ P_lift。"""
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
        restore_state(env, state)
        obs = env._env._get_obs()
        chunk = dp.sample(norm_obs(obs), onehot(SID_LIFT), n_steps=24,
                          seed=seed0 + kk)
        step_in = 0
        info = {}
        for t in range(25):
            a_raw = np.clip(chunk[0, step_in].cpu().numpy() * act_std
                            + act_mean, -1.0, 1.0)
            obs, rew, term, trunc, info = env.step(a_raw)
            step_in += 1
            if step_in >= 8:
                chunk = dp.sample(norm_obs(obs), onehot(SID_LIFT),
                                  n_steps=24, seed=seed0 + kk + t)
                step_in = 0
        succs.append(float(diag_success(SCENE, "lift", env, obs, info)))
    return float(np.mean(succs))


# ---------------- 五臂 ----------------

@torch.no_grad()
def arm_ready(dp, norm, env, s0, models, pop=64, gens=5, seed=0):
    """READY: CEM 搜索 H 步动作序列, max mean V_lift(ŝ_H) + 不变量。"""
    rng = np.random.default_rng(seed)
    mu = np.zeros((H_REPAIR, 4))
    sigma = np.ones((H_REPAIR, 4)) * 0.5
    best_a, best_score = None, -1e9
    for g in range(gens):
        cand = rng.normal(mu[None], sigma[None], size=(pop, H_REPAIR, 4))
        cand = np.clip(cand, -1.0, 1.0)
        scores = []
        for ci in range(pop):
            restore_state(env, s0)
            obs = env._env._get_obs()
            for t in range(H_REPAIR):
                obs, *_ = env.step(cand[ci, t].astype(np.float32))
            v, _ = ensemble_V(models, obs)
            inv = 1.0 if invariant_ok(obs) else 0.0
            scores.append(v * inv - 10.0 * (1.0 - inv))
        scores = np.array(scores)
        if scores.max() > best_score:
            best_score = scores.max()
            best_a = cand[scores.argmax()].copy()
        # CEM 更新（精英 top 25%）
        elite = cand[np.argsort(scores)[-max(4, pop // 4):]]
        mu = elite.mean(0)
        sigma = elite.std(0) + 0.05
    # 执行 best_a 于真实 env
    restore_state(env, s0)
    obs = env._env._get_obs()
    for t in range(H_REPAIR):
        obs, *_ = env.step(best_a[t].astype(np.float32))
    return obs, best_a


@torch.no_grad()
def arm_random(dp, norm, env, s0, n_try=32, seed=0):
    """随机动作序列（不变量过滤, 无 V 引导）。"""
    rng = np.random.default_rng(seed)
    for _ in range(n_try):
        a = rng.uniform(-1, 1, size=(H_REPAIR, 4)).astype(np.float32)
        restore_state(env, s0)
        obs = env._env._get_obs()
        for t in range(H_REPAIR):
            obs, *_ = env.step(a[t])
        if invariant_ok(obs):
            return obs, a
    return obs, a     # 无满足者, 返回最后一个（如实评估）


@torch.no_grad()
def arm_reexecute(dp, norm, env, s0, seed):
    """重执行前序技能 DP grasp。"""
    restore_state(env, s0)
    obs = env._env._get_obs()
    obs = run_skill(dp, norm, env, obs, "grasp", 30, seed)
    return obs, None


@torch.no_grad()
def arm_oracle(dp, norm, env, s0):
    """oracle regrasp: 松爪放下 + settle + 脚本闭环重抓。"""
    restore_state(env, s0)
    obs = env._env._get_obs()
    # 1) 松爪 10 步（-1 = 张开）让物体落回桌面
    for _ in range(10):
        obs, *_ = env.step(np.array([0.0, 0.0, 0.0, -1.0],
                                    dtype=np.float32))
    # 2) settle（零动作）
    for _ in range(20):
        obs, *_ = env.step(np.array([0.0, 0.0, 0.0, 0.0],
                                    dtype=np.float32))
    # 3) 脚本闭环重抓（PPGrasp 30 步）
    ctrl = SKILLS[SCENE]["grasp"](env)
    for _ in range(30):
        obs, *_ = env.step(ctrl.act(obs))
    # 4) 保持抓取 settle
    for _ in range(10):
        obs, *_ = env.step(np.array([0.0, 0.0, 0.0, 1.0],
                                    dtype=np.float32))
    return obs, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+",
                    default=["termdiv_grasp_lift", "termdiv_grasp_lift_ext"])
    ap.add_argument("--pop", type=int, default=64)
    ap.add_argument("--gens", type=int, default=5)
    ap.add_argument("--out", default="ready_poc")
    args = ap.parse_args()

    dp = SkillDP.load(os.path.join(MODEL_DIR, f"dp_{SCENE}.pt"), DEVICE)
    dp.eval()
    norm = load_norm()

    # ---- 合并数据 & 训练 ensemble ----
    states_all, rows_all = [], []
    for tag in args.tags:
        z = np.load(os.path.join(EVAL_DIR, tag + "_states.npz"))
        with open(os.path.join(EVAL_DIR, tag + ".json")) as f:
            rows = json.load(f)["rows"]
        for r, s in zip(rows, z["states"]):
            r["_tag"] = tag
            states_all.append(s)
            rows_all.append(r)
    states_all = np.array(states_all)
    print(f"[poc] merged {len(rows_all)} states from {args.tags}")
    models = train_ensemble(rows_all)
    print(f"[poc] ensemble trained (M={len(models)})")

    fails = [i for i, r in enumerate(rows_all) if r["p_emp"] <= 0.2]
    print(f"[poc] failure states (p_emp<=0.2): {len(fails)}")

    # ---- 五臂 ----
    results = []
    t0 = time.time()
    for fi, idx in enumerate(fails):
        r = rows_all[idx]
        s0 = states_all[idx]
        env = make_env(SCENE, seed=r["env_seed"])
        env.reset()
        row = dict(idx=int(idx), tag=r["_tag"], env_seed=r["env_seed"],
                   p_emp_before=r["p_emp"], grip=r["grip"])
        # 1. direct
        row["direct"] = eval_lift(dp, norm, env, s0, seed0=9000 + fi * 100)
        # 2. random
        t1 = time.time()
        obs_r, a_r = arm_random(dp, norm, env, s0, seed=7000 + fi)
        st_r = save_state(env)
        row["random"] = eval_lift(dp, norm, env, st_r, seed0=9100 + fi * 100)
        row["random_inv"] = bool(invariant_ok(obs_r))
        row["random_cost"] = float(np.mean(np.abs(a_r)))
        row["random_s"] = time.time() - t1
        # 3. re-execute grasp
        t1 = time.time()
        obs_e, _ = arm_reexecute(dp, norm, env, s0, seed=8000 + fi)
        st_e = save_state(env)
        row["reexec"] = eval_lift(dp, norm, env, st_e, seed0=9200 + fi * 100)
        row["reexec_inv"] = bool(invariant_ok(obs_e))
        v_e, _ = ensemble_V(models, obs_e)
        row["reexec_V"] = v_e
        row["reexec_s"] = time.time() - t1
        # 4. READY repair
        t1 = time.time()
        obs_y, a_y = arm_ready(dp, norm, env, s0, models, pop=args.pop,
                               gens=args.gens, seed=6000 + fi)
        st_y = save_state(env)
        row["ready"] = eval_lift(dp, norm, env, st_y, seed0=9300 + fi * 100)
        row["ready_inv"] = bool(invariant_ok(obs_y))
        v_y, sd_y = ensemble_V(models, obs_y)
        row["ready_V"], row["ready_Vsd"] = v_y, sd_y
        row["ready_cost"] = float(np.mean(np.abs(a_y)))
        row["ready_s"] = time.time() - t1
        # 5. oracle regrasp
        t1 = time.time()
        obs_o, _ = arm_oracle(dp, norm, env, s0)
        st_o = save_state(env)
        row["oracle"] = eval_lift(dp, norm, env, st_o, seed0=9400 + fi * 100)
        row["oracle_inv"] = bool(invariant_ok(obs_o))
        row["oracle_s"] = time.time() - t1
        env.close()
        results.append(row)
        print(f"[poc] {fi + 1}/{len(fails)} grip={r['grip']:.3f}  "
              f"direct={row['direct']:.1f} rand={row['random']:.1f} "
              f"reexec={row['reexec']:.1f} ready={row['ready']:.1f} "
              f"oracle={row['oracle']:.1f}  "
              f"({time.time() - t0:.0f}s)")

    # ---- 汇总 ----
    def arm_stats(k):
        v = np.array([r[k] for r in results])
        return dict(mean=float(v.mean()), n=len(v))

    inv_k = [k for k in ("random", "reexec", "ready", "oracle")]
    summary = dict(
        n_fail=len(results),
        arms={k: arm_stats(k) for k in
              ("direct", "random", "reexec", "ready", "oracle")},
        invariant_keep={k: float(np.mean([r[f"{k}_inv"] for r in results]))
                        for k in inv_k},
        ready_vs_reexec=float(np.mean([r["ready"] - r["reexec"]
                                       for r in results])),
        ready_vs_random=float(np.mean([r["ready"] - r["random"]
                                       for r in results])),
        wall_s=time.time() - t0,
    )
    print("\n===== Phase 1 PoC 汇总 =====")
    for k in ("direct", "random", "reexec", "ready", "oracle"):
        print(f"  {k:<8}: P_lift = {summary['arms'][k]['mean']:.3f}")
    print(f"  不变量保持率: {summary['invariant_keep']}")
    print(f"  READY - reexecute = {summary['ready_vs_reexec']:+.3f}")
    print(f"  READY - random    = {summary['ready_vs_random']:+.3f}")
    path = os.path.join(EVAL_DIR, args.out + ".json")
    with open(path, "w") as f:
        json.dump(dict(args=vars(args), summary=summary, rows=results),
                  f, indent=2)
    print(f"[poc] saved {path}")


if __name__ == "__main__":
    main()
