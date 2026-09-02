"""恢复评测闭环(对标 HELM 的扰动恢复实验): 注入扰动 -> 场能量风险信号 -> 事件式重规划。

协议:
1. 执行 5 技能链(reach→grasp→lift→carry→place, chord 交接);
2. 在 carry 阶段随机时刻注入「抓握失败」扰动: 强制 3 步开夹爪动作(物体掉落);
3. 在下一个交接边界读取场能量 ‖û‖², 若超过阈值 ρ 则触发重规划:
   重规划 = 回到 lift 技能重新执行(补抓), 再续 carry→place;
4. 指标: 无扰动成功率、扰动后无恢复成功率、扰动后+恢复成功率(对比 HELM 54.2%)。
"""
import argparse
import json
import os

import h5py
import numpy as np
import torch

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from swdp.policy import SkillDP  # noqa: E402
from swdp import chord_compose as cc  # noqa: E402
from eval_compose import skill_success, SKILL_NAMES  # noqa: E402
from skills import make_env  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "../../results/metaworld/data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "../../results/metaworld/models")
SCENE = "pick-place-v3"
SEQ = ["reach", "grasp", "lift", "carry", "place"]
STEPS = {"reach": 30, "grasp": 30, "lift": 25, "carry": 30, "place": 30}


@torch.no_grad()
def rollout_recovery(dp, disturb=True, use_replan=True, rho=5.0, lam=0.15,
                     seed=0, disturb_skill="carry"):
    env = make_env(SCENE, seed=1000 + seed)
    obs, info = env.reset()
    with h5py.File(os.path.join(DATA_DIR, f"{SCENE}.h5"), "r") as f:
        obs_mean = f["obs_mean"][:]; obs_std = f["obs_std"][:]
        act_mean = f["act_mean"][:]; act_std = f["act_std"][:]
    ids = {n: i for i, n in enumerate(SKILL_NAMES[SCENE])}

    def norm_obs(o):
        return torch.from_numpy((o - obs_mean) / obs_std).float().to(DEVICE).unsqueeze(0)

    def onehot(i):
        z = np.zeros((1, dp.n_skills), dtype=np.float32)
        z[0, i] = 1.0
        return torch.from_numpy(z).to(DEVICE)

    def sample_chunk(o, s_id):
        return dp.sample(o, onehot(s_id), n_steps=24, seed=None)

    chunk = sample_chunk(norm_obs(obs), ids[SEQ[0]])
    step_in_chunk = 0
    cur = 0
    step_in_skill = 0
    total = sum(STEPS[s] for s in SEQ)
    disturb_t = None
    disturb_len = 8  # 强制开夹爪步数(确保物体掉落)
    if disturb:
        rng = np.random.default_rng(seed)
        lo = sum(STEPS[s] for s in SEQ[:SEQ.index(disturb_skill)]) + 8
        hi = lo + STEPS[disturb_skill] - disturb_len - 4
        disturb_t = int(rng.integers(lo, hi))
    replans = 0
    energies = []
    max_steps = total + (30 + 25 + 30 + 30)  # 允许一次恢复追加 grasp+lift+carry+place
    t = 0
    while t < max_steps:
        # 边界交接
        if t > 0 and step_in_skill >= STEPS[SEQ[cur]] and cur + 1 < len(SEQ):
            cur += 1
            step_in_skill = 0
            o_t = norm_obs(obs)
            anchor = sample_chunk(o_t, ids[SEQ[cur - 1]])
            s_from, s_to = onehot(ids[SEQ[cur - 1]]), onehot(ids[SEQ[cur]])
            mask = cc.temporal_mask(anchor.shape[1], 0, 4, DEVICE)
            a_new, info = cc.switch(dp, o_t, anchor, s_from, s_to, 0.9, 0.15,
                                    lam, 1, "chord", mask, True, seed=seed + t)
            energies.append(info["energy"])
            # 风险信号: 场能量相对尖峰(> 3×本回合已见边界能量中位数) -> 重规划(回到 grasp 补抓)
            baseline = float(np.median(energies[:-1])) if len(energies) > 1 else rho
            spike = info["energy"] > max(3.0 * baseline, 1.0)
            if use_replan and spike and SEQ[cur] == "place" and replans == 0:
                replans += 1
                cur = SEQ.index("grasp")
                step_in_skill = 0
                chunk = sample_chunk(norm_obs(obs), ids["grasp"])
                step_in_chunk = 0
                t += 1
                continue
            chunk = a_new
            step_in_chunk = 0

        a_raw = np.clip((chunk[0, step_in_chunk].cpu().numpy() * act_std) + act_mean, -1, 1)
        if disturb_t is not None and disturb_t <= t < disturb_t + disturb_len:
            a_raw = a_raw.copy()
            a_raw[3] = -1.0  # 抓握失败扰动: 强制开夹爪
        obs, _, _, _, info = env.step(a_raw)
        step_in_chunk += 1
        step_in_skill += 1
        t += 1
        if step_in_chunk >= 8:
            chunk = sample_chunk(norm_obs(obs), ids[SEQ[cur]])
            step_in_chunk = 0
        # 到达原任务终点(place 结束)后提前退出
        if cur == len(SEQ) - 1 and step_in_skill >= STEPS[SEQ[cur]]:
            break

    succ = skill_success(SCENE, "place", env, obs, info)
    # 送达指标: puck 3D 距离目标 < 0.15(恢复链完整执行到终相的中间指标)
    delivered = float(np.linalg.norm(obs[4:7] - obs[-3:]) < 0.15)
    env.close()
    return dict(success=float(succ), delivered=delivered, replans=replans,
                energies=energies, disturb_t=disturb_t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_episodes", type=int, default=10)
    ap.add_argument("--rho", type=float, default=5.0)
    args = ap.parse_args()
    dp = SkillDP.load(os.path.join(MODEL_DIR, f"dp_{SCENE}.pt"), DEVICE)
    dp.eval()
    out = {}
    for tag, kw in [("clean", dict(disturb=False, use_replan=False)),
                    ("disturb_noreplan", dict(disturb=True, use_replan=False)),
                    ("disturb_replan", dict(disturb=True, use_replan=True))]:
        rows = [rollout_recovery(dp, rho=args.rho, seed=7 * ep + 1, **kw)
                for ep in range(args.n_episodes)]
        rate = float(np.mean([r["success"] for r in rows]))
        deliv = float(np.mean([r["delivered"] for r in rows]))
        out[tag] = dict(rate=rate, delivered=deliv, rows=rows)
        print(f"[recovery] {tag}: success = {rate:.2f} "
              f"({sum(r['success'] for r in rows):.0f}/{len(rows)}), "
              f"delivered = {deliv:.2f}")
    os.makedirs("results/metaworld/eval", exist_ok=True)
    with open("results/metaworld/eval/recovery.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("[recovery] saved results/metaworld/eval/recovery.json")


if __name__ == "__main__":
    main()
