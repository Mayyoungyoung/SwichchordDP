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
from swdp.harness import (SkillRuntime, TransitionSpec,  # noqa: E402
                          ChainExecutor, RiskMonitor)
from eval_compose import skill_success, SKILL_NAMES  # noqa: E402
from skills import make_env  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "../../results/metaworld/data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "../../results/metaworld/models")
SCENE = "pick-place-v3"
SEQ = ["reach", "grasp", "lift", "carry", "place"]
STEPS = {"reach": 30, "grasp": 30, "lift": 25, "carry": 30, "place": 40}


@torch.no_grad()
def rollout_recovery(dp, disturb=True, use_replan=True, rho=5.0, lam=0.15,
                     seed=0, disturb_skill="carry", replan_to="grasp"):
    env = make_env(SCENE, seed=1000 + seed)
    obs, info = env.reset()
    with h5py.File(os.path.join(DATA_DIR, f"{SCENE}.h5"), "r") as f:
        norm = (f["obs_mean"][:], f["obs_std"][:],
                f["act_mean"][:], f["act_std"][:])
    ids = {n: i for i, n in enumerate(SKILL_NAMES[SCENE])}
    seq_sid = [ids[s] for s in SEQ]
    steps = [STEPS[s] for s in SEQ]

    rt = SkillRuntime(dp, norm, device=DEVICE, n_ddim=24, resample=8)
    spec = TransitionSpec(mode="chord", lam=lam, use_mask=True,
                          use_proj=True)
    risk = RiskMonitor(k=3.0, floor=1.0, first=rho)

    def on_risk(spike, energy, state):
        # 尖峰且即将进入 place 且未重规划过 -> 回到 replan_to 补抓
        if use_replan and spike and \
                state["next"] == ids["place"] and state["replans"] == 0:
            return ids[replan_to]
        return None

    disturb_t = None
    disturb_len = 8  # 强制开夹爪步数(确保物体掉落)
    if disturb:
        rng = np.random.default_rng(seed)
        lo = sum(STEPS[s] for s in SEQ[:SEQ.index(disturb_skill)]) + 8
        hi = lo + STEPS[disturb_skill] - disturb_len - 4
        disturb_t = int(rng.integers(lo, hi))

    def action_filter(t, a):
        if disturb_t is not None and disturb_t <= t < disturb_t + disturb_len:
            a = a.copy()
            a[3] = -1.0  # 抓握失败扰动: 强制开夹爪
        return a

    def stop_fn(state):
        # 到达原任务终点(place 结束)后提前退出
        return state["cur"] == len(SEQ) - 1 and \
            state["step_in_skill"] >= state["skill_steps"][-1]

    ex = ChainExecutor(rt, spec, on_risk=on_risk, risk_monitor=risk)
    max_steps = sum(steps) + (30 + 25 + 30 + 40)  # 允许一次恢复追加
    out = ex.run(env, obs, seq_sid, steps, seed=seed, max_steps=max_steps,
                 action_filter=action_filter, stop_fn=stop_fn)

    succ = skill_success(SCENE, "place", env, out["obs"], out["info"])
    # 送达指标: puck 3D 距离目标 < 0.15(恢复链完整执行到终相的中间指标)
    delivered = float(np.linalg.norm(out["obs"][4:7] - out["obs"][-3:]) < 0.15)
    env.close()
    return dict(success=float(succ), delivered=delivered,
                replans=out["replans"], energies=out["energies"],
                disturb_t=disturb_t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_episodes", type=int, default=10)
    ap.add_argument("--rho", type=float, default=5.0)
    ap.add_argument("--lam", type=float, default=0.15)
    ap.add_argument("--place_steps", type=int, default=30)
    ap.add_argument("--replan_to", default="grasp",
                    choices=["grasp", "reach"])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    global STEPS
    STEPS["place"] = args.place_steps
    dp = SkillDP.load(os.path.join(MODEL_DIR, f"dp_{SCENE}.pt"), DEVICE)
    dp.eval()
    out = {}
    for tag, kw in [("clean", dict(disturb=False, use_replan=False)),
                    ("disturb_noreplan", dict(disturb=True, use_replan=False)),
                    ("disturb_replan", dict(disturb=True, use_replan=True))]:
        rows = [rollout_recovery(dp, rho=args.rho, lam=args.lam, seed=7 * ep + 1,
                                 replan_to=args.replan_to, **kw)
                for ep in range(args.n_episodes)]
        rate = float(np.mean([r["success"] for r in rows]))
        deliv = float(np.mean([r["delivered"] for r in rows]))
        out[tag] = dict(rate=rate, delivered=deliv, rows=rows)
        print(f"[recovery] {tag}: success = {rate:.2f} "
              f"({sum(r['success'] for r in rows):.0f}/{len(rows)}), "
              f"delivered = {deliv:.2f}")
    os.makedirs("results/metaworld/eval", exist_ok=True)
    name = f"recovery_l{args.lam}_p{args.place_steps}_r{args.replan_to}{args.tag}.json"
    with open(f"results/metaworld/eval/{name}", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[recovery] saved results/metaworld/eval/{name}")


if __name__ == "__main__":
    main()
