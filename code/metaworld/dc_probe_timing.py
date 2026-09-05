"""预言实验：Successor-Aware Early Handoff 的时序可行性（离线重放）。

假说（§13.5 因果链）：DP grasp 的劣质终态是"过度执行"产物——grip 在 30 步
预算后期从正常 0.44 被过度压到 0.29。因此离线可检验的预言：
  H1. 失败轨迹（y=0）的 V_lift(s_t) 峰值出现在执行中段 t*<30，
      且终态 V 显著低于峰值（切换过晚损失了可行性）；
  H2. 成功轨迹（y=1）的 V 峰值在末端附近 → early handoff 对其无害；
  H3. 过度闭合（grip 下降）确实发生在后期（机制之锚）。

检验量：
  - 每组轨迹的平均 V 曲线形态（按执行步）
  - t* = argmax_t V(s_t) 的分布（失败组 vs 成功组）
  - delta = V_peak - V_after_settle（失败组应显著 > 0）
  - "本可挽救"统计：V_peak >= 0.5 且终态 V < 0.5 的失败轨迹数

输入: results/metaworld/eval/dc_grasp_lift.h5（dc_collect.py 产物）
      results/metaworld/models/fb_pick-place-v3_v2.pt
输出: results/metaworld/eval/dc_timing_probe.json
"""
import argparse
import json
import os

import h5py
import numpy as np
import torch

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from swdp.success_model import featurize, load as fb_load  # noqa: E402
from skills import parse_pp  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "../../results/metaworld/models")
EVAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "../../results/metaworld/eval")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(
        EVAL_DIR, "dc_grasp_lift.h5"))
    ap.add_argument("--fb", default=os.path.join(
        MODEL_DIR, "fb_pick-place-v3_v2.pt"))
    ap.add_argument("--tau_hi", type=float, default=0.5,
                    help="V 阈值: V>=tau 视为 lift 可接管")
    ap.add_argument("--out", default="dc_timing_probe")
    args = ap.parse_args()

    fb, fb_skills = fb_load(args.fb, DEVICE)
    sid_lift = fb_skills.index("lift")
    onehot = np.zeros(len(fb_skills), dtype=np.float32)
    onehot[sid_lift] = 1.0

    with h5py.File(args.data, "r") as f:
        obs = f["obs"][:]
        traj = f["traj_id"][:]
        y = f["y"][:]
        obs_t = f["obs_t"][:]
        n_traj = len(y)

    def v_of(o):
        p = parse_pp(o)
        return float(fb.predict(featurize(
            p["hand"], float(p["grip"]), p["puck"], p["goal"]), onehot))

    # 逐轨迹滚动计算 V 曲线(执行步 1..30) + grip 曲线 + settle 后终态 V
    V_curves, G_curves, V_t, t_stars = [], [], [], []
    for ep in range(n_traj):
        idx = np.where(traj == ep)[0]
        # 每条轨迹约 30 步; 以 ep 内的索引为时间轴
        vc = np.array([v_of(o) for o in obs[idx]], dtype=np.float32)
        gc = np.array([float(parse_pp(o)["grip"]) for o in obs[idx]],
                      dtype=np.float32)
        V_curves.append(vc)
        G_curves.append(gc)
        V_t.append(v_of(obs_t[ep]))
        t_stars.append(int(np.argmax(vc)) + 1)

    V_curves = np.array(V_curves)          # (n_traj, 30)
    G_curves = np.array(G_curves)
    V_t = np.array(V_t)
    t_stars = np.array(t_stars)
    V_peak = V_curves.max(axis=1)

    grp_fail = y == 0
    grp_part = (y > 0) & (y < 1)
    grp_succ = y == 1.0

    def group_stats(mask, name):
        if not mask.any():
            return None
        return dict(
            name=name, n=int(mask.sum()),
            V_curve_mean=[float(v) for v in V_curves[mask].mean(axis=0)],
            grip_curve_mean=[float(v) for v in G_curves[mask].mean(axis=0)],
            t_star_mean=float(t_stars[mask].mean()),
            t_star_early=float((t_stars[mask] < 30).mean()),
            V_peak_mean=float(V_peak[mask].mean()),
            V_settle_mean=float(V_t[mask].mean()),
            delta_mean=float((V_peak - V_t)[mask].mean()),
            rescurable=int(((V_peak >= args.tau_hi)
                            & (V_t < args.tau_hi))[mask].sum()))

    out = dict(
        args=vars(args),
        total=dict(n_traj=int(n_traj),
                   y_pos_rate=float(y.mean())),
        groups={})
    for mask, name in ((grp_fail, "y=0 全败"),
                       (grp_part, "0<y<1 部分"),
                       (grp_succ, "y=1 全成")):
        st = group_stats(mask, name)
        if st:
            out["groups"][name] = st
    # 检验量摘要
    fa = out["groups"].get("y=0 全败")
    su = out["groups"].get("y=1 全成")
    out["verdict"] = {
        "h1_tstar_early_fail": fa["t_star_early"] if fa else None,
        "h1_delta_fail": fa["delta_mean"] if fa else None,
        "h2_tstar_early_succ": su["t_star_early"] if su else None,
        "rescurable_fail": (fa["rescurable"] if fa else None,
                            fa["n"] if fa else None)}

    # ---- 追加分析 1: 逐步判别能力(哪个时间窗的 V 与最终失败相关) ----

    def spearman(a, b):
        ra = np.argsort(np.argsort(a)).astype(np.float64)
        rb = np.argsort(np.argsort(b)).astype(np.float64)
        ra -= ra.mean()
        rb -= rb.mean()
        denom = np.sqrt((ra ** 2).sum()) * np.sqrt((rb ** 2).sum())
        return float((ra * rb).sum()) / denom if denom > 0 else float("nan")

    y_bin = (y < 1.0).astype(np.float32)   # 1=失败(含部分), 0=全成
    T = V_curves.shape[1]
    per_step = []
    for t in range(T):
        v = V_curves[:, t]
        vf = float(v[y_bin == 1].mean()) if (y_bin == 1).any() else float("nan")
        vs = float(v[y_bin == 0].mean())
        per_step.append(dict(t=int(t + 1), spearman=spearman(v, y_bin),
                             V_fail=vf, V_succ=vs,
                             gap=vf - vs if not np.isnan(vf) else float("nan")))
    out["per_step_discrimination"] = per_step
    # ---- 追加分析 2: Stop-at-Edge 沙盘模拟 ----
    # 协议: V 跌破 tau 即停(用上一步观测), 否则跑满 30 步。
    # 沙盘只能测"停在了哪里", 真实成败需环境内验证(见 dc_eval 改造)。
    stop_sim = {}
    for tau in (0.45, 0.5, 0.55, 0.6):
        stop_at = np.full(n_traj, 30, dtype=np.int64)
        for ep in range(n_traj):
            vc = V_curves[ep]
            below = np.where(vc < tau)[0]
            if len(below):
                first = below[0]
                stop_at[ep] = max(first, 1)   # 停在跌破前一步(1-based)
        stop_sim[str(tau)] = dict(
            stop_mean=float(stop_at.mean()),
            stop_early_all=float((stop_at < 30).mean()),
            stop_early_fail=float(stop_at[y_bin == 1].mean())
            if (y_bin == 1).any() else float("nan"),
            stop_early_succ=float(stop_at[y_bin == 0].mean()))
    out["stop_sim"] = stop_sim

    # ---- 追加分析 3: Stop-at-Success 沙盘(语义成功即停, 揃掉过度执行段) ----
    # grasp 谓词(与 diag_handoff.diag_success 一致, 纯 obs 可算):
    #   G(t) = grip<0.75 且 hand-puck xy<0.04
    def grasp_success_mask(obs_rows):
        hand = obs_rows[:, :3]
        grip = obs_rows[:, 3]
        puck = obs_rows[:, 4:7]
        hp = np.linalg.norm(hand[:, :2] - puck[:, :2], axis=1)
        return (grip < 0.75) & (hp < 0.04)

    succ_at = np.full(n_traj, 30, dtype=np.int64)   # 首次语义成功步(1-based)
    grip_at_succ, hp_at_succ, v_at_succ = [], [], []
    all_succ_times = []
    for ep in range(n_traj):
        m = grasp_success_mask(obs[traj == ep])
        ts = np.where(m)[0]
        if len(ts):
            t0 = int(ts[0]) + 1
            succ_at[ep] = t0
            g_at = G_curves[ep][ts[0]]   # 成功时刻 grip
            o = parse_pp(obs[traj == ep][ts[0]])
            hp_at = float(np.linalg.norm(
                np.asarray(o["hand"][:2]) - np.asarray(o["puck"][:2])))
        else:
            g_at, hp_at = float("nan"), float("nan")
        grip_at_succ.append(g_at)
        hp_at_succ.append(hp_at)
        v_at_succ.append(V_curves[ep][succ_at[ep] - 1])
        all_succ_times.append(succ_at[ep])
    grip_at_succ = np.array(grip_at_succ)
    hp_at_succ = np.array(hp_at_succ)
    v_at_succ = np.array(v_at_succ)
    succ_at = np.array(succ_at)

    def mask_stat(mask, name):
        if not mask.any():
            return None
        idxs = np.where(mask)[0]
        grip_t = np.array([G_curves[i][-1] for i in idxs])
        gs = grip_at_succ[mask]
        return dict(
            name=name, n=int(mask.sum()),
            succ_at_mean=float(succ_at[mask].mean()),
            overexec_mean=float((30 - succ_at)[mask].mean()),
            grip_at_succ_mean=float(np.nanmean(gs)),
            grip_terminal_mean=float(grip_t.mean()),
            grip_drop=float(grip_t.mean() - np.nanmean(gs)),
            hp_at_succ_mean=float(np.nanmean(hp_at_succ[mask])),
            v_at_succ_mean=float(np.nanmean(v_at_succ[mask])),
            # 成功时刻 grip 在良好区间[0.40,0.48]的轨迹数(良好终态判定, §13.5)
            good_at_succ=int(np.nansum((gs >= 0.40) & (gs <= 0.48))),
            bad_terminal=int(np.sum(
                (grip_t < 0.40) | (grip_t > 0.48))))
    out["stop_at_success"] = {}
    for mask, name in ((grp_fail, "y=0 全败"),
                       (grp_part, "0<y<1 部分"),
                       (grp_succ, "y=1 全成")):
        st = mask_stat(mask, name)
        if st:
            out["stop_at_success"][name] = st

    # ---- 追加分析 4: Stop-at-Stable-Grasp(首次进入良好夹持区即停) ----
    # 协议: grip 首次跨入稳定夹持带(<=0.50)时停止继续加压, 交接下游。
    # 检验: 失败之源是「后期加压」还是「早期失偏」——比较 t_good 时刻的
    # 偏心 hp 与终态 hp: 若失败组在 t_good 已偏, 时序控制救不了偏心成分。
    def hp_row(o):
        p = parse_pp(o)
        return float(np.linalg.norm(
            np.asarray(p["hand"][:2]) - np.asarray(p["puck"][:2])))

    t_good_arr, hp_good, grip_good, hp_term = [], [], [], []
    for ep in range(n_traj):
        ose = obs[traj == ep]
        gc = G_curves[ep]
        below = np.where(gc <= 0.50)[0]
        if len(below):
            tg = int(below[0]) + 1
        else:
            tg = len(gc)
        t_good_arr.append(tg)
        hp_good.append(hp_row(ose[tg - 1]))
        grip_good.append(float(gc[tg - 1]))
        hp_term.append(hp_row(ose[-1]))
    t_good_arr = np.array(t_good_arr)
    hp_good = np.array(hp_good)
    grip_good = np.array(grip_good)
    hp_term = np.array(hp_term)
    out["stop_at_stable"] = {}
    for mask, name in ((grp_fail, "y=0 全败"),
                       (grp_part, "0<y<1 部分"),
                       (grp_succ, "y=1 全成")):
        if not mask.any():
            continue
        out["stop_at_stable"][name] = dict(
            n=int(mask.sum()),
            t_good_mean=float(t_good_arr[mask].mean()),
            grip_good_mean=float(grip_good[mask].mean()),
            hp_good_mean=float(hp_good[mask].mean()),
            hp_terminal_mean=float(hp_term[mask].mean()),
            hp_growth=float((hp_term - hp_good)[mask].mean()))

    path = os.path.join(EVAL_DIR, args.out + ".json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2)[:4000])
    print(f"[probe] saved {path}")


if __name__ == "__main__":
    main()