"""多对边界传导分析（全链数据, 免标签）。

对 dc_collect_fullchain.h5 的三个完整边界逐边检验「入口定型 + 动力学放大」:
  P1. reach→grasp: 入口 hand-puck xy(hp); 终态 hp/grip(夹持质量)
  P2. grasp→lift:  入口 hp/grip; 终态 puck_z(抬起高度)/hp(携物稳度)
  P3. lift→carry:  入口 puck_z/hp; 终态 puck-goal xy(pg, 目标对中)

每对输出:
  - 缺陷率(终态质量低于阈值) 与 质量标签定义
  - 入口特征 缺陷组 vs 良好组(比值 = 入口传导强度)
  - 逐步分叉曲线: B 执行期间两组主特征差(第几步开始显著扩大)
  - corr(入口, 终态)
  - 入口 AUC(入口特征判别缺陷, 传导强度量度; 注意终态自证仅限 0.98 级)

用法: python dc_probe_pairs.py
输出: results/metaworld/eval/dc_pairs_probe.json
"""
import argparse
import json
import os

import h5py
import numpy as np

EVAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "../../results/metaworld/eval")
SID = {"reach": 0, "grasp": 1, "lift": 2, "carry": 3, "place": 4}


def geom(row):
    """obs 39 维 -> 几何量 dict(hand/puck/goal 均为 np 数组)。"""
    hand = row[:3].astype(np.float64)
    grip = float(row[3])
    puck = row[4:7].astype(np.float64)
    goal = row[-3:].astype(np.float64)
    return dict(
        hp=float(np.linalg.norm(hand[:2] - puck[:2])),
        hg=float(np.linalg.norm(hand[:2] - goal[:2])),
        pg=float(np.linalg.norm(puck[:2] - goal[:2])),
        puck_z=float(puck[2]),
        grip=grip)


def auc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    for v in np.unique(scores):
        m = scores == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    n1, n0 = labels.sum(), (1 - labels).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(
        EVAL_DIR, "dc_chain_full.h5"))
    ap.add_argument("--out", default="dc_pairs_probe")
    args = ap.parse_args()

    with h5py.File(args.data, "r") as f:
        obs = f["obs"][:]
        skill = f["skill"][:]
        traj = f["traj_id"][:]
    n_traj = int(traj.max()) + 1

    # 每条轨迹每段的首/末索引
    seg = {k: [] for k in SID}   # seg[sid] = list of (start_idx, end_idx)
    for ep in range(n_traj):
        m = traj == ep
        sk = skill[m]
        obs_ep = obs[m]
        for sid, name in SID.items():
            idx = np.where(sk == sid)[0]
            if len(idx):
                seg[sid].append((idx[0], idx[-1] + 1))

    def seg_obs(sid, ep):
        m = traj == ep
        return obs[m][skill[m] == sid]

    out = {"args": vars(args), "pairs": {}}

    # ---- P1: reach→grasp ----
    entry_hp, term_hp, term_grip = [], [], []
    for ep in range(n_traj):
        g = seg_obs(SID["grasp"], ep)
        e = geom(g[0])
        t = geom(g[-1])
        entry_hp.append(e["hp"])
        term_hp.append(t["hp"])
        term_grip.append(t["grip"])
    entry_hp = np.array(entry_hp)
    term_hp = np.array(term_hp)
    term_grip = np.array(term_grip)
    defective = (term_hp > 0.012) | (term_grip < 0.40)   # 与 §16.2 一致
    out["pairs"]["reach->grasp"] = dict(
        label="defective: term_hp>0.012 or term_grip<0.40",
        defective_rate=float(defective.mean()),
        entry_feat="hp",
        entry_fail=float(entry_hp[defective].mean()),
        entry_succ=float(entry_hp[~defective].mean()),
        entry_ratio=float(entry_hp[defective].mean()
                          / entry_hp[~defective].mean()),
        term_fail=float(term_hp[defective].mean()),
        term_succ=float(term_hp[~defective].mean()),
        corr=float(np.corrcoef(entry_hp, term_hp)[0, 1]),
        entry_auc=auc(entry_hp, defective.astype(float)))

    # ---- P2: grasp→lift ----
    entry_hp, entry_grip, term_z, term_hp = [], [], [], []
    for ep in range(n_traj):
        l = seg_obs(SID["lift"], ep)
        e = geom(l[0])
        t = geom(l[-1])
        entry_hp.append(e["hp"])
        entry_grip.append(e["grip"])
        term_z.append(t["puck_z"])
        term_hp.append(t["hp"])
    entry_hp = np.array(entry_hp)
    term_z = np.array(term_z)
    term_hp = np.array(term_hp)
    defective = (term_z < 0.06) | (term_hp > 0.03)   # 没抬起来或携物松脱
    out["pairs"]["grasp->lift"] = dict(
        label="defective: lift_term puck_z<0.06 or hp>0.03",
        defective_rate=float(defective.mean()),
        entry_feat="hp",
        entry_fail=float(entry_hp[defective].mean()),
        entry_succ=float(entry_hp[~defective].mean()),
        entry_ratio=float(entry_hp[defective].mean()
                          / entry_hp[~defective].mean()),
        term_z_fail=float(term_z[defective].mean()),
        term_z_succ=float(term_z[~defective].mean()),
        corr=float(np.corrcoef(entry_hp, term_z)[0, 1]),
        entry_auc=auc(entry_hp, defective.astype(float)))

    # ---- P3: lift→carry ----
    entry_z, entry_hp, term_pg = [], [], []
    for ep in range(n_traj):
        c = seg_obs(SID["carry"], ep)
        e = geom(c[0])
        t = geom(c[-1])
        entry_z.append(e["puck_z"])
        entry_hp.append(e["hp"])
        term_pg.append(t["pg"])
    entry_z = np.array(entry_z)
    entry_hp = np.array(entry_hp)
    term_pg = np.array(term_pg)
    defective = term_pg > 0.07   # carry 谓词 puck-goal xy<0.07
    out["pairs"]["lift->carry"] = dict(
        label="defective: carry_term puck-goal xy>0.07",
        defective_rate=float(defective.mean()),
        entry_feat="puck_z",
        entry_z_fail=float(entry_z[defective].mean()),
        entry_z_succ=float(entry_z[~defective].mean()),
        entry_hp_fail=float(entry_hp[defective].mean()),
        entry_hp_succ=float(entry_hp[~defective].mean()),
        term_pg_fail=float(term_pg[defective].mean()),
        term_pg_succ=float(term_pg[~defective].mean()),
        corr=float(np.corrcoef(entry_z, term_pg)[0, 1]),
        entry_auc=auc(entry_z, defective.astype(float)))

    # ---- 逐步分叉曲线(每对 B 段内两组主特征差) ----
    def divergence_curve(a_sid, b_sid, feat_fn, defective):
        """B 段逐步: 缺陷组 vs 良好组特征差曲线 + 首显著步。"""
        diffs = []
        T = 30
        for t in range(T):
            vals = []
            for ep in range(n_traj):
                b = seg_obs(b_sid, ep)
                if len(b) > t:
                    vals.append(feat_fn(geom(b[t])))
            vals = np.array(vals)
            d = float(vals[defective].mean() - vals[~defective].mean())
            diffs.append(d)
        return diffs

    out["pairs"]["reach->grasp"]["divergence"] = divergence_curve(
        SID["reach"], SID["grasp"], lambda g: g["hp"],
        out["pairs"]["reach->grasp"]["defective_rate"] > 0 and
        np.array([(geom(seg_obs(SID["grasp"], ep)[-1])["hp"] > 0.012)
                  or (geom(seg_obs(SID["grasp"], ep)[-1])["grip"] < 0.40)
                  for ep in range(n_traj)]))
    g2l_def = np.array(
        [(geom(seg_obs(SID["lift"], ep)[-1])["puck_z"] < 0.06)
         or (geom(seg_obs(SID["lift"], ep)[-1])["hp"] > 0.03)
         for ep in range(n_traj)])
    out["pairs"]["grasp->lift"]["divergence"] = divergence_curve(
        SID["grasp"], SID["lift"], lambda g: g["puck_z"], g2l_def)
    l2c_def = np.array(
        [geom(seg_obs(SID["carry"], ep)[-1])["pg"] > 0.07
         for ep in range(n_traj)])
    out["pairs"]["lift->carry"]["divergence"] = divergence_curve(
        SID["lift"], SID["carry"], lambda g: g["pg"], l2c_def)

    path = os.path.join(EVAL_DIR, args.out + ".json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    # 控制台摘要
    for name, p in out["pairs"].items():
        print(f"=== {name} ===")
        print(f"  缺陷率 {p['defective_rate']:.3f} | 入口比值 "
              f"{p.get('entry_ratio', p.get('entry_z_fail', 0)/max(p.get('entry_z_succ', 1e-9), 1e-9)):.2f}"
              f" | corr(入口,终态) {p['corr']:+.3f} | 入口AUC {p['entry_auc']:.3f}")
        div = p.get("divergence", [])
        if div:
            print(f"  分叉曲线(每5步): {[round(v, 4) for v in div[::5]]}")
    print(f"[probe-pairs] saved {path}")


if __name__ == "__main__":
    main()