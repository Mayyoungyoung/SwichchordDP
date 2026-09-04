"""四臂评测配对分析: naive / chord / chord+random-cand / chord+F_B-cand。

同种子配对(seed=ep*7+1, 同链同回合), McNemar 精确检验(二项)。
用法:
    python arm_analysis.py --chord <json> --naive <json> \
        [--random <json>] [--fb <json>] [--criterion <json>]
每臂 JSON 取每链前 n_episodes 回合(ep 索引对齐 -> 配对)。
"""
import argparse
import json

import numpy as np
from scipy import stats


def load_arm(path, n_eps=None):
    """返回 {(tuple(seq), ep): e2e}。"""
    with open(path) as f:
        d = json.load(f)
    out = {}
    for r in d["results"]:
        seq = tuple(r["seq"])
        for ep, e in enumerate(r["episodes"]):
            if n_eps is None or ep < n_eps:
                out[(seq, ep)] = float(e["e2e"])
    return out


def mcnemar_exact(x, y):
    """配对二元结果 McNemar 精确检验。返回 (b, c, p)。"""
    b = int(np.sum((x == 1) & (y == 0)))   # x 成功 y 失败
    c = int(np.sum((x == 0) & (y == 1)))
    n = b + c
    if n == 0:
        return b, c, 1.0
    p = stats.binomtest(b, n, 0.5).pvalue
    return b, c, float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chord", required=True)
    ap.add_argument("--naive", required=True)
    ap.add_argument("--random", default="")
    ap.add_argument("--fb", default="")
    ap.add_argument("--criterion", default="")
    ap.add_argument("--n_eps", type=int, default=24)
    args = ap.parse_args()

    arms = {"chord": load_arm(args.chord, args.n_eps),
            "naive": load_arm(args.naive, args.n_eps)}
    for k in ("random", "fb", "criterion"):
        if getattr(args, k):
            arms[k] = load_arm(getattr(args, k), args.n_eps)

    keys = sorted(set.intersection(*(set(a) for a in arms.values())),
                  key=lambda k: (k[0], k[1]))
    print(f"[arms] 配对回合数: {len(keys)} (每臂 {len(arms['chord'])})")
    chains = sorted({k[0] for k in keys})

    # ---- 总表 ----
    print("\n== 各臂总体与分链成功率 ==")
    header = f"{'chain':<48}" + "".join(f"{k:>10}" for k in arms)
    print(header)
    for ch in chains:
        row = f"{'->'.join(ch) if len(ch) > 1 else ch[0]:<48}"
        for a in arms.values():
            ys = [a[(ch, ep)] for ep in range(args.n_eps) if (ch, ep) in a]
            row += f"{np.mean(ys):>10.3f}"
        print(row)
    row = f"{'ALL (mean over paired eps)':<48}"
    for a in arms.values():
        row += f"{np.mean([a[k] for k in keys]):>10.3f}"
    print(row)

    # ---- 配对检验 ----
    print("\n== 配对检验 (McNemar exact) ==")
    base = np.array([arms["chord"][k] for k in keys])
    pairs = [("chord", "naive")]
    if "fb" in arms:
        pairs += [("fb", "chord"), ("fb", "random")]
    if "criterion" in arms:
        pairs += [("criterion", "chord")]
    for a, b in pairs:
        xa = np.array([arms[a][k] for k in keys])
        n_pos, n_neg, p = mcnemar_exact(xa, base if b == "chord"
                                        else np.array([arms[b][k] for k in keys]))
        rate_a, rate_b = xa.mean(), (base if b == "chord"
                                     else np.array([arms[b][k] for k in keys])).mean()
        print(f"{a:>10} vs {b:<10}: {rate_a:.3f} vs {rate_b:.3f} "
              f"(+{n_pos}/-{n_neg}, p={p:.4f})"
              f"{' *' if p < 0.05 else ''}")

    # ---- 逐回合明细(5 链) ----
    print("\n== 5 链逐回合明细 (ep: 各臂 e2e) ==")
    main_chain = ("reach", "grasp", "lift", "carry", "place")
    for ep in range(args.n_eps):
        k = (main_chain, ep)
        if k not in arms["chord"]:
            continue
        cells = "".join(f"{a[k]:>6.0f}" for a in arms.values())
        print(f"ep{ep:>3}: {cells}")


if __name__ == "__main__":
    main()
