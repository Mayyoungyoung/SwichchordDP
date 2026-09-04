"""Downstream-Compatible Skill Learning 结果分析: 配对显著性 + Go/No-Go 判读。

读取 dc_eval.py 输出的 JSON(含逐回合 rows), 计算:
1. 各模型聚合指标表(P(grasp) / P(lift|grasp) / e2e / F_B / grip);
2. base vs 各变体的配对检验:
   - lift_p 连续值: Wilcoxon 符号秩检验(仅 grasp 成功回合, 配对种子);
   - e2e 二值化(lift_p>0.5): McNemar 精确检验;
   - grip 分布: 双样本 KS 检验(终态分布是否被塑造);
3. 按 paper_plan_READY.md §5 的 Go/No-Go 判据输出结论:
   outcome 加权使 P(e2e) 显著上升且 P(grasp) 无明显下降 -> 升级 fb 连续加权;
   否则此路价值有限, 回到「现象+检测」的 empirical study。

用法: python analyze_dc.py [--data results/metaworld/eval/dc_eval.json]
"""
import argparse
import json
import os

import numpy as np
from scipy.stats import binomtest, wilcoxon, ks_2samp

EVAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "../../results/metaworld/eval")


def mcnemar(a, b):
    """配对二值序列 McNemar 精确检验 -> (a胜b, b胜a, 双尾p)。"""
    a, b = np.asarray(a) > 0.5, np.asarray(b) > 0.5
    n_ab = int((a & ~b).sum())
    n_ba = int((~a & b).sum())
    n = n_ab + n_ba
    if n == 0:
        return n_ab, n_ba, 1.0
    p = min(1.0, 2.0 * binomtest(min(n_ab, n_ba), n, 0.5).pvalue)
    return n_ab, n_ba, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(EVAL_DIR, "dc_eval.json"))
    args = ap.parse_args()

    with open(args.data) as f:
        d = json.load(f)
    results, rows = d["results"], d["rows"]
    names = list(rows.keys())

    # 1. 聚合表
    print("=" * 78)
    print(f"{'model':<20}{'grasp':>8}{'lift|g':>9}{'e2e':>9}{'fb':>8}"
          f"{'grip m±s':>14}")
    for n in names:
        r = results[n]
        print(f"{n:<20}{r['grasp_rate']:>8.3f}{r['lift_cond']:>9.3f}"
              f"{r['e2e']:>9.3f}{r['fb_mean']:>8.3f}"
              f"{r['grip']['mean']:>7.3f}±{r['grip']['std']:.3f}")

    # 2. 配对检验(base vs 各变体)
    print("-" * 78)
    print("配对检验 (vs base, 同种子配对):")
    base_rows = rows["base"]
    base_lift = np.array([r["lift_p"] for r in base_rows
                          if r["grasp_succ"] > 0.5])
    base_grip = np.array([r["grip"] for r in base_rows])
    base_e2e = np.array([r["e2e"] for r in base_rows])
    for n in names:
        if n == "base":
            continue
        rr = rows[n]
        lift = np.array([r["lift_p"] for r in rr if r["grasp_succ"] > 0.5])
        grip = np.array([r["grip"] for r in rr])
        e2e = np.array([r["e2e"] for r in rr])
        # Wilcoxon 符号秩(lift_p 连续值, 配对)
        w = wilcoxon(lift, base_lift, alternative="greater")
        # McNemar(e2e 二值化口径)
        nab, nba, pm = mcnemar(e2e, base_e2e)
        # KS(grip 分布)
        ks = ks_2samp(grip, base_grip)
        print(f"{n:<20} Wilcoxon(lift) p={w.pvalue:.4f} | "
              f"McNemar(e2e) {nab}-{nba} p={pm:.4f} | "
              f"KS(grip) p={ks.pvalue:.4f}")

    # 3. Go/No-Go 判读
    print("=" * 78)
    r_base = results["base"]
    out = {n: results[n] for n in names if n.startswith("dc_outcome")}
    best_n = max(out, key=lambda n: out[n]["e2e"])
    de = out[best_n]["e2e"] - r_base["e2e"]
    dg = out[best_n]["grasp_rate"] - r_base["grasp_rate"]
    # e2e 口径的 McNemar(base vs 最优 outcome 变体)
    nab, nba, pm = mcnemar(np.array([r["e2e"] for r in rows[best_n]]),
                           np.array([r["e2e"] for r in rows["base"]]))
    n = r_base["n"]
    se = np.sqrt(r_base["e2e"] * (1 - r_base["e2e"]) / n)
    print(f"最优 outcome 变体: {best_n}, Δe2e={de:+.4f} "
          f"(base SE≈{se:.4f}), Δgrasp={dg:+.3f}")
    print(f"McNemar(e2e 二值): {nab}-{nba}, p={pm:.4f}")
    if de > 0 and pm < 0.05 and dg >= -0.02:
        print("判读: GO -> 升级 F_B 连续加权 + 泛化到第二对")
    else:
        print("判读: NO-GO(未达显著) -> 按 §5 此路价值有限, "
              "建议: (a) DP 链 setup 扩大劣质抓取样本再测, "
              "或 (b) 回到现象+检测 empirical study")


if __name__ == "__main__":
    main()
