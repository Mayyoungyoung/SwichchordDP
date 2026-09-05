"""Boundary Monitor + Best-of-K 实验矩阵汇总分析。

读取 dc_eval_bm 产出的多个 JSON（各 pair × 种子段），输出:
- 每 (pair, seed) 行: off/on 的 P_target、缺陷率、各技能段成功率、检验 p 值
- 跨种子合并的配对检验(McNemar + Wilcoxon, rows 合并)
- GO/NO-GO 判读:
  GO:   on 的 P_target 显著高于 off(合并 p<0.05) 且上游 per_skill 无显著回退
  NO-GO: 不满足(注明原因)

用法: python analyze_bm.py
输出: results/metaworld/eval/dc_bm_analysis.json + 控制台表格
"""
import argparse
import glob
import json
import math
import os

import numpy as np

EVAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "../../results/metaworld/eval")


def mcnemar_p(on_wins, off_wins):
    n = on_wins + off_wins
    if n == 0:
        return 1.0
    total = 0.0
    for k in range(max(on_wins, off_wins), n + 1):
        total += math.comb(n, k)
    return min(1.0, 2.0 * total * (0.5 ** n))


def wilcoxon_p(a, b):
    try:
        from scipy.stats import wilcoxon
        return float(wilcoxon(a, b).pvalue)
    except Exception:  # noqa: BLE001
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="dc_bm_*_s*.json")
    ap.add_argument("--out", default="dc_bm_analysis")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(EVAL_DIR, args.pattern)))
    runs = []   # (pair, seed0, data)
    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        runs.append((d["pair"], d["args"]["seed0"], d))

    table = []
    merged = {}   # pair -> dict(so, sn, rows_off, rows_on)
    for pair, seed0, d in runs:
        off, on = d["off"], d["on"]
        so = np.array([r["target_succ"] for r in d["rows_off"]])
        sn = np.array([r["target_succ"] for r in d["rows_on"]])
        a = int(((so < 0.5) & (sn >= 0.5)).sum())
        b = int(((so >= 0.5) & (sn < 0.5)).sum())
        row = dict(pair=pair, seed0=seed0,
                   n=len(so),
                   off_target=off["P_target"],
                   on_target=on["P_target"],
                   delta=on["P_target"] - off["P_target"],
                   off_defect=1 - off["P_target"],
                   on_defect=1 - on["P_target"],
                   off_per_skill=off.get("per_skill", {}),
                   on_per_skill=on.get("per_skill", {}),
                   mcnemar=(a, b, mcnemar_p(a, b)),
                   wilcoxon_p=wilcoxon_p(sn, so),
                   attempts_mean=on["attempts_mean"])
        table.append(row)
        m = merged.setdefault(pair, dict(so=[], sn=[]))
        m["so"].extend(so.tolist())
        m["sn"].extend(sn.tolist())

    # 跨种子合并检验
    merged_results = {}
    for pair, m in merged.items():
        so = np.array(m["so"])
        sn = np.array(m["sn"])
        a = int(((so < 0.5) & (sn >= 0.5)).sum())
        b = int(((so >= 0.5) & (sn < 0.5)).sum())
        merged_results[pair] = dict(
            n=len(so),
            off_target=float(so.mean()),
            on_target=float(sn.mean()),
            delta=float(sn.mean() - so.mean()),
            mcnemar=(a, b, mcnemar_p(a, b)),
            wilcoxon_p=wilcoxon_p(sn, so))

    # 上游回退检查: per_skill 里除目标/下游外的技能(前缀技能) on vs off
    def upstream_regress(row, tol=0.03):
        """上游技能(prefix)成功率 on 比 off 低超过 tol 则回退。"""
        ret = {}
        for k, v in row["off_per_skill"].items():
            if k in ("lift_p",) or k == row["pair"].split("2g")[0]:
                continue
            v_on = row["on_per_skill"].get(k, float("nan"))
            if not math.isnan(v_on) and v_on < v - tol:
                ret[k] = (v, v_on)
        return ret

    verdict = {}
    for pair, mr in merged_results.items():
        p_m = mr["mcnemar"][2]
        p_w = mr["wilcoxon_p"]
        sig = (p_m < 0.05) or (p_w < 0.05)
        reg = [upstream_regress(r) for r in table if r["pair"] == pair]
        reg_all = {}
        for r in reg:
            reg_all.update(r)
        verdict[pair] = dict(
            significant=bool(sig),
            p_mcnemar=p_m, p_wilcoxon=p_w,
            delta=mr["delta"],
            upstream_regress=reg_all,
            go=bool(sig and not reg_all))

    n_go = sum(1 for v in verdict.values() if v["go"])
    overall = dict(
        n_pairs=len(verdict), n_go=n_go,
        go=(n_go >= 2) and (n_go == len(verdict) or n_go >= 2),
        note=("GO: 多数边界显著且上游无回退" if n_go >= 2
              else "NO-GO: 显著性/回退未达标, 见 verdict 明细"))

    out = dict(args=vars(args), runs=table, merged=merged_results,
               verdict=verdict, overall=overall)
    path = os.path.join(EVAL_DIR, args.out + ".json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # 控制台表格
    print(f"{'pair':6s} {'seed':>5s} {'n':>4s} {'off':>7s} {'on':>7s} "
          f"{'delta':>7s} {'McNemar(on/off,p)':>18s} {'Wilcoxon p':>10s}")
    for r in table:
        print(f"{r['pair']:6s} {r['seed0']:5d} {r['n']:4d} "
              f"{r['off_target']:7.4f} {r['on_target']:7.4f} "
              f"{r['delta']:+7.4f} "
              f"{r['mcnemar'][0]}/{r['mcnemar'][1]} {r['mcnemar'][2]:.4f} "
              f"{r['wilcoxon_p']:10.4f}")
    print()
    print("=== 跨种子合并 ===")
    for pair, mr in merged_results.items():
        print(f"{pair:6s} n={mr['n']:4d} off={mr['off_target']:.4f} "
              f"on={mr['on_target']:.4f} delta={mr['delta']:+.4f} "
              f"McNemar p={mr['mcnemar'][2]:.4f} "
              f"Wilcoxon p={mr['wilcoxon_p']:.4f} "
              f"-> {'GO' if verdict[pair]['go'] else 'NO-GO'}")
    print()
    print(f"OVERALL: {overall['note']} (go pairs {n_go}/{len(verdict)})")
    print(f"[analyze-bm] saved {path}")


if __name__ == "__main__":
    main()