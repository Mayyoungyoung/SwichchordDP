"""汇总评测 JSON, 输出主结果表(成功率/能量/NFE/OOS/Lipschitz 理论闭环)。"""
import argparse
import glob
import json
import os

import numpy as np

EVAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "../../results/metaworld/eval")


def load(path):
    with open(path) as f:
        return json.load(f)


def summarize(scene="pick-place-v3", tag_filter=""):
    rows = []
    for path in sorted(glob.glob(os.path.join(EVAL_DIR, "*.json"))):
        name = os.path.basename(path).replace(".json", "")
        if scene not in name:
            continue
        if tag_filter and tag_filter not in name:
            continue
        data = load(path)
        if "results" not in data:
            continue
        for r in data["results"]:
            eps = r["episodes"]
            e2e = np.mean([e["e2e"] for e in eps])
            energy = np.mean([e["energy"] for e in eps])
            nfe = np.mean([e["nfe"] for e in eps])
            oos = np.mean([e.get("oos", 0.0) for e in eps])
            lips = {}
            for e in eps:
                for l in e.get("lips", []):
                    lips.setdefault(l["pair"], []).append(
                        max(l["L_from"], l["L_to"]))
            lips_avg = {k: float(np.mean(v)) for k, v in lips.items()}
            rows.append(dict(method=name, seq="->".join(r["seq"]),
                             kind=r.get("kind", "?"), e2e=float(e2e),
                             energy=float(energy), nfe=float(nfe),
                             oos=float(oos), lips=lips_avg,
                             per_skill=r.get("per_skill", {})))
    return rows


def print_table(rows):
    print(f"{'method':<42} {'seq':<32} {'kind':<7} {'e2e':>5} {'energy':>7} {'nfe':>5} {'oos':>5}")
    for r in rows:
        print(f"{r['method']:<42} {r['seq']:<32} {r['kind']:<7} "
              f"{r['e2e']:>5.2f} {r['energy']:>7.3f} {r['nfe']:>5.0f} {r['oos']:>5.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="pick-place-v3")
    ap.add_argument("--filter", default="")
    args = ap.parse_args()
    rows = summarize(args.scene, args.filter)
    print_table(rows)
    out = os.path.join(EVAL_DIR, f"summary_{args.scene}.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
