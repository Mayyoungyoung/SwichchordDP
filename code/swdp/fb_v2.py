"""F_B v2 修复实验: 诊断 F_B 在自然终态分布上反号(Spearman -0.39)的根因修复。

根因: 诊断训练数据扰动只在 xy 方向, 自然终态的真实成败驱动是 z 高度
(goal z 任务差异, rho=-0.47)。修复: 用 termdiv 自然分布数据(220 状态 × 10
rollouts 的 Bernoulli 计数)重训/混合训练, 以 out-of-fold 评估:
FB_gap(Top20%-Bottom20% mean P_emp)、Spearman、ECE。

用法: python fb_v2.py --mode a|b|both
  a: 只用自然数据重训(2200 Bernoulli 样本)
  b: 诊断数据(752) + 自然数据混合
评估协议: 5 折 CV, out-of-fold 预测计算 FB_gap/Spearman/ECE。
"""
import argparse
import json
import os

import numpy as np
import torch
from scipy import stats

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from swdp.success_model import (SuccessModel, featurize, ece,  # noqa: E402
                                build_dataset, save)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SKILL_NAMES = ["reach", "grasp", "lift", "carry", "place"]
TERMDIV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "../../results/metaworld/eval/termdiv_carry_place.json")
DIAG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "../../results/metaworld/eval/diag_handoff.json")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "../../results/metaworld/models")


def load_termdiv():
    """自然终态数据 -> Bernoulli 展开 (feat, onehot, y) × 10。"""
    with open(TERMDIV) as f:
        rows = json.load(f)["rows"]
    sid = SKILL_NAMES.index("place")
    X, S, y = [], [], []
    for r in rows:
        f = featurize(r["hand"], r["grip"], r["puck"], r["goal"])
        onehot = np.eye(len(SKILL_NAMES))[sid].astype(np.float32)
        n = int(r["n_succ"])
        for kk in range(10):            # 每 rollout 一个 Bernoulli 样本
            X.append(f)
            S.append(onehot)
            y.append(1.0 if kk < n else 0.0)
    return (np.array(X), np.array(S), np.array(y, dtype=np.float32),
            [r["p_emp"] for r in rows])


def load_diag():
    """诊断数据(全部 4 边界, skill one-hot 区分)。"""
    with open(DIAG) as f:
        rows = json.load(f)["rows"]
    X, y, S, sids, kinds = build_dataset(rows, SKILL_NAMES)
    return X, S, y


def train_eval(X, S, y, p_emp_state=None, epochs=300, lr=1e-3, seed=0,
               n_folds=5, tag=""):
    """5 折 CV: out-of-fold 预测 -> FB_gap/Spearman/ECE。

    p_emp_state: 每状态(非 Bernoulli 展开)的真实成功率, 用于按状态聚合评估。
    """
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, n_folds)
    oof = np.zeros(n)
    for va in folds:
        tr = np.setdiff1d(idx, va)
        model = SuccessModel(n_skills=len(SKILL_NAMES)).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        Xt = torch.from_numpy(X[tr]).to(DEVICE)
        St = torch.from_numpy(S[tr]).to(DEVICE)
        yt = torch.from_numpy(y[tr]).to(DEVICE)
        for _ in range(epochs):
            opt.zero_grad()
            logit = model.net(torch.cat([Xt, St], dim=-1)).squeeze(-1)
            torch.nn.functional.binary_cross_entropy_with_logits(
                logit, yt).backward()
            opt.step()
        oof[va] = model.predict(X[va], S[va])
    # 按状态聚合(每状态 10 个 Bernoulli 副本取均值)
    n_state = len(p_emp_state) if p_emp_state is not None else n
    rep = n // n_state
    p_pred = oof.reshape(n_state, rep).mean(axis=1)
    p_true = np.array(p_emp_state)
    kk = max(1, int(n_state * 0.2))
    order_f = np.argsort(p_pred)
    fb_gap = float(p_true[order_f[-kk:]].mean() - p_true[order_f[:kk]].mean())
    sp = stats.spearmanr(p_pred, p_true)
    out = dict(tag=tag, n_state=n_state, fb_gap=fb_gap,
               spearman=float(sp.statistic), spearman_p=float(sp.pvalue),
               ece=ece(p_pred, p_true),
               brier=float(np.mean((p_pred - p_true) ** 2)),
               top_mean=float(p_true[order_f[-kk:]].mean()),
               bottom_mean=float(p_true[order_f[:kk]].mean()))
    print(f"[fbv2 {tag}] n={n_state}  FB_gap={fb_gap:+.3f}  "
          f"Spearman={sp.statistic:+.3f} (p={sp.pvalue:.4f})  "
          f"ECE={out['ece']:.3f}  top={out['top_mean']:.3f} "
          f"bottom={out['bottom_mean']:.3f}")
    return out, oof, p_pred, p_true


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="both", choices=["a", "b", "both"])
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args()

    Xn, Sn, yn, p_emp = load_termdiv()
    print(f"[fbv2] natural: {Xn.shape[0]} Bernoulli samples "
          f"({len(p_emp)} states, pos={yn.mean():.3f})")

    if args.mode in ("a", "both"):
        out_a, oof, p_pred, p_true = train_eval(
            Xn, Sn, yn, p_emp_state=p_emp, epochs=args.epochs,
            tag="natural-only")
        # 部署模型: 全量自然数据训练(Reachability 阶段用)
        model = SuccessModel(n_skills=len(SKILL_NAMES)).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        Xt = torch.from_numpy(Xn).to(DEVICE)
        St = torch.from_numpy(Sn).to(DEVICE)
        yt = torch.from_numpy(yn).to(DEVICE)
        for _ in range(args.epochs):
            opt.zero_grad()
            logit = model.net(torch.cat([Xt, St], dim=-1)).squeeze(-1)
            torch.nn.functional.binary_cross_entropy_with_logits(
                logit, yt).backward()
            opt.step()
        path = os.path.join(OUT, "fb_pick-place-v3_v2.pt")
        save(model, path, SKILL_NAMES)
        with open(path.replace(".pt", "_eval.json"), "w") as f:
            json.dump(dict(v1_baseline=dict(fb_gap=-0.250, spearman=-0.391),
                           v2_natural_only=out_a), f, indent=2)
        print(f"[fbv2] saved deploy model {path}")
    if args.mode in ("b", "both"):
        Xd, Sd, yd = load_diag()
        print(f"[fbv2] diag: {Xd.shape[0]} samples (pos={yd.mean():.3f})")
        Xm = np.concatenate([Xn, Xd])
        Sm = np.concatenate([Sn, Sd])
        ym = np.concatenate([yn, yd])
        # 混合训练: 自然部分仍按状态聚合评估(诊断部分不参与自然分布评估)
        train_eval_mixed(Xm, Sm, ym, Xn.shape[0], p_emp,
                         epochs=args.epochs)


def train_eval_mixed(X, S, y, n_natural, p_emp_state, epochs=300, seed=0,
                     n_folds=5):
    """混合训练: CV 划分在自然样本上做 out-of-fold(诊断样本全进训练)。"""
    n = len(y)
    rng = np.random.default_rng(seed)
    idx_n = rng.permutation(n_natural)
    folds = np.array_split(idx_n, n_folds)
    diag_idx = np.arange(n_natural, n)
    oof = np.zeros(n_natural)
    for va in folds:
        tr = np.concatenate([np.setdiff1d(idx_n, va), diag_idx])
        model = SuccessModel(n_skills=len(SKILL_NAMES)).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        Xt = torch.from_numpy(X[tr]).to(DEVICE)
        St = torch.from_numpy(S[tr]).to(DEVICE)
        yt = torch.from_numpy(y[tr]).to(DEVICE)
        for _ in range(epochs):
            opt.zero_grad()
            logit = model.net(torch.cat([Xt, St], dim=-1)).squeeze(-1)
            torch.nn.functional.binary_cross_entropy_with_logits(
                logit, yt).backward()
            opt.step()
        oof[va] = model.predict(X[va], S[va])
    n_state = len(p_emp_state)
    rep = n_natural // n_state
    p_pred = oof.reshape(n_state, rep).mean(axis=1)
    p_true = np.array(p_emp_state)
    kk = max(1, int(n_state * 0.2))
    order_f = np.argsort(p_pred)
    fb_gap = float(p_true[order_f[-kk:]].mean() - p_true[order_f[:kk]].mean())
    sp = stats.spearmanr(p_pred, p_true)
    print(f"[fbv2 mixed] n={n_state}  FB_gap={fb_gap:+.3f}  "
          f"Spearman={sp.statistic:+.3f} (p={sp.pvalue:.4f})  "
          f"ECE={ece(p_pred, p_true):.3f}  top={p_true[order_f[-kk:]].mean():.3f}"
          f"  bottom={p_true[order_f[:kk]].mean():.3f}")


if __name__ == "__main__":
    main()
