"""F_B: Future Success Predictor — P(B succeeds | s) 轻量 MLP。

阶段 A 诊断判读 SENSITIVE(连续类扰动 max_drop>=0.25) -> 阶段 C 的调度信号。
数据: diag_handoff.py 的 rows (s, B, y) 三元组, 扰动覆盖真实边界散布
(hand/grip/puck/goal/skill/success 字段)。

特征(19 维手工几何 + 技能 one-hot):
    [hand(3), grip(1), puck(3), goal(3),
     hand-puck(3), hand-goal(3), puck-goal(3)]
绝对坐标提供桌面系上下文, 三个相对量提供平移不变的技能几何
(诊断显示方向不对称 2x, 保留带符号相对量)。

评估: 5 折 CV AUC + ECE(10 bin, 目标 < 0.1) + 按扰动维度分桶 AUC。
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

FEAT_NAMES = ["hand", "grip", "puck", "goal",
              "hand-puck", "hand-goal", "puck-goal"]
FEAT_DIM = 19


def featurize(hand, grip, puck, goal) -> np.ndarray:
    """19 维几何特征(输入均可为 np 数组或 list)。"""
    hand, puck, goal = map(np.asarray, (hand, puck, goal))
    return np.concatenate([
        hand, [float(grip)], puck, goal,
        hand - puck, hand - goal, puck - goal,
    ]).astype(np.float32)


class SuccessModel(nn.Module):
    """F_theta: (feat, skill_onehot) -> P(success)。"""

    def __init__(self, feat_dim: int = FEAT_DIM, n_skills: int = 5,
                 hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim + n_skills, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, feat: torch.Tensor, skill: torch.Tensor) -> torch.Tensor:
        """feat [B, feat_dim], skill one-hot [B, n_skills] -> [B] 概率。"""
        logit = self.net(torch.cat([feat, skill], dim=-1)).squeeze(-1)
        return torch.sigmoid(logit)

    def predict(self, feat: np.ndarray, skill_onehot: np.ndarray) -> np.ndarray:
        """numpy 便捷接口(单样本或批量)。"""
        dev = next(self.parameters()).device
        with torch.no_grad():
            f = torch.from_numpy(np.atleast_2d(feat).astype(np.float32)).to(dev)
            s = torch.from_numpy(
                np.atleast_2d(skill_onehot).astype(np.float32)).to(dev)
            return self.forward(f, s).cpu().numpy().squeeze()


def build_dataset(rows, skill_names):
    """诊断 rows -> (X, y, skill_ids, kinds)。"""
    sid = {n: i for i, n in enumerate(skill_names)}
    X, y, sids, kinds = [], [], [], []
    for r in rows:
        if "goal" not in r:
            raise ValueError("rows 缺 goal 字段(需 diag_handoff.py 新版数据)")
        X.append(featurize(r["hand"], r["grip"], r["puck"], r["goal"]))
        y.append(float(r["success"]))
        sids.append(sid[r["skill"]])
        kinds.append(r["kind"])
    X = np.stack(X)
    sids = np.array(sids)
    S = np.eye(len(skill_names))[sids].astype(np.float32)
    return X, np.array(y, dtype=np.float32), S, sids, kinds


def ece(probs, ys, n_bins: int = 10) -> float:
    """Expected Calibration Error(等频 bin)。"""
    probs, ys = np.asarray(probs), np.asarray(ys)
    edges = np.linspace(0, 1, n_bins + 1)
    e, tot = 0.0, len(probs)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (probs >= lo) & (probs < hi if hi < 1 else probs <= hi)
        if m.sum() == 0:
            continue
        e += m.sum() / tot * abs(probs[m].mean() - ys[m].mean())
    return float(e)


def auc(probs, ys) -> float:
    """AUC(秩统计实现, 免依赖 sklearn)。"""
    probs, ys = np.asarray(probs), np.asarray(ys)
    if len(np.unique(ys)) < 2:
        return float("nan")
    order = np.argsort(probs)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(probs) + 1)
    # 并列秩取平均
    for v in np.unique(probs):
        m = probs == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    n1, n0 = ys.sum(), (1 - ys).sum()
    return float((ranks[ys == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def train_cv(X, y, S, n_skills, k_folds=5, epochs=300, lr=1e-3,
             weight_decay=1e-4, seed=0, device="cpu", verbose=True):
    """k 折交叉验证训练(返回每折指标 + 全量重训模型)。"""
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k_folds)
    metrics = []
    for k, va in enumerate(folds):
        tr = np.setdiff1d(idx, va)
        model = SuccessModel(n_skills=n_skills).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr,
                               weight_decay=weight_decay)
        Xt = torch.from_numpy(X[tr]).to(device)
        St = torch.from_numpy(S[tr]).to(device)
        yt = torch.from_numpy(y[tr]).to(device)
        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            logit = model.net(torch.cat([Xt, St], dim=-1)).squeeze(-1)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logit, yt)
            loss.backward()
            opt.step()
        model.eval()
        p = model.predict(X[va], S[va])
        metrics.append(dict(
            fold=k, n_train=len(tr), n_val=len(va),
            auc=auc(p, y[va]), ece=ece(p, y[va]),
            brier=float(np.mean((p - y[va]) ** 2)),
            base_rate=float(y[va].mean())))
        if verbose:
            m = metrics[-1]
            print(f"[fb] fold{k}: AUC={m['auc']:.3f} ECE={m['ece']:.3f} "
                  f"Brier={m['brier']:.3f} (base={m['base_rate']:.2f})")
    # 全量重训(部署用)
    model = SuccessModel(n_skills=n_skills).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    Xt, St, yt = (torch.from_numpy(X).to(device), torch.from_numpy(S).to(device),
                  torch.from_numpy(y).to(device))
    for _ in range(epochs):
        opt.zero_grad()
        logit = model.net(torch.cat([Xt, St], dim=-1)).squeeze(-1)
        nn.functional.binary_cross_entropy_with_logits(logit, yt).backward()
        opt.step()
    model.eval()
    return model, metrics


def bucket_auc_by_kind(X, y, S, sids, kinds, model, skill_names):
    """按扰动维度分桶的预测力(AUC), 检验 F_B 学到的是几何而非噪声。"""
    out = {}
    p = model.predict(X, S)
    for kind in dict.fromkeys(kinds):
        m = np.array([c == kind for c in kinds])
        out[kind] = dict(n=int(m.sum()), auc=auc(p[m], y[m]))
    return out


def save(model, path, skill_names):
    torch.save({"model": model.state_dict(),
                "skill_names": skill_names,
                "feat_dim": FEAT_DIM}, path)


def load(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    m = SuccessModel(feat_dim=ckpt["feat_dim"],
                     n_skills=len(ckpt["skill_names"])).to(device)
    m.load_state_dict(ckpt["model"])
    m.eval()
    return m, ckpt["skill_names"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../../results/metaworld/eval/"
                                      "diag_handoff.json")
    ap.add_argument("--out", default="../../results/metaworld/models/"
                                     "fb_pick-place-v3.pt")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.data) as f:
        d = json.load(f)
    skill_names = ["reach", "grasp", "lift", "carry", "place"]
    X, y, S, sids, kinds = build_dataset(d["rows"], skill_names)
    print(f"[fb] dataset: {X.shape[0]} rows, pos_rate={y.mean():.3f}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, metrics = train_cv(X, y, S, len(skill_names), epochs=args.epochs,
                              seed=args.seed, device=device)
    mean = {k: float(np.nanmean([m[k] for m in metrics]))
            for k in ("auc", "ece", "brier")}
    buckets = bucket_auc_by_kind(X, y, S, sids, kinds, model, skill_names)
    report = dict(n_rows=int(X.shape[0]), pos_rate=float(y.mean()),
                  cv_mean=mean, cv_folds=metrics, bucket_auc=buckets,
                  skill_names=skill_names)
    print(f"[fb] CV mean: AUC={mean['auc']:.3f} ECE={mean['ece']:.3f} "
          f"Brier={mean['brier']:.3f}")
    for k, v in buckets.items():
        print(f"[fb]   {k}: AUC={v['auc']:.3f} (n={v['n']})")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save(model, args.out, skill_names)
    with open(args.out.replace(".pt", "_metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[fb] saved {args.out}")

    # 可靠性图 + 分桶 AUC 图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        p_full = model.predict(X, S)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        ax = axes[0]
        edges = np.linspace(0, 1, 11)
        mids, obs = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (p_full >= lo) & (p_full < hi if hi < 1 else p_full <= hi)
            if m.sum() == 0:
                continue
            mids.append(p_full[m].mean())
            obs.append(y[m].mean())
        ax.plot([0, 1], [0, 1], "k:", lw=1)
        ax.plot(mids, obs, "o-")
        ax.set_xlabel("F_B 预测 P(success)")
        ax.set_ylabel("实测频率")
        ax.set_title(f"可靠性图 (ECE={mean['ece']:.3f})")
        ax = axes[1]
        names = list(buckets)
        ax.bar(names, [buckets[n]["auc"] for n in names], color="tab:blue")
        ax.axhline(mean["auc"], color="r", ls=":",
                   label=f"CV AUC={mean['auc']:.3f}")
        ax.set_ylim(0.5, 1.0)
        ax.set_ylabel("AUC")
        ax.set_title("按扰动维度分桶预测力")
        ax.legend()
        fig.tight_layout()
        png = args.out.replace(".pt", "_calib.png")
        fig.savefig(png, dpi=130)
        plt.close(fig)
        print(f"[fb] saved {png}")
    except Exception as e:  # noqa: BLE001
        print(f"[fb] plot failed: {e}")


if __name__ == "__main__":
    main()
