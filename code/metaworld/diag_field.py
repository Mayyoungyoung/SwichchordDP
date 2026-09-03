"""诊断: x0 空间(1 步学生)chord vs naive 结果完全相同的根因。

量化链路每一环:
1. 场幅值 |u| vs 编辑需要跨越的 gap ‖anchor−target‖
2. 编辑后与目标的距离(编辑是否有用/有害/无效)
3. 反归一化+clip 后的饱和率(编辑是否被执行截断)
4. R(tau) 与 R(tau-delta) 的相关性(平滑是否有物可平滑)
5. 教师eps空间对照
"""
import os
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from swdp.policy import SkillDP          # noqa: E402
from swdp.distil import ConsistencyStudent  # noqa: E402
from swdp import chord_compose as cc     # noqa: E402
from swdp.feasibility import prox_feasible  # noqa: E402
from swdp.nets import SIGMA, ALPHA       # noqa: E402

DEVICE = "cuda"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "../../results/metaworld/data/pick-place-v3.h5")
MODEL = os.path.join(HERE, "../../results/metaworld/models")
H, NS = 16, 5


def onehot(i):
    z = np.zeros((1, NS), dtype=np.float32)
    z[0, i] = 1.0
    return torch.from_numpy(z).to(DEVICE)


def main():
    with h5py.File(DATA, "r") as f:
        obs = f["obs"][:]
        act = f["action"][:]
        skill = f["skill"][:]
        obs_mean, obs_std = f["obs_mean"][:], f["obs_std"][:]
        act_mean, act_std = f["act_mean"][:], f["act_std"][:]
    print(f"data: {len(obs)} frames, act_std = {np.round(act_std, 3)}")

    teacher = SkillDP.load(os.path.join(MODEL, "dp_pick-place-v3.pt"), DEVICE).eval()
    ckpt = torch.load(os.path.join(MODEL, "dp_pick-place-v3_cd.pt"),
                      map_location=DEVICE)
    student = ConsistencyStudent(**ckpt["cfg"], device=DEVICE)
    student.load_state_dict(ckpt["model"])
    student.eval()

    # 技能边界索引(演示数据中 skill 变化处, 且前后各留 H 步)
    bounds = [i + 1 for i in range(H, len(skill) - H - 1)
              if skill[i] != skill[i + 1]]
    rng = np.random.default_rng(0)
    sel = rng.choice(bounds, size=min(40, len(bounds)), replace=False)

    names = ["reach", "grasp", "lift", "carry", "place"]
    stats = {k: [] for k in
             ["gap4", "u_chord", "u_naive", "corr",
              "edit4_s", "err4_s_base", "err4_s_chord", "err4_s_naive",
              "sat_chord", "err4_t_base", "err4_t_chord", "err4_t_naive",
              "u_teacher"]}
    for b in sel:
        s_from, s_to = onehot(int(skill[b - 1])), onehot(int(skill[b]))
        o = torch.from_numpy((obs[b] - obs_mean) / obs_std).float().to(DEVICE).unsqueeze(0)
        target = torch.from_numpy(act[b:b + H]).float().to(DEVICE).unsqueeze(0)

        for tag, dp, x0s in [("s", student, True), ("t", teacher, False)]:
            anchor = dp.sample(o, s_from, n_steps=(1 if x0s else 24))
            mask = cc.temporal_mask(H, 0, 4, DEVICE)
            # 场(chord / naive)
            u_c, _ = cc.chord_field(dp, anchor, 0.9, 0.15, o, s_from, s_to,
                                    1, 42, "chord", x0_space=x0s)
            u_n, _ = cc.chord_field(dp, anchor, 0.9, 0.15, o, s_from, s_to,
                                    1, 42, "naive", x0_space=x0s)
            if tag == "s":
                stats["gap4"].append(float((anchor[0, :4] - target[0, :4]).norm()))
                stats["u_chord"].append(float(u_c[0, :4].norm()))
                stats["u_naive"].append(float(u_n[0, :4].norm()))
                r_tau = u_n[0, :4]
                r_prev, _ = cc.chord_field(dp, anchor, 0.75, 0.15, o, s_from,
                                           s_to, 1, 42, "naive", x0_space=x0s)
                cos = torch.nn.functional.cosine_similarity(
                    r_tau.flatten(), r_prev[0, :4].flatten(), dim=0)
                stats["corr"].append(float(cos))
            else:
                stats["u_teacher"].append(float(u_c[0, :4].norm()))
            lam = 0.6 if x0s else 0.3
            base4 = float((anchor[0, :4] - target[0, :4]).norm())
            stats[f"err4_{tag}_base"].append(base4)
            for m, u in [("chord", u_c), ("naive", u_n)]:
                a_new = prox_feasible(anchor + lam * u * mask)
                stats[f"err4_{tag}_{m}"].append(
                    float((a_new[0, :4] - target[0, :4]).norm()))
                if tag == "s" and m == "chord":
                    stats["edit4_s"].append(
                        float((a_new[0, :4] - anchor[0, :4]).norm()))
                    raw = (a_new[0, :4].cpu().numpy() * act_std) + act_mean
                    stats["sat_chord"].append(
                        float(np.mean(np.abs(raw) >= 0.999)))

    def ms(k):
        return f"{np.mean(stats[k]):.3f}"

    print("\n=== 学生(x0 空间, λ=0.6) ===")
    print(f"gap(需跨越)          : {ms('gap4')}")
    print(f"|u_chord| / |u_naive|: {ms('u_chord')} / {ms('u_naive')}")
    print(f"cos(R(tau), R(tau-d)): {ms('corr')}")
    print(f"proj后实际编辑量      : {ms('edit4_s')}  (饱和率 {ms('sat_chord')})")
    print(f"err: base {ms('err4_s_base')} -> chord {ms('err4_s_chord')} "
          f"/ naive {ms('err4_s_naive')}")
    print("\n=== 教师(eps 空间, λ=0.3) ===")
    print(f"|u_chord|: {ms('u_teacher')}")
    print(f"err: base {ms('err4_t_base')} -> chord {ms('err4_t_chord')} "
          f"/ naive {ms('err4_t_naive')}")


if __name__ == "__main__":
    main()
