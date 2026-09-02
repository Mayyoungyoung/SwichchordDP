"""物理可行性投影 prox_feasible。

Meta-World 状态空间下的凸约束投影(均为非扩张算子, 理论见 SWDP.md 新增定理 1):
1. 动作幅值限幅: 夹爪开合归一化到 [-1, 1](env 内部会 clip, 这里显式保证);
2. 关节速度限幅: 动作块内相邻步的动作差限幅(Δx/Δy/Δz/Δgrip);
3. 工作空间限幅: 末端位移不超过单步最大位移。
所有操作都是 clip/缩放, 为 1-Lipschitz 非扩张映射。
"""
import torch

# Meta-World 动作空间: [dx, dy, dz, grip], 各分量范围
POS_MAX = 1.0      # 单步位移上限(米)
GRIP_MAX = 1.0     # 夹爪动作上限


def action_clip(a: torch.Tensor):
    """动作幅值限幅(盒约束投影)。"""
    return torch.clamp(a, -1.0, 1.0)


def velocity_clip(a: torch.Tensor, max_step: float = 0.12):
    """动作块内相邻步速度限幅: |a[t+1] - a[t]| <= max_step(除最后一维夹爪)。"""
    out = a.clone()
    if a.shape[1] < 2:
        return out
    pos = out[..., :-1]
    grip = out[..., -1:]
    diff = pos[:, 1:] - pos[:, :-1]
    diff = torch.clamp(diff, -max_step, max_step)
    for t in range(1, pos.shape[1]):
        pos[:, t] = pos[:, t - 1] + diff[:, t - 1]
    out = torch.cat([pos, grip], dim=-1)
    return out


def smooth(a: torch.Tensor, k: int = 3):
    """滑动平均平滑(卷积核非负且和为 1 -> 非扩张)。"""
    if k <= 1 or a.shape[1] < k:
        return a
    pad = k // 2
    ap = torch.nn.functional.pad(a, (0, 0, pad, pad), mode="replicate")
    c = a.shape[-1]
    kernel = torch.ones(c, 1, k, device=a.device) / k
    out = torch.nn.functional.conv1d(
        ap.transpose(1, 2), kernel, groups=c
    ).transpose(1, 2)
    return out


def prox_feasible(a: torch.Tensor, max_step: float = 0.12, smooth_k: int = 3):
    """单步可行性投影: 幅值限幅 -> 速度限幅 -> 平滑。"""
    a = action_clip(a)
    a = velocity_clip(a, max_step=max_step)
    a = smooth(a, k=smooth_k)
    return action_clip(a)
