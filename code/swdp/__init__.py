"""SWDP 核心库: 技能条件扩散策略 + ChordCompose 免训组合。

模块:
- nets: 技能条件扩散网络(FiLM)
- policy: DDPM 扩散策略(训练/采样/Q 查询/B_t 映射)
- chord_compose: ChordCompose 算法(Switch/Chain/Combine + 时间掩码 + 可行性投影)
- feasibility: 物理可行性投影
- distil: 一致性蒸馏(1/2/4 步策略)
"""
from . import nets, policy, chord_compose, feasibility, distil  # noqa: F401
